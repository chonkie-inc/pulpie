"""Shared model loading and configuration utilities."""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pulpie.chunker import SEP_TOKEN

MODELS = {
    "orange-small": "chonkie-ai/pulpie-orange-small",
    "orange-base": "chonkie-ai/pulpie-orange-base",
    "orange-large": "chonkie-ai/pulpie-orange-large",
}

ITEM_ID_PATTERN = re.compile(r'_item_id="(\d+)"')


def resolve_model_id(model: str) -> str:
    """Resolve model name to HuggingFace model ID."""
    return MODELS.get(model, model)


def default_device() -> str:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_id: str, device: torch.device
) -> tuple[torch.nn.Module, AutoTokenizer, int]:
    """Load model and tokenizer with standard configuration.

    Returns:
        Tuple of (model, tokenizer, sep_token_id).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    model = (
        AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=2,
            trust_remote_code=True,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa" if device.type == "cuda" else "eager",
        )
        .to(device)
        .eval()
    )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    fix_rotary_embeddings(model, device)

    return model, tokenizer, sep_token_id


def fix_rotary_embeddings(model: torch.nn.Module, device: torch.device) -> None:
    """Recompute rotary embedding inv_freq after model load.

    EuroBERT registers inv_freq as a non-persistent buffer that doesn't
    survive from_pretrained weight loading in newer transformers.
    """
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", 250000.0)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    rotary = model.model.rotary_emb
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq


def extract_item_ids(blocks: list[str]) -> list[str | None]:
    """Extract _item_id values from HTML block strings."""
    item_ids = []
    for block in blocks:
        m = ITEM_ID_PATTERN.search(block)
        item_ids.append(m.group(1) if m else None)
    return item_ids


def predictions_to_labels(item_ids: list[str | None], predictions: list[int]) -> dict[str, str]:
    """Convert model predictions to label dictionary."""
    labels = {}
    for idx, item_id in enumerate(item_ids):
        if item_id is not None:
            labels[item_id] = "main" if predictions[idx] == 1 else "other"
    return labels
