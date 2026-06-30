"""Main extractor: HTML in, markdown/text out."""

from __future__ import annotations

import torch

from pulpie.chunker import extract_blocks, pack_chunks, tokenize_blocks
from pulpie.model_utils import (
    default_device,
    extract_item_ids,
    load_model_and_tokenizer,
    predictions_to_labels,
    resolve_model_id,
)
from pulpie.reconstruct import extract_main_html
from pulpie.simplify import simplify

DEFAULT_MODEL = "orange-small"


class Extractor:
    """Extract main content from raw HTML.

    Usage:
        from pulpie import Extractor

        extractor = Extractor()
        result = extractor.extract(html)
        print(result.markdown)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        max_tokens: int = 8192,
    ):
        if device is None:
            device = default_device()
        self.device = torch.device(device)
        self.max_tokens = max_tokens

        model_id = resolve_model_id(model)
        self.model, self.tokenizer, self.sep_token_id = load_model_and_tokenizer(
            model_id, self.device
        )

    def extract(self, html: str) -> ExtractionResult:
        """Extract main content from raw HTML."""
        simplified, map_html = simplify(html)
        labels = self._classify(simplified)
        main_html = extract_main_html(map_html, labels)
        markdown = self._to_markdown(main_html)

        return ExtractionResult(
            html=main_html,
            markdown=markdown,
            labels=labels,
        )

    def extract_from_simplified(
        self, simplified_html: str, map_html: str | None = None
    ) -> ExtractionResult:
        """Extract from pre-simplified HTML (skip simplify step)."""
        labels = self._classify(simplified_html)
        source = map_html if map_html is not None else simplified_html
        main_html = extract_main_html(source, labels)
        markdown = self._to_markdown(main_html)

        return ExtractionResult(
            html=main_html,
            markdown=markdown,
            labels=labels,
        )

    @torch.no_grad()
    def _classify(self, simplified_html: str) -> dict[str, str]:
        """Classify each block as main/other."""
        blocks = extract_blocks(simplified_html)
        if not blocks:
            return {}

        item_ids = extract_item_ids(blocks)
        block_token_ids = tokenize_blocks(blocks, self.tokenizer)
        chunks = pack_chunks(
            block_token_ids,
            max_tokens=self.max_tokens,
            sep_token_id=self.sep_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        predictions = [0] * len(blocks)

        for chunk_ids, block_indices in chunks:
            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]

            sep_positions = (input_ids[0] == self.sep_token_id).nonzero(as_tuple=True)[0]
            preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()

            for i, block_idx in enumerate(block_indices):
                if i < len(preds):
                    predictions[block_idx] = preds[i]

        return predictions_to_labels(item_ids, predictions)

    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        try:
            import html2text

            h = html2text.HTML2Text(bodywidth=0)
            h.ignore_links = False
            h.ignore_images = False
            return h.handle(html).strip()
        except ImportError:
            return html


class ExtractionResult:
    """Result of content extraction."""

    __slots__ = ("html", "labels", "markdown")

    def __init__(self, html: str, markdown: str, labels: dict[str, str]):
        self.html = html
        self.markdown = markdown
        self.labels = labels

    @property
    def n_blocks(self) -> int:
        return len(self.labels)

    @property
    def n_main(self) -> int:
        return sum(1 for v in self.labels.values() if v == "main")

    @property
    def n_other(self) -> int:
        return sum(1 for v in self.labels.values() if v == "other")

    def __repr__(self):
        return (
            f"ExtractionResult(blocks={self.n_blocks}, "
            f"main={self.n_main}, other={self.n_other}, "
            f"markdown_len={len(self.markdown)})"
        )
