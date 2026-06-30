"""Main extractor: HTML in, markdown/text out."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pulpie.chunker import extract_blocks, pack_chunks, tokenize_blocks
from pulpie.markdown import to_markdown
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
        max_batch_tokens: int = 16384,
    ):
        if device is None:
            device = default_device()
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.max_batch_tokens = max_batch_tokens

        model_id = resolve_model_id(model)
        self.model, self.tokenizer, self.sep_token_id = load_model_and_tokenizer(
            model_id, self.device
        )
        self.pad_id = self.model.config.pad_token_id or 0

    def extract(self, html: str) -> ExtractionResult:
        """Extract main content from raw HTML."""
        simplified, map_html = simplify(html)
        labels = self._classify(simplified)
        main_html = extract_main_html(map_html, labels)

        return ExtractionResult(
            html=main_html,
            markdown=to_markdown(main_html),
            labels=labels,
        )

    def extract_from_simplified(
        self, simplified_html: str, map_html: str | None = None
    ) -> ExtractionResult:
        """Extract from pre-simplified HTML (skip simplify step)."""
        labels = self._classify(simplified_html)
        source = map_html if map_html is not None else simplified_html
        main_html = extract_main_html(source, labels)

        return ExtractionResult(
            html=main_html,
            markdown=to_markdown(main_html),
            labels=labels,
        )

    def extract_batch(self, htmls: list[str]) -> list[ExtractionResult]:
        """Extract many pages with one call. Returns results in input order.

        A convenience wrapper over :meth:`extract` that packs chunks from
        different pages into shared forward passes. Note: with this model's eager
        attention it is not meaningfully faster than calling :meth:`extract` in a
        loop (a single pass already saturates the GPU); it's offered for
        ergonomics. For large-scale, multi-GPU throughput use
        :class:`pulpie.Pipeline`.
        """
        prepared = []  # (item_ids, n_blocks, map_html, [(chunk_ids, block_indices), ...])
        all_chunks = []  # (page_idx, chunk_ids, block_indices)
        for page_idx, html in enumerate(htmls):
            simplified, map_html = simplify(html)
            blocks = extract_blocks(simplified)
            item_ids = extract_item_ids(blocks)
            chunks = self._chunk(blocks)
            prepared.append((item_ids, len(blocks), map_html))
            for chunk_ids, block_indices in chunks:
                all_chunks.append((page_idx, chunk_ids, block_indices))

        predictions = [[0] * n for _, n, _ in prepared]
        self._run_batched(all_chunks, predictions)

        results = []
        for (item_ids, _n, map_html), preds in zip(prepared, predictions):
            labels = predictions_to_labels(item_ids, preds)
            main_html = extract_main_html(map_html, labels)
            results.append(
                ExtractionResult(html=main_html, markdown=to_markdown(main_html), labels=labels)
            )
        return results

    def _chunk(self, blocks: list[str]) -> list[tuple[list[int], list[int]]]:
        """Tokenize blocks and pack them into model-sized chunks."""
        block_token_ids = tokenize_blocks(blocks, self.tokenizer)
        return pack_chunks(
            block_token_ids,
            max_tokens=self.max_tokens,
            sep_token_id=self.sep_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    @torch.no_grad()
    def _classify(self, simplified_html: str) -> dict[str, str]:
        """Classify each block of a single page as main/other (sequential)."""
        blocks = extract_blocks(simplified_html)
        if not blocks:
            return {}

        item_ids = extract_item_ids(blocks)
        predictions = [0] * len(blocks)
        for chunk_ids, block_indices in self._chunk(blocks):
            input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            sep_positions = (input_ids[0] == self.sep_token_id).nonzero(as_tuple=True)[0]
            preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()
            for i, block_idx in enumerate(block_indices):
                if i < len(preds):
                    predictions[block_idx] = preds[i]

        return predictions_to_labels(item_ids, predictions)

    @torch.no_grad()
    def _run_batched(self, all_chunks, predictions) -> None:
        """Run chunks (tagged with their page index) in length-sorted, padded batches.

        Memory for eager attention scales as ``batch * max_len^2``, so the batch
        size is capped by the squared longest length to stay safe on long chunks
        while grouping short ones (the common case across many single-chunk pages).
        """
        if not all_chunks:
            return
        budget = self.max_batch_tokens * self.max_batch_tokens
        ordered = sorted(all_chunks, key=lambda c: len(c[1]))

        batch: list = []
        for item in ordered:
            cand = batch + [item]
            max_len = len(cand[-1][1])  # ascending -> last is longest
            if batch and len(cand) * max_len * max_len > budget:
                self._infer_batch(batch, predictions)
                batch = [item]
            else:
                batch = cand
        if batch:
            self._infer_batch(batch, predictions)

    def _infer_batch(self, batch, predictions) -> None:
        """Run one padded batch of (page_idx, chunk_ids, block_indices); write preds."""
        max_len = max(len(c[1]) for c in batch)
        input_ids = torch.full(
            (len(batch), max_len), self.pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=self.device)
        for row, (_page_idx, chunk_ids, _bi) in enumerate(batch):
            input_ids[row, : len(chunk_ids)] = torch.tensor(chunk_ids, dtype=torch.long)
            attention_mask[row, : len(chunk_ids)] = 1

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        for row, (page_idx, _chunk_ids, block_indices) in enumerate(batch):
            sep_positions = (input_ids[row] == self.sep_token_id).nonzero(as_tuple=True)[0]
            preds = logits[row][sep_positions].argmax(dim=-1).cpu().tolist()
            for j, block_idx in enumerate(block_indices):
                if j < len(preds):
                    predictions[page_idx][block_idx] = preds[j]


@dataclass
class ExtractionResult:
    """Result of content extraction."""

    html: str
    markdown: str
    labels: dict[str, str]

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
