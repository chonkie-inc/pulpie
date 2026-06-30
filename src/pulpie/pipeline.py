"""Fully overlapped multi-GPU extraction pipeline.

Architecture:
    [CPU Pool: simplify+chunk] → Shared Queue → [N GPU threads] → Shared Queue → [CPU Pool: reconstruct+md]

Each GPU thread pulls from the same prep queue, accumulates a local batch,
sorts by length, runs inference on its device. Whichever GPU finishes first
pulls the next batch — natural load balancing.
"""

from __future__ import annotations

import math
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import torch

from pulpie.chunker import SEP_TOKEN, extract_blocks, pack_chunks, tokenize_blocks
from pulpie.model_utils import (
    default_device,
    extract_item_ids,
    load_model_and_tokenizer,
    predictions_to_labels,
    resolve_model_id,
)
from pulpie.reconstruct import extract_main_html
from pulpie.simplify import simplify


@dataclass
class PageInput:
    """Input to the pipeline."""

    html: str
    page_id: int = 0
    metadata: dict | None = None


@dataclass
class PageResult:
    """Output from the pipeline."""

    page_id: int
    labels: dict[str, str]
    html: str
    markdown: str
    error: str | None = None


@dataclass
class _PreparedPage:
    """Internal: CPU-prepared page ready for GPU inference."""

    page_id: int
    batch_idx: int
    chunks: list[tuple[list[int], list[int]]]
    item_ids: list[str | None]
    n_blocks: int
    map_html: str
    error: str | None = None


# ── Worker state (initialized once per process) ──

_worker_tokenizer = None
_worker_sep_token_id = None


def _init_worker(tokenizer_path: str) -> None:
    """Initialize tokenizer once per worker process."""
    global _worker_tokenizer, _worker_sep_token_id

    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if SEP_TOKEN not in _worker_tokenizer.get_vocab():
        _worker_tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})
    _worker_sep_token_id = _worker_tokenizer.convert_tokens_to_ids(SEP_TOKEN)


def _cpu_prepare(
    html: str,
    page_id: int,
    batch_idx: int,
    max_tokens: int,
    cutoff_length: int,
) -> _PreparedPage:
    """CPU worker: simplify + tokenize + chunk a single page."""
    try:
        simplified, map_html = simplify(html, cutoff_length=cutoff_length)
    except Exception as e:
        return _PreparedPage(
            page_id=page_id,
            batch_idx=batch_idx,
            chunks=[],
            item_ids=[],
            n_blocks=0,
            map_html="",
            error=f"simplify failed: {e}",
        )

    blocks = extract_blocks(simplified)
    if not blocks:
        return _PreparedPage(
            page_id=page_id,
            batch_idx=batch_idx,
            chunks=[],
            item_ids=[],
            n_blocks=0,
            map_html=map_html,
        )

    item_ids = extract_item_ids(blocks)
    assert _worker_tokenizer is not None
    block_token_ids = tokenize_blocks(blocks, _worker_tokenizer)
    chunks = pack_chunks(
        block_token_ids,
        max_tokens=max_tokens,
        sep_token_id=_worker_sep_token_id,
        bos_token_id=_worker_tokenizer.bos_token_id,
        eos_token_id=_worker_tokenizer.eos_token_id,
    )

    return _PreparedPage(
        page_id=page_id,
        batch_idx=batch_idx,
        chunks=chunks,
        item_ids=item_ids,
        n_blocks=len(blocks),
        map_html=map_html,
    )


def _postprocess(
    page_id: int, batch_idx: int, labels: dict[str, str], map_html: str
) -> tuple[int, PageResult]:
    """CPU post-processing: reconstruct HTML + convert to markdown."""
    main_html = extract_main_html(map_html, labels)

    try:
        import html2text

        h = html2text.HTML2Text(bodywidth=0)
        h.ignore_links = False
        h.ignore_images = False
        markdown = h.handle(main_html).strip()
    except ImportError:
        markdown = main_html

    return batch_idx, PageResult(
        page_id=page_id,
        labels=labels,
        html=main_html,
        markdown=markdown,
    )


_SENTINEL = None

# Per-GPU throughput and per-worker throughput for auto-sizing
_GPU_PPS = 14.9
_PRE_WORKER_PPS = 21.0
_POST_WORKER_PPS = 43.0


class Pipeline:
    """Multi-GPU overlapped HTML content extraction pipeline.

    All three stages run concurrently via shared queues:
      1. CPU process pool → simplify + tokenize + chunk
      2. N GPU threads → each pulls from shared queue, batches, infers
      3. CPU process pool → reconstruct + markdown

    GPUs self-balance: whichever finishes first pulls the next batch.

    Usage:
        from pulpie import Pipeline, PageInput

        # Auto-detect all GPUs
        pipeline = Pipeline()

        # Specific GPUs
        pipeline = Pipeline(devices=["cuda:0", "cuda:2", "cuda:5"])

        # Single GPU (backward compatible)
        pipeline = Pipeline(devices=["cuda:0"])

        results = pipeline.extract_batch([
            PageInput(html=page1, page_id=0),
            PageInput(html=page2, page_id=1),
        ])
    """

    def __init__(
        self,
        model: str = "orange-small",
        devices: list[str] | str | None = None,
        n_pre_workers: int | None = None,
        n_post_workers: int | None = None,
        max_tokens: int = 8192,
        cutoff_length: int = 500,
        max_batch_tokens: int = 16384,
    ):
        # Resolve devices
        if devices is None:
            if torch.cuda.is_available():
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                devices = [default_device()]  # "mps" on Apple, else "cpu"
        elif isinstance(devices, str):
            devices = [devices]
        self.devices = [torch.device(d) for d in devices]
        self.n_gpus = len(self.devices)

        # Auto-size worker pools based on GPU count
        gpu_capacity = self.n_gpus * _GPU_PPS
        self.n_pre_workers = n_pre_workers or max(2, math.ceil(gpu_capacity / _PRE_WORKER_PPS))
        self.n_post_workers = n_post_workers or max(2, math.ceil(gpu_capacity / _POST_WORKER_PPS))

        self.max_tokens = max_tokens
        self.cutoff_length = cutoff_length
        self.max_batch_tokens = max_batch_tokens

        self.model_id = resolve_model_id(model)

        # Load model on each device
        self._gpu_models: list[_GPUWorkerState] = []
        for device in self.devices:
            model_inst, _tokenizer, sep_token_id = load_model_and_tokenizer(self.model_id, device)
            self._gpu_models.append(
                _GPUWorkerState(
                    model=model_inst,
                    sep_token_id=sep_token_id,
                    device=device,
                    pad_id=model_inst.config.pad_token_id or 0,
                )
            )

    def extract_batch(self, pages: list[PageInput]) -> list[PageResult]:
        """Extract content from a batch of pages with multi-GPU overlap.

        Returns results in the same order as input.
        """
        n_pages = len(pages)
        results: list[PageResult | None] = [None] * n_pages

        # Shared queues
        prep_queue: queue.Queue = queue.Queue(maxsize=self.n_pre_workers * 8)
        post_queue: queue.Queue = queue.Queue(maxsize=64)

        # Stage 3: post-processing (process pool)
        post_thread = threading.Thread(
            target=self._stage3_postprocess,
            args=(post_queue, results),
            daemon=True,
        )
        post_thread.start()

        # Stage 2: one GPU consumer thread per device
        gpu_threads = []
        for gpu_state in self._gpu_models:
            t = threading.Thread(
                target=self._stage2_gpu_worker,
                args=(gpu_state, prep_queue, post_queue),
                daemon=True,
            )
            t.start()
            gpu_threads.append(t)

        # Stage 1: CPU producers
        self._stage1_cpu(pages, prep_queue)

        # Signal all GPU threads to stop (one sentinel per thread)
        for _ in gpu_threads:
            prep_queue.put(_SENTINEL)

        # Wait for all GPUs to finish
        for t in gpu_threads:
            t.join()

        # Signal post-processing to stop
        post_queue.put(_SENTINEL)
        post_thread.join()

        # Fill gaps
        for i in range(n_pages):
            if results[i] is None:
                results[i] = PageResult(page_id=pages[i].page_id, labels={}, html="", markdown="")

        return results  # type: ignore[return-value]

    def _stage1_cpu(self, pages: list[PageInput], out_queue: queue.Queue) -> None:
        """Stage 1: parallel CPU simplify+chunk, push to shared queue as completed."""
        with ProcessPoolExecutor(
            max_workers=self.n_pre_workers,
            initializer=_init_worker,
            initargs=(self.model_id,),
        ) as executor:
            futures = [
                executor.submit(
                    _cpu_prepare,
                    page.html,
                    page.page_id,
                    idx,
                    self.max_tokens,
                    self.cutoff_length,
                )
                for idx, page in enumerate(pages)
            ]
            for future in as_completed(futures):
                out_queue.put(future.result())

    @staticmethod
    @torch.no_grad()
    def _stage2_gpu_worker(
        state: _GPUWorkerState,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
    ) -> None:
        """Stage 2: single GPU consumer — pull from shared queue, batch, infer."""
        pending: list[_PreparedPage] = []

        while True:
            # Accumulate pages
            while True:
                try:
                    timeout = 0.05 if len(pending) < 4 else 0.001
                    item = in_queue.get(timeout=timeout)
                    if item is _SENTINEL:
                        # Process remaining, then exit
                        if pending:
                            _infer_and_push(state, pending, out_queue)
                        return
                    if item.error or not item.chunks:
                        out_queue.put((item.page_id, item.batch_idx, {}, item.map_html))
                    else:
                        pending.append(item)
                except queue.Empty:
                    break

            if pending:
                _infer_and_push(state, pending, out_queue)
                pending.clear()

    def _stage3_postprocess(self, in_queue: queue.Queue, results: list[PageResult | None]) -> None:
        """Stage 3: reconstruct + markdown in separate process pool."""
        with ProcessPoolExecutor(max_workers=self.n_post_workers) as pool:
            pending_futures: dict = {}

            while True:
                try:
                    item = in_queue.get(timeout=0.01)
                except queue.Empty:
                    _collect_futures(pending_futures, results)
                    continue

                if item is _SENTINEL:
                    break

                if item is None:
                    continue

                page_id, batch_idx, labels, map_html = item
                future = pool.submit(_postprocess, page_id, batch_idx, labels, map_html)
                pending_futures[future] = batch_idx

                _collect_futures(pending_futures, results)

            for future in as_completed(pending_futures):
                batch_idx, result = future.result()
                results[batch_idx] = result


@dataclass
class _GPUWorkerState:
    """State for a single GPU worker thread."""

    model: torch.nn.Module
    sep_token_id: int
    device: torch.device
    pad_id: int


def _infer_and_push(
    state: _GPUWorkerState,
    pages: list[_PreparedPage],
    out_queue: queue.Queue,
) -> None:
    """Run batched inference on accumulated pages and push results."""
    max_batch_tokens = 16384

    # Flatten chunks, sort by length
    all_chunks: list[tuple[list[int], list[int], int]] = []
    for page_local_idx, page in enumerate(pages):
        for chunk_ids, block_indices in page.chunks:
            all_chunks.append((chunk_ids, block_indices, page_local_idx))

    all_chunks.sort(key=lambda x: len(x[0]))

    chunk_predictions: dict[int, list[tuple[list[int], list[int]]]] = {}
    i = 0
    while i < len(all_chunks):
        max_seq = len(all_chunks[min(i + 64, len(all_chunks) - 1)][0])
        bs = max(1, max_batch_tokens // max(max_seq, 1))
        batch = all_chunks[i : i + bs]
        i += bs

        max_len = max(len(c[0]) for c in batch)
        input_ids = []
        attention_mask = []
        for chunk_ids, _, _ in batch:
            pad_len = max_len - len(chunk_ids)
            input_ids.append(chunk_ids + [state.pad_id] * pad_len)
            attention_mask.append([1] * len(chunk_ids) + [0] * pad_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=state.device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=state.device)
        outputs = state.model(input_ids=input_ids_t, attention_mask=attention_mask_t)

        for batch_idx, (_, block_indices, page_local_idx) in enumerate(batch):
            logits = outputs.logits[batch_idx]
            sep_positions = (input_ids_t[batch_idx] == state.sep_token_id).nonzero(as_tuple=True)[0]
            preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()
            if page_local_idx not in chunk_predictions:
                chunk_predictions[page_local_idx] = []
            chunk_predictions[page_local_idx].append((block_indices, preds))

    for page_local_idx, page in enumerate(pages):
        predictions = [0] * page.n_blocks
        for block_indices, preds in chunk_predictions.get(page_local_idx, []):
            for idx, block_idx in enumerate(block_indices):
                if idx < len(preds):
                    predictions[block_idx] = preds[idx]
        labels = predictions_to_labels(page.item_ids, predictions)
        out_queue.put((page.page_id, page.batch_idx, labels, page.map_html))


def _collect_futures(pending: dict, results: list) -> None:
    """Collect completed futures without blocking."""
    done = [f for f in pending if f.done()]
    for f in done:
        batch_idx, result = f.result()
        results[batch_idx] = result
        del pending[f]
