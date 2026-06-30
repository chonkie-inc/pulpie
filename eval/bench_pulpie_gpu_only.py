"""Pure GPU-inference throughput for pulpie on local A100.

Pipeline (CPU) is done up-front and cached, then we time only the GPU
inference stage (the same logic as Pipeline._stage2_gpu).
"""

from __future__ import annotations

import os
import time
import argparse

import torch

from pulpie.chunker import extract_blocks, pack_chunks, tokenize_blocks
from pulpie.model_utils import (
    extract_item_ids,
    load_model_and_tokenizer,
    resolve_model_id,
)
from pulpie.simplify import simplify


def load_pages(html_dir: str) -> list[str]:
    pages = []
    for name in sorted(os.listdir(html_dir)):
        if name.endswith(".html"):
            with open(os.path.join(html_dir, name)) as f:
                pages.append(f.read())
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(
        os.path.dirname(__file__),
        "data/block_classifier_eurobert_210m_distill/final"
    ))
    parser.add_argument("--html-dir", default=os.path.join(
        os.path.dirname(__file__), "eval/html"
    ))
    parser.add_argument("--n-pages", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    samples = load_pages(args.html_dir)
    pages = [samples[i % len(samples)] for i in range(args.n_pages)]
    print(f"Loaded {len(samples)} sample HTMLs → {len(pages)} pages")

    device = torch.device(args.device)
    model_id = resolve_model_id(args.model)
    model, tokenizer, sep_token_id = load_model_and_tokenizer(model_id, device)

    # ── CPU pre-processing (not timed) ──
    print("\nCPU pre-processing (simplify + tokenize + chunk)...")
    t_cpu = time.perf_counter()
    per_page_chunks: list[list[tuple[list[int], list[int]]]] = []
    n_blocks_total = 0
    n_failed = 0
    for html in pages:
        try:
            simplified, _ = simplify(html)
        except Exception:
            per_page_chunks.append([])
            n_failed += 1
            continue
        blocks = extract_blocks(simplified)
        n_blocks_total += len(blocks)
        if not blocks:
            per_page_chunks.append([])
            continue
        block_token_ids = tokenize_blocks(blocks, tokenizer)
        chunks = pack_chunks(
            block_token_ids,
            max_tokens=args.max_tokens,
            sep_token_id=sep_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        per_page_chunks.append(chunks)
    t_cpu_done = time.perf_counter() - t_cpu
    print(f"  {n_blocks_total} blocks across {len(pages)} pages in {t_cpu_done:.2f}s ({n_failed} failures)")

    # Flatten, length-sort
    all_chunks: list[tuple[list[int], list[int], int]] = []
    for page_idx, chunks in enumerate(per_page_chunks):
        for chunk_ids, block_indices in chunks:
            all_chunks.append((chunk_ids, block_indices, page_idx))
    all_chunks.sort(key=lambda c: len(c[0]))
    n_chunks = len(all_chunks)
    n_tokens = sum(len(c[0]) for c in all_chunks)
    print(f"  {n_chunks} chunks, {n_tokens:,} tokens "
          f"(avg {n_tokens/max(1,n_chunks):.0f}, "
          f"max {max(len(c[0]) for c in all_chunks):,})")

    pad_id = model.config.pad_token_id or 0

    @torch.no_grad()
    def run_inference() -> float:
        total_padded = 0
        t0 = time.perf_counter()
        i = 0
        while i < n_chunks:
            max_seq = len(all_chunks[min(i + 64, n_chunks - 1)][0])
            bs = max(1, args.max_batch_tokens // max(max_seq, 1))
            batch = all_chunks[i : i + bs]
            i += bs
            max_len = max(len(c[0]) for c in batch)
            input_ids = []
            attention_mask = []
            for chunk_ids, _, _ in batch:
                pad_len = max_len - len(chunk_ids)
                input_ids.append(chunk_ids + [pad_id] * pad_len)
                attention_mask.append([1] * len(chunk_ids) + [0] * pad_len)
            input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
            attn_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
            total_padded += input_ids_t.numel()
            model(input_ids=input_ids_t, attention_mask=attn_t)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, total_padded

    # Warmup
    print("\nWarming up GPU...")
    _ = run_inference()
    _ = run_inference()

    # Timed runs
    print("\nTiming GPU inference (3 runs)...")
    times = []
    for run_idx in range(3):
        elapsed, total_padded = run_inference()
        pps = len(pages) / elapsed
        pad_pct = 100 * (1 - n_tokens / total_padded)
        print(f"  run {run_idx+1}: {elapsed:.2f}s  → {pps:.1f} pps  "
              f"({n_tokens/elapsed:,.0f} real tok/s, "
              f"{total_padded/elapsed:,.0f} padded tok/s, "
              f"pad waste {pad_pct:.1f}%)")
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"\n=== GPU-only throughput (single A100) ===")
    print(f"  {len(pages)} pages, {n_chunks} chunks, {n_tokens:,} tokens")
    print(f"  Avg GPU time:   {avg:.2f}s")
    print(f"  Pages/sec:      {len(pages)/avg:.1f}")
    print(f"  Chunks/sec:     {n_chunks/avg:.1f}")
    print(f"  Tokens/sec:     {n_tokens/avg:,.0f}")
    print(f"\n  CPU prep was:   {t_cpu_done:.2f}s "
          f"({len(pages)/t_cpu_done:.1f} pps sequential, 1 process)")
    print(f"  End-to-end if CPU at ideal parallelism: min({len(pages)/avg:.1f}, cpu_pps)")


if __name__ == "__main__":
    main()
