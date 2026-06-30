"""Like bench_pulpie_gpu_only.py but with torch.compile on the model.

Compares tokens/sec with and without compile to see what's on the table.
"""

from __future__ import annotations

import os
import time
import argparse

import torch

from pulpie.chunker import extract_blocks, pack_chunks, tokenize_blocks
from pulpie.model_utils import load_model_and_tokenizer, resolve_model_id
from pulpie.simplify import simplify


def load_pages(html_dir: str) -> list[str]:
    return [
        open(os.path.join(html_dir, n)).read()
        for n in sorted(os.listdir(html_dir)) if n.endswith(".html")
    ]


def prepare(pages, tokenizer, sep_token_id, max_tokens):
    all_chunks = []
    for idx, html in enumerate(pages):
        try:
            simplified, _ = simplify(html)
        except Exception:
            continue
        blocks = extract_blocks(simplified)
        if not blocks:
            continue
        toks = tokenize_blocks(blocks, tokenizer)
        chunks = pack_chunks(
            toks, max_tokens=max_tokens,
            sep_token_id=sep_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for c, b in chunks:
            all_chunks.append((c, b, idx))
    all_chunks.sort(key=lambda x: len(x[0]))
    return all_chunks


@torch.no_grad()
def run_inference(model, all_chunks, device, max_batch_tokens, pad_id):
    t0 = time.perf_counter()
    i = 0
    n_chunks = len(all_chunks)
    while i < n_chunks:
        max_seq = len(all_chunks[min(i + 64, n_chunks - 1)][0])
        bs = max(1, max_batch_tokens // max(max_seq, 1))
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
        model(input_ids=input_ids_t, attention_mask=attn_t)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    samples = load_pages(args.html_dir)
    pages = [samples[i % len(samples)] for i in range(args.n_pages)]
    print(f"{len(pages)} pages, compile={args.compile}")

    device = torch.device(args.device)
    model_id = resolve_model_id(args.model)
    model, tokenizer, sep_token_id = load_model_and_tokenizer(model_id, device)

    if args.compile:
        print("torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")

    print("CPU prep...")
    t_cpu = time.perf_counter()
    all_chunks = prepare(pages, tokenizer, sep_token_id, args.max_tokens)
    print(f"  {len(all_chunks)} chunks in {time.perf_counter()-t_cpu:.1f}s")
    n_tokens = sum(len(c[0]) for c in all_chunks)
    print(f"  {n_tokens:,} tokens (avg {n_tokens/len(all_chunks):.0f}, "
          f"max {max(len(c[0]) for c in all_chunks):,})")

    pad_id = model.config.pad_token_id if hasattr(model, 'config') else 0
    if pad_id is None:
        pad_id = 0

    print("Warmup...")
    _ = run_inference(model, all_chunks, device, args.max_batch_tokens, pad_id)
    _ = run_inference(model, all_chunks, device, args.max_batch_tokens, pad_id)

    print("Timed runs...")
    times = []
    for _ in range(3):
        t = run_inference(model, all_chunks, device, args.max_batch_tokens, pad_id)
        times.append(t)
        print(f"  {t:.2f}s  → {len(pages)/t:.1f} pps, {n_tokens/t:,.0f} tok/s")

    avg = sum(times) / len(times)
    print(f"\nGPU-only: {len(pages)/avg:.1f} pps, {n_tokens/avg:,.0f} tokens/sec "
          f"(compile={args.compile})")


if __name__ == "__main__":
    main()
