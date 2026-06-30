"""Throughput benchmark for pulpie on the local A100.

Loads the 13 sample HTML pages in eval/html, repeats them to N pages, then
runs two modes:
  1. Extractor — single-page synchronous loop (bound by sequential CPU→GPU→CPU)
  2. Pipeline — fully overlapped multi-worker CPU + GPU

Reports end-to-end pages/sec.
"""

from __future__ import annotations

import os
import sys
import time
import argparse


def load_pages(html_dir: str) -> list[str]:
    pages = []
    for name in sorted(os.listdir(html_dir)):
        if not name.endswith(".html"):
            continue
        with open(os.path.join(html_dir, name)) as f:
            pages.append(f.read())
    return pages


def bench_extractor(model: str, pages: list[str], warmup: int = 3) -> None:
    from pulpie import Extractor

    print(f"\n=== Extractor (single-page loop) ===")
    ext = Extractor(model=model)

    for i in range(warmup):
        ext.extract(pages[i % len(pages)])

    t0 = time.perf_counter()
    for html in pages:
        ext.extract(html)
    elapsed = time.perf_counter() - t0

    pps = len(pages) / elapsed
    print(f"  {len(pages)} pages in {elapsed:.2f}s → {pps:.1f} pps ({1000/pps:.1f} ms/page)")


def bench_pipeline(model: str, pages: list[str], n_workers: int) -> None:
    from pulpie import Pipeline, PageInput

    print(f"\n=== Pipeline (overlapped, n_workers={n_workers}) ===")
    pipeline = Pipeline(model=model, n_workers=n_workers)

    # Warmup: run on a small slice
    _ = pipeline.extract_batch([PageInput(html=pages[i], page_id=i) for i in range(min(8, len(pages)))])

    inputs = [PageInput(html=html, page_id=i) for i, html in enumerate(pages)]
    t0 = time.perf_counter()
    results = pipeline.extract_batch(inputs)
    elapsed = time.perf_counter() - t0

    pps = len(pages) / elapsed
    n_ok = sum(1 for r in results if r and not r.error)
    print(f"  {len(pages)} pages in {elapsed:.2f}s → {pps:.1f} pps ({1000/pps:.1f} ms/page)")
    print(f"  {n_ok}/{len(pages)} successful")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(
        os.path.dirname(__file__),
        "data/block_classifier_eurobert_210m_distill/final"
    ))
    parser.add_argument("--html-dir", default=os.path.join(
        os.path.dirname(__file__), "eval/html"
    ))
    parser.add_argument("--n-pages", type=int, default=100,
                        help="Total pages to process (will cycle the samples).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=["extractor", "pipeline", "both"],
                        default="both")
    args = parser.parse_args()

    samples = load_pages(args.html_dir)
    print(f"Loaded {len(samples)} sample HTMLs from {args.html_dir}")
    print(f"Sizes: {[len(s)//1024 for s in samples]} KB")

    # Cycle to reach requested count
    pages = [samples[i % len(samples)] for i in range(args.n_pages)]
    print(f"Benchmarking {len(pages)} pages (cycled from {len(samples)} unique).")

    if args.mode in ("extractor", "both"):
        bench_extractor(args.model, pages)

    if args.mode in ("pipeline", "both"):
        bench_pipeline(args.model, pages, args.workers)


if __name__ == "__main__":
    main()
