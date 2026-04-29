"""Benchmark EuroBERT-210M end-to-end throughput on real CC pages.

Measures each pipeline stage:
  1. simplify_html (CPU)
  2. extract_blocks + tokenize_blocks + pack_chunks (CPU)
  3. Model inference (GPU)

Reports pages/sec, tokens/sec, and per-stage breakdown.
"""

import os
import sys
import json
import time
import re
import statistics

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── MinerU-HTML module loading (same as training script) ──
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod

if 'mineru_html' not in sys.modules:
    _make_module('mineru_html')
    _c = _make_module('mineru_html.constants')
    _c.ITEM_ID_ATTR = '_item_id'
    _c.TAIL_BLOCK_TAG = 'cc-alg-uc-text'
    _c.SELECT_ATTR = 'cc-select'
    _c.CLASS_ATTR = 'mark-selected'

    class TagType(Enum):
        Main = 'main'
        Other = 'other'
    _c.TagType = TagType

    _e = _make_module('mineru_html.exceptions')
    class MinerUHTMLError(Exception): pass
    for cn in ['MinerUHTMLPreprocessError', 'MinerUHTMLPromptError',
               'MinerUHTMLResponseParseError', 'MinerUHTMLMapToMainError',
               'MinerUHTMLFallbackError']:
        setattr(_e, cn, type(cn, (MinerUHTMLError,), {}))
    _e.MinerUHTMLError = MinerUHTMLError

    _b = _make_module('mineru_html.base')
    @dataclass
    class MinerUHTMLProcessData:
        simpled_html: str = ''
        map_html: str = ''
    @dataclass
    class MinerUHTMLGenerateInput:
        full_prompt: str = ''
    @dataclass
    class MinerUHTMLParseResult:
        item_label: dict = field(default_factory=dict)
    @dataclass
    class MinerUHTMLOutput:
        main_html: str = ''
    @dataclass
    class MinerUHTMLInput:
        raw_html: str = ''
    @dataclass
    class MinerUHTMLCase:
        case_id: str = ''
        input_data: MinerUHTMLInput = field(default_factory=MinerUHTMLInput)
        process_data: MinerUHTMLProcessData = field(default_factory=MinerUHTMLProcessData)
        generate_input: MinerUHTMLGenerateInput = field(default_factory=MinerUHTMLGenerateInput)
        generate_output: Optional[object] = None
        parse_result: MinerUHTMLParseResult = field(default_factory=MinerUHTMLParseResult)
        output_data: MinerUHTMLOutput = field(default_factory=MinerUHTMLOutput)
    for cls in [MinerUHTMLCase, MinerUHTMLProcessData, MinerUHTMLGenerateInput,
                MinerUHTMLParseResult, MinerUHTMLOutput, MinerUHTMLInput]:
        setattr(_b, cls.__name__, cls)

    _make_module('mineru_html.process')

    def _load_file(mod_name, filename):
        path = os.path.join(MINERU_PATH, 'mineru_html', 'process', filename)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_file('mineru_html.process.html_utils', 'html_utils.py')
    _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
    _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
from block_chunker import extract_blocks, tokenize_blocks, pack_chunks, SEP_TOKEN

MODEL_PATH = os.path.join(SCRIPT_DIR, 'block_classifier_eurobert_210m_distill', 'final')
CC_DATA = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
MAX_TOKENS = 8192


def load_pages(path, n=500):
    pages = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            html = rec.get('html', '')
            if html and len(html) > 500:
                pages.append(html)
            if len(pages) >= n:
                break
    return pages


@torch.no_grad()
def bench_inference_batched(model, all_chunks, device, batch_size=32):
    """Batch inference with length-bucketed batching to minimize padding waste."""
    total_tokens = 0
    total_padded_tokens = 0
    t0 = time.perf_counter()

    flat_chunks = []
    for page_chunks in all_chunks:
        flat_chunks.extend(page_chunks)

    sorted_chunks = sorted(flat_chunks, key=lambda c: len(c[0]))

    pad_id = model.config.pad_token_id or 0
    i = 0
    while i < len(sorted_chunks):
        max_seq = len(sorted_chunks[min(i + batch_size - 1, len(sorted_chunks) - 1)][0])
        effective_bs = max(1, min(batch_size, 128_000 // max(max_seq, 1)))
        batch = sorted_chunks[i:i + effective_bs]
        i += effective_bs

        max_len = max(len(c[0]) for c in batch)

        input_ids = []
        attention_mask = []
        for chunk_ids, _ in batch:
            pad_len = max_len - len(chunk_ids)
            input_ids.append(chunk_ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(chunk_ids) + [0] * pad_len)
            total_tokens += len(chunk_ids)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)
        total_padded_tokens += input_ids_t.numel()

        model(input_ids=input_ids_t, attention_mask=attention_mask_t)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    pad_pct = 100 * (1 - total_tokens / total_padded_tokens) if total_padded_tokens else 0
    return elapsed, total_tokens, len(flat_chunks), total_padded_tokens, pad_pct


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Model: {MODEL_PATH}')
    print(f'Pages: {args.pages}, Batch size: {args.batch_size}, Compile: {args.compile}')

    # Load model
    print('\nLoading model...')
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [SEP_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    if args.compile:
        print('Compiling model...')
        model = torch.compile(model, mode='reduce-overhead')
        with torch.no_grad():
            dummy = torch.randint(100, 30000, (1, 512), device=device)
            mask = torch.ones(1, 512, dtype=torch.long, device=device)
            for _ in range(3):
                model(input_ids=dummy, attention_mask=mask)
        torch.cuda.synchronize()
        print('  Compiled.')

    # Load pages
    print(f'\nLoading {args.pages} CC pages...')
    pages = load_pages(CC_DATA, args.pages)
    print(f'  Loaded {len(pages)} pages (avg {np.mean([len(p) for p in pages])/1024:.0f}KB HTML)')

    # Stage 1: simplify_html
    print('\n--- Stage 1: simplify_html ---')
    simplified = []
    errors_simplify = 0
    t0 = time.perf_counter()
    for html in pages:
        try:
            s, _ = simplify_html(html)
            simplified.append(s)
        except Exception:
            errors_simplify += 1
            simplified.append(None)
    t_simplify = time.perf_counter() - t0
    valid = [s for s in simplified if s is not None]
    print(f'  {len(valid)}/{len(pages)} pages simplified in {t_simplify:.1f}s '
          f'({len(valid)/t_simplify:.0f} pages/sec, {t_simplify/len(pages)*1000:.1f}ms/page)')
    if errors_simplify:
        print(f'  {errors_simplify} errors')

    # Stage 2: block extraction + tokenization + chunking
    print('\n--- Stage 2: extract + tokenize + chunk ---')
    all_chunks = []
    total_blocks = 0
    total_tokens_prepared = 0
    chunk_lengths = []
    blocks_per_page = []
    chunks_per_page = []
    t0 = time.perf_counter()
    for s in simplified:
        if s is None:
            all_chunks.append([])
            continue
        blocks = extract_blocks(s)
        total_blocks += len(blocks)
        blocks_per_page.append(len(blocks))
        block_token_ids = tokenize_blocks(blocks, tokenizer)
        chunks = pack_chunks(
            block_token_ids, max_tokens=MAX_TOKENS,
            sep_token_id=sep_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        all_chunks.append(chunks)
        chunks_per_page.append(len(chunks))
        for chunk_ids, _ in chunks:
            chunk_lengths.append(len(chunk_ids))
            total_tokens_prepared += len(chunk_ids)
    t_prepare = time.perf_counter() - t0

    total_chunks = sum(len(c) for c in all_chunks)
    print(f'  {total_blocks} blocks → {total_chunks} chunks in {t_prepare:.1f}s '
          f'({len(pages)/t_prepare:.0f} pages/sec)')
    print(f'  Blocks/page: median={np.median(blocks_per_page):.0f}, '
          f'mean={np.mean(blocks_per_page):.0f}, p95={np.percentile(blocks_per_page, 95):.0f}')
    print(f'  Chunks/page: median={np.median(chunks_per_page):.0f}, '
          f'mean={np.mean(chunks_per_page):.1f}, max={max(chunks_per_page)}')
    print(f'  Tokens/chunk: median={np.median(chunk_lengths):.0f}, '
          f'mean={np.mean(chunk_lengths):.0f}, p95={np.percentile(chunk_lengths, 95):.0f}')
    print(f'  Total tokens: {total_tokens_prepared:,}')

    # Stage 3: GPU inference (warmup)
    print('\n--- Stage 3: GPU inference ---')
    print('  Warmup...')
    _ = bench_inference_batched(model, all_chunks[:10], device, args.batch_size)

    # Actual benchmark
    print('  Benchmarking...')
    t_infer, tokens_inferred, n_chunks_inferred, padded_tokens, pad_pct = bench_inference_batched(
        model, all_chunks, device, args.batch_size,
    )
    print(f'  {n_chunks_inferred} chunks, {tokens_inferred:,} tokens in {t_infer:.1f}s')
    print(f'  GPU: {tokens_inferred/t_infer:,.0f} tokens/sec, {n_chunks_inferred/t_infer:.0f} chunks/sec')
    print(f'  Padding waste: {pad_pct:.1f}% ({padded_tokens:,} padded vs {tokens_inferred:,} real)')

    # End-to-end
    print(f'\n{"="*60}')
    print(f'END-TO-END THROUGHPUT ({len(pages)} real CC pages)')
    print(f'{"="*60}')
    t_total = t_simplify + t_prepare + t_infer
    pps = len(pages) / t_total
    print(f'  simplify_html:   {t_simplify:>6.1f}s  ({100*t_simplify/t_total:>4.1f}%)')
    print(f'  extract+chunk:   {t_prepare:>6.1f}s  ({100*t_prepare/t_total:>4.1f}%)')
    print(f'  GPU inference:   {t_infer:>6.1f}s  ({100*t_infer/t_total:>4.1f}%)')
    print(f'  ─────────────────────────')
    print(f'  Total:           {t_total:>6.1f}s')
    print(f'  Pages/sec:       {pps:>6.1f}')
    print(f'  ms/page:         {1000/pps:>6.1f}')

    # Extrapolation
    print(f'\n  Extrapolation to 1B pages:')
    hours_1b = 1e9 / pps / 3600
    print(f'    Single GPU:    {hours_1b:,.0f} hours ({hours_1b/24:,.0f} days)')
    for n_gpu in [4, 8, 16, 32]:
        days = hours_1b / 24 / n_gpu
        print(f'    {n_gpu} GPUs:        {days:,.1f} days')


if __name__ == '__main__':
    main()
