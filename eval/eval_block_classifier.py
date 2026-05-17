"""Evaluate [BLOCK] marker classifier on WebMainBench.

Pipeline per page:
  1. simplify_html → simplified HTML + map_html
  2. Insert [BLOCK] markers before each _item_id
  3. Tokenize → model forward → classify at [BLOCK] positions
  4. extract_main_html(map_html, labels) → main content HTML
  5. html2text → prediction text
  6. ROUGE-5 F1 vs convert_main_content reference

Reports: ROUGE-5 by difficulty, throughput (pages/sec), comparison table.
"""

import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import html2text
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
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
extract_main_html = sys.modules['mineru_html.process.map_to_main'].extract_main_html

# ── Config ──
MODEL_PATH = os.path.join(DATA_DIR, 'block_classifier_0.6B', 'final')
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
BLOCK_TOKEN = "[BLOCK]"
MAX_LENGTH = 32768
LABEL_MAIN = 1


def html_to_text(html_str):
    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    pred_ngrams = Counter(ngrams(pred_tokens, n))
    if not ref_ngrams or not pred_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & pred_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def insert_block_markers(simplified_html):
    """Insert [BLOCK] tokens and extract item_ids in order."""
    pattern = re.compile(r'(_item_id="(\d+)")')
    item_ids = []
    parts = []
    last_end = 0

    for m in pattern.finditer(simplified_html):
        item_id = m.group(2)
        parts.append(simplified_html[last_end:m.start()])
        parts.append(BLOCK_TOKEN + ' ')
        parts.append(m.group(0))
        last_end = m.end()
        item_ids.append(item_id)

    if not item_ids:
        return None, []

    parts.append(simplified_html[last_end:])
    marked_html = ''.join(parts)
    return marked_html, item_ids


@torch.no_grad()
def classify_page(model, tokenizer, block_token_id, simplified_html, device):
    """Classify blocks on a single page. Returns dict {item_id: 'main'/'other'}."""
    marked_html, item_ids = insert_block_markers(simplified_html)
    if marked_html is None:
        return {}

    encoding = tokenizer(
        marked_html,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding=False,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attn_mask = {'full_attention': None}

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits[0]  # [seq_len, 2]

    # Extract predictions at [BLOCK] positions
    block_positions = (input_ids[0] == block_token_id).nonzero(as_tuple=True)[0]
    preds = logits[block_positions].argmax(dim=-1).cpu().tolist()

    # Map to labels
    labels = {}
    for i, item_id in enumerate(item_ids):
        if i < len(preds):
            labels[item_id] = 'main' if preds[i] == LABEL_MAIN else 'other'
        else:
            # Truncated — default to other
            labels[item_id] = 'other'

    return labels


@torch.no_grad()
def classify_batch(model, tokenizer, block_token_id, pages_data, device):
    """Classify multiple pages in a batch. Returns list of label dicts."""
    # Prepare all pages
    all_marked = []
    all_item_ids = []
    valid_indices = []

    for idx, (simplified, _) in enumerate(pages_data):
        marked_html, item_ids = insert_block_markers(simplified)
        if marked_html is None:
            all_marked.append(None)
            all_item_ids.append([])
        else:
            all_marked.append(marked_html)
            all_item_ids.append(item_ids)
            valid_indices.append(idx)

    if not valid_indices:
        return [{} for _ in pages_data]

    # Tokenize valid pages
    texts = [all_marked[i] for i in valid_indices]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
    )

    input_ids = encodings['input_ids'].to(device)
    attn_mask = {'full_attention': None}

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits  # [batch, seq_len, 2]

    # Extract predictions
    results = [{} for _ in pages_data]
    for batch_idx, page_idx in enumerate(valid_indices):
        ids = input_ids[batch_idx]
        block_positions = (ids == block_token_id).nonzero(as_tuple=True)[0]
        preds = logits[batch_idx][block_positions].argmax(dim=-1).cpu().tolist()

        item_ids = all_item_ids[page_idx]
        labels = {}
        for i, item_id in enumerate(item_ids):
            if i < len(preds):
                labels[item_id] = 'main' if preds[i] == LABEL_MAIN else 'other'
            else:
                labels[item_id] = 'other'
        results[page_idx] = labels

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f'Loading model: {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    ).to(device).eval()

    # Make bidirectional
    for m in model.modules():
        if hasattr(m, 'is_causal'):
            m.is_causal = False

    block_token_id = tokenizer.convert_tokens_to_ids(BLOCK_TOKEN)
    print(f'  [BLOCK] token id = {block_token_id}', flush=True)

    # Load WebMainBench
    print(f'\nLoading WebMainBench...', flush=True)
    records = []
    with open(WMB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                records.append(rec)

    if args.limit > 0:
        records = records[:args.limit]
    print(f'  {len(records)} English pages', flush=True)

    # Eval loop
    scores = []
    scores_by_level = {}
    empty_count = 0
    simplify_fail = 0
    n_blocks_total = 0
    n_main_total = 0
    timings = []

    for i, rec in enumerate(records):
        html_content = rec.get('html', '')
        reference = rec.get('convert_main_content', '')
        level = rec.get('meta', {}).get('level', 'unknown')

        if not html_content or not reference:
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        t0 = time.time()

        # Step 1: Simplify HTML
        try:
            simplified, map_html = simplify_html(html_content)
        except Exception:
            simplify_fail += 1
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        # Step 2-3: Classify blocks
        labels = classify_page(model, tokenizer, block_token_id, simplified, device)

        n_blocks = len(labels)
        n_main = sum(1 for v in labels.values() if v == 'main')
        n_blocks_total += n_blocks
        n_main_total += n_main

        # Step 4: Reconstruct main HTML
        if n_main == 0:
            pred_text = ''
        else:
            try:
                main_html = extract_main_html(map_html, labels)
                pred_text = html_to_text(main_html).strip()
            except Exception:
                pred_text = ''

        elapsed = time.time() - t0
        timings.append(elapsed)

        # Step 5: Score
        if not pred_text:
            empty_count += 1
            r5 = 0.0
        else:
            r5 = rouge_n_f1(reference, pred_text, n=5)

        scores.append(r5)
        scores_by_level.setdefault(level, []).append(r5)

        if (i + 1) % 500 == 0:
            n = len(scores)
            avg = sum(scores) / n
            pg_s = len(timings) / sum(timings) if timings else 0
            print(f'  {i+1}/{len(records)}: ROUGE-5={avg:.4f}  empty={empty_count}  simplify_fail={simplify_fail}  {pg_s:.1f} pg/s', flush=True)

    # Final results
    n = len(scores)
    avg_all = sum(scores) / max(n, 1)
    total_time = sum(timings)
    throughput = len(timings) / total_time if total_time > 0 else 0

    print(f'\n{"="*70}', flush=True)
    print(f'[BLOCK] CLASSIFIER — WebMainBench ROUGE-5 F1 (English, {n} pages)', flush=True)
    print(f'{"="*70}', flush=True)

    print(f'\n  Model: {args.model_path}', flush=True)
    print(f'  Blocks classified: {n_blocks_total:,} ({n_main_total:,} main, {n_blocks_total-n_main_total:,} other)', flush=True)
    print(f'  Main block rate: {n_main_total/max(n_blocks_total,1)*100:.1f}%', flush=True)
    print(f'  Simplify failures: {simplify_fail}', flush=True)
    print(f'  Empty extractions: {empty_count}', flush=True)

    print(f'\n  Throughput: {throughput:.1f} pages/sec (single GPU, BS=1)', flush=True)
    print(f'  Avg latency: {total_time/max(len(timings),1)*1000:.0f} ms/page', flush=True)
    print(f'  Total time: {total_time:.0f}s', flush=True)

    print(f'\n  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}', flush=True)
    print(f'  {"-"*67}', flush=True)

    level_avgs = {}
    for lev in ['simple', 'mid', 'hard']:
        vals = scores_by_level.get(lev, [])
        level_avgs[lev] = sum(vals) / max(len(vals), 1)

    comparisons = [
        ('DeepSeek-V3.2 (LLM)', 0.9098, 0.9415, 0.9104, 0.8771),
        ('GPT-4 (LLM)', 0.9024, 0.9382, 0.9042, 0.8638),
        ('Dripper 0.6B', 0.8779, 0.9205, 0.8804, 0.8313),
        ('** [BLOCK] classifier **', avg_all, level_avgs['simple'], level_avgs['mid'], level_avgs['hard']),
        ('Pulpie GBM (h2t)', 0.808, 0.885, 0.805, 0.740),
        ('magic-html', 0.7138, 0.7857, 0.7121, 0.6434),
        ('Readability', 0.6543, 0.7415, 0.6550, 0.5652),
        ('Trafilatura', 0.6402, 0.7309, 0.6417, 0.5466),
    ]
    comparisons.sort(key=lambda x: -x[1])

    for name, r_all, r_s, r_m, r_h in comparisons:
        marker = ' <--' if 'BLOCK' in name else ''
        print(f'  {name:<35} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}{marker}', flush=True)

    # F1 distribution
    print(f'\n  F1 Distribution:', flush=True)
    bins = [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
    for lo, hi in bins:
        count = sum(1 for s in scores if lo <= s < hi)
        pct = count / max(n, 1) * 100
        bar = '#' * int(pct / 2)
        label = f'{lo:.1f}-{hi:.1f}' if hi <= 1.0 else f'{lo:.1f}-1.0'
        print(f'    {label}: {count:>5} ({pct:>5.1f}%)  {bar}', flush=True)
    zero_count = sum(1 for s in scores if s == 0.0)
    print(f'    exact 0: {zero_count:>5} ({zero_count/max(n,1)*100:>5.1f}%)', flush=True)

    print(f'\n  NOTE: Dripper/GBM numbers from work.log. Dripper paper includes non-English.', flush=True)


if __name__ == '__main__':
    main()
