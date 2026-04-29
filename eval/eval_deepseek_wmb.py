"""Verify DeepSeek V3.2 label quality on WebMainBench.

Runs the exact same pipeline as label_cc.py (simplify → v0 prompt → DeepSeek → parse → reconstruct)
on WMB English pages and measures ROUGE-5 F1 against reference text.

Usage:
  python eval/eval_deepseek_wmb.py --limit 200
  python eval/eval_deepseek_wmb.py  # full WMB English
"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import boto3
import html2text

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod

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
_simplify = _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
_prompt = _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
_parse = _load_file('mineru_html.process.parse_result', 'parse_result.py')
_map = _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = _simplify.simplify_html
get_full_prompt = _prompt.get_full_prompt
parse_llm_response = _parse.parse_llm_response
extract_main_html = _map.extract_main_html

# ── Config ──
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
BEDROCK_REGION = 'us-west-2'
MODEL_ID = 'deepseek.v3.2'
MAX_TOKENS = 4096
PROMPT_VERSION = 'v0'
CONCURRENCY = 10


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


def process_page(client, page_idx, html_content, prompt_version):
    """Run full DeepSeek classification pipeline on one page."""
    try:
        simplified, map_html = simplify_html(html_content)
    except Exception as e:
        return page_idx, None, f'simplify_fail: {e}'

    try:
        prompt = get_full_prompt(simplified, version=prompt_version)
    except Exception as e:
        return page_idx, None, f'prompt_fail: {e}'

    body = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': MAX_TOKENS,
        'temperature': 0,
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID, body=body,
            contentType='application/json', accept='application/json',
        )
        result = json.loads(response['body'].read())
        text = result['choices'][0]['message']['content']
    except Exception as e:
        return page_idx, None, f'llm_fail: {e}'

    try:
        labels = parse_llm_response(text)
    except Exception as e:
        return page_idx, None, f'parse_fail: {e}'

    n_main = sum(1 for v in labels.values() if v == 'main')
    if n_main == 0:
        return page_idx, '', 'ok_empty'

    try:
        main_html = extract_main_html(map_html, labels)
        pred_text = html_to_text(main_html).strip()
        return page_idx, pred_text, 'ok'
    except Exception as e:
        return page_idx, None, f'map_fail: {e}'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY)
    parser.add_argument('--prompt-version', type=str, default=PROMPT_VERSION,
                        choices=['v0', 'v1', 'v2', 'compact', 'short_compact'])
    args = parser.parse_args()

    print(f'Loading WebMainBench (English only)...', flush=True)
    pages = []
    with open(WMB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                pages.append(rec)

    if args.limit > 0:
        pages = pages[:args.limit]
    print(f'  {len(pages)} pages', flush=True)
    print(f'  Prompt version: {args.prompt_version}', flush=True)
    print(f'  Model: {MODEL_ID}', flush=True)
    print(f'  Concurrency: {args.concurrency}', flush=True)

    client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)

    # Run all pages concurrently
    results = [None] * len(pages)
    stats = {'ok': 0, 'ok_empty': 0, 'simplify_fail': 0, 'prompt_fail': 0,
             'llm_fail': 0, 'parse_fail': 0, 'map_fail': 0}
    t0 = time.time()

    print(f'\nClassifying with DeepSeek V3.2...', flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_page, client, i, page['html'], args.prompt_version): i
            for i, page in enumerate(pages)
        }
        done_count = 0
        for future in as_completed(futures):
            page_idx, pred_text, status = future.result()
            results[page_idx] = (pred_text, status)

            status_key = status.split(':')[0] if ':' in status else status
            if status_key in stats:
                stats[status_key] += 1
            else:
                stats[status_key] = stats.get(status_key, 0) + 1

            done_count += 1
            if done_count % 50 == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed
                eta = (len(pages) - done_count) / rate
                print(f'  {done_count}/{len(pages)} ({elapsed:.0f}s, {rate:.1f}pg/s, ETA={eta:.0f}s) '
                      f'ok={stats["ok"]} empty={stats["ok_empty"]} '
                      f'llm_fail={stats.get("llm_fail",0)} parse_fail={stats.get("parse_fail",0)}',
                      flush=True)

    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.0f}s ({len(pages)/elapsed:.1f} pg/s)', flush=True)
    for k, v in sorted(stats.items()):
        print(f'  {k}: {v}', flush=True)

    # Score with ROUGE-5
    print(f'\nScoring ROUGE-5...', flush=True)
    scores = []
    scores_by_level = {}
    empty_count = 0

    for i, page in enumerate(pages):
        reference = page.get('convert_main_content', '')
        level = page.get('meta', {}).get('level', 'unknown')

        pred_text, status = results[i] if results[i] else (None, 'missing')

        if not reference or pred_text is None or pred_text == '':
            r5 = 0.0
            if pred_text == '' or pred_text is None:
                empty_count += 1
        else:
            r5 = rouge_n_f1(reference, pred_text, n=5)

        scores.append(r5)
        scores_by_level.setdefault(level, []).append(r5)

    # Report
    n = len(scores)
    avg_all = sum(scores) / max(n, 1)

    print(f'\n{"="*70}')
    print(f'DEEPSEEK V3.2 — WebMainBench ROUGE-5 F1 (English, {n} pages)')
    print(f'{"="*70}')
    print(f'  Prompt version: {args.prompt_version}')
    print(f'  Empty extractions: {empty_count}')

    print(f'\n  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}')
    print(f'  {"-"*67}')

    level_avgs = {}
    for lev in ['simple', 'mid', 'hard']:
        vals = scores_by_level.get(lev, [])
        level_avgs[lev] = sum(vals) / max(len(vals), 1)

    comparisons = [
        (f'** DeepSeek V3.2 ({args.prompt_version}) **', avg_all,
         level_avgs.get('simple', 0), level_avgs.get('mid', 0), level_avgs.get('hard', 0)),
        ('DeepSeek V3.2 (paper)', 0.9098, 0.9415, 0.9104, 0.8771),
        ('Dripper 0.6B (paper)', 0.8779, 0.9205, 0.8804, 0.8313),
        ('Hummingbird Latte Large', 0.8642, 0.8909, 0.8724, 0.8293),
        ('Hummingbird Latte Base', 0.857, 0.889, 0.866, 0.816),
        ('Hummingbird Espresso (GBM)', 0.808, 0.885, 0.805, 0.740),
    ]
    comparisons.sort(key=lambda x: -x[1])

    for name, r_all, r_s, r_m, r_h in comparisons:
        marker = ' <--' if 'DeepSeek' in name and 'paper' not in name else ''
        print(f'  {name:<35} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}{marker}')

    # F1 distribution
    print(f'\n  F1 Distribution:')
    bins = [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
    for lo, hi in bins:
        count = sum(1 for s in scores if lo <= s < hi)
        pct = count / max(n, 1) * 100
        bar = '#' * int(pct / 2)
        label = f'{lo:.1f}-{hi:.1f}' if hi <= 1.0 else f'{lo:.1f}-1.0'
        print(f'    {label}: {count:>5} ({pct:>5.1f}%)  {bar}')
    zero_count = sum(1 for s in scores if s == 0.0)
    print(f'    exact 0: {zero_count:>5} ({zero_count/max(n,1)*100:>5.1f}%)')


if __name__ == '__main__':
    main()
