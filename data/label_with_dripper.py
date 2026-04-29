"""Label CC pages using Dripper 0.5B via local vLLM.

Runs MinerU-HTML simplify → Dripper prompt → parse labels for each page.
No DeepSeek needed — Dripper is the sole labeler.

Usage:
  # Start vLLM first:
  vllm serve opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact \
      --port 8235 --max-model-len 65536 --gpu-memory-utilization 0.9

  # Then run:
  python data/label_with_dripper.py [--limit N] [--concurrency 32]
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
    _load_file('mineru_html.process.parse_result', 'parse_result.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
get_full_prompt = sys.modules['mineru_html.process.build_prompt'].get_full_prompt
parse_llm_response = sys.modules['mineru_html.process.parse_result'].parse_llm_response

# ── Config ──
VLLM_URL = "http://localhost:8235/v1/chat/completions"
MODEL_NAME = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
PROMPT_VERSION = 'short_compact'
MAX_TOKENS = 4096
CONCURRENCY = 32

INPUT_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled_100k.jsonl')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_dripper_100k.jsonl')


def extract_item_ids(html_str):
    return [int(m) for m in re.findall(r'_item_id="(\d+)"', html_str)]


def build_guided_regex(item_ids):
    item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
    return f'<answer>\\s*{item_pattern}\\s*</answer>'


def call_dripper(prompt, item_ids=None):
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    if item_ids:
        body["guided_regex"] = build_guided_regex(item_ids)
    resp = requests.post(VLLM_URL, json=body, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    text = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    return text, usage


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=CONCURRENCY)
    parser.add_argument('--resume', action='store_true', help='Resume from existing output')
    args = parser.parse_args()

    # Load pages
    print(f'Loading pages from {args.input}...', flush=True)
    pages = []
    with open(args.input) as f:
        for line in f:
            pages.append(json.loads(line))
    print(f'  {len(pages)} pages loaded', flush=True)

    if args.limit > 0:
        pages = pages[:args.limit]
        print(f'  Limited to {len(pages)}', flush=True)

    # Resume: skip already-labeled URLs
    done_urls = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                r = json.loads(line)
                done_urls.add(r.get('url', ''))
        pages = [p for p in pages if p['url'] not in done_urls]
        print(f'  Resuming: {len(done_urls)} already done, {len(pages)} remaining', flush=True)

    # Simplify all pages first (CPU-bound, not parallelized)
    print('\nSimplifying HTML...', flush=True)
    prepared = []
    simp_fail = 0
    for i, page in enumerate(pages):
        try:
            simplified, _ = simplify_html(page['html'])
            prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
            item_ids = extract_item_ids(simplified)
            prepared.append({
                'url': page['url'],
                'domain': page.get('domain', ''),
                'prompt': prompt,
                'item_ids': item_ids,
            })
        except Exception:
            simp_fail += 1
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{len(pages)} simplified, {simp_fail} failed', flush=True)
    print(f'  Done: {len(prepared)} prepared, {simp_fail} failed', flush=True)

    # Run Dripper
    print(f'\nLabeling with Dripper (concurrency={args.concurrency})...', flush=True)
    write_lock = Lock()
    stats = {'ok': 0, 'llm_fail': 0, 'parse_fail': 0}
    t_start = time.time()

    mode = 'a' if args.resume else 'w'
    out_file = open(args.output, mode)

    def do_one(p):
        url = p['url']
        domain = p['domain']
        try:
            response_text, usage = call_dripper(p['prompt'], item_ids=p.get('item_ids'))
        except Exception as e:
            return {'url': url, 'domain': domain, 'status': 'llm_fail', 'error': str(e)[:200]}

        try:
            labels = parse_llm_response(response_text)
        except Exception:
            return {'url': url, 'domain': domain, 'status': 'parse_fail'}

        n_main = sum(1 for v in labels.values() if v == 'main')
        n_total = len(labels)

        return {
            'url': url,
            'domain': domain,
            'status': 'ok',
            'labels': labels,
            'n_main': n_main,
            'n_total': n_total,
            'input_tokens': usage.get('prompt_tokens', 0),
            'output_tokens': usage.get('completion_tokens', 0),
        }

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(do_one, p): p for p in prepared}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                with write_lock:
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    if (stats['ok'] + stats.get('llm_fail', 0) + stats.get('parse_fail', 0)) % 100 == 0:
                        out_file.flush()
                    stats[result['status']] = stats.get(result['status'], 0) + 1

                done = sum(stats.values())
                if done % 500 == 0 or done == len(prepared):
                    elapsed = time.time() - t_start
                    rate = done / max(elapsed, 1)
                    eta = (len(prepared) - done) / max(rate, 0.001)
                    print(f'  {done:>6}/{len(prepared)} ok={stats["ok"]} fail={stats.get("llm_fail",0)+stats.get("parse_fail",0)} '
                          f'{rate:.1f}pg/s ETA={eta/60:.0f}m', flush=True)
    except KeyboardInterrupt:
        print('\n  Interrupted! Progress saved.', flush=True)
    finally:
        out_file.close()

    elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'Done in {elapsed:.0f}s ({elapsed/60:.1f}m)')
    print(f'  OK: {stats["ok"]}')
    print(f'  LLM fail: {stats.get("llm_fail", 0)}')
    print(f'  Parse fail: {stats.get("parse_fail", 0)}')
    print(f'  Throughput: {stats["ok"]/max(elapsed,1):.1f} pg/s')
    print(f'  Output: {args.output}')


if __name__ == '__main__':
    main()
