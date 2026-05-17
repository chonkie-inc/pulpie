"""Evaluate Dripper using native vLLM (same as MinerU-HTML's official pipeline).

Key differences from our previous eval:
- Uses inline vLLM (not API server)
- Uses StructuredOutputsParams for guided decoding
- Uses short_compact prompt
- Uses apply_chat_template with enable_thinking=False
- Uses trafilatura fallback on failure
- max_model_len=32768 (not 8192)

Usage:
  python eval/eval_dripper_native.py --limit 200
"""

import json
import os
import re
import sys
import time
from collections import Counter

import html2text
import torch

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
    _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
    _load_file('mineru_html.process.parse_result', 'parse_result.py')
    _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
get_full_prompt = sys.modules['mineru_html.process.build_prompt'].get_full_prompt
parse_llm_response = sys.modules['mineru_html.process.parse_result'].parse_llm_response
extract_main_html = sys.modules['mineru_html.process.map_to_main'].extract_main_html

# ── Config ──
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
DRIPPER_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
PROMPT_VERSION = 'short_compact'
MAX_MODEL_LEN = 32768
MAX_TOKENS = 16384


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-model-len', type=int, default=MAX_MODEL_LEN)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer

    # Load pages
    print(f'Loading WebMainBench (English, limit={args.limit})...', flush=True)
    pages = []
    with open(WMB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                pages.append(rec)
    if args.limit > 0:
        pages = pages[:args.limit]
    print(f'  {len(pages)} pages', flush=True)

    # Simplify all pages first
    print(f'\nSimplifying HTML...', flush=True)
    simplified_pages = []
    simp_fail = 0
    for page in pages:
        try:
            simplified, map_html = simplify_html(page['html'])
            simplified_pages.append((simplified, map_html, page))
        except Exception:
            simplified_pages.append((None, None, page))
            simp_fail += 1
    print(f'  {len(pages) - simp_fail} ok, {simp_fail} failed', flush=True)

    # Build prompts
    print(f'Building prompts (version={PROMPT_VERSION})...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(DRIPPER_MODEL)

    prompts = []
    item_ids_list = []
    valid_indices = []
    too_long = 0

    for i, (simplified, map_html, page) in enumerate(simplified_pages):
        if simplified is None:
            continue

        try:
            prompt = get_full_prompt(simplified, version=PROMPT_VERSION)
        except Exception:
            continue

        # Apply chat template (same as official pipeline)
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True,
        )

        # Extract item_ids for guided decoding
        item_ids = [int(m) for m in re.findall(r'_item_id="(\d+)"', simplified)]
        if not item_ids:
            continue

        # Check length
        token_ids = tokenizer.encode(chat_prompt)
        if len(token_ids) > args.max_model_len - 1000:
            too_long += 1
            continue

        prompts.append(chat_prompt)
        item_ids_list.append(item_ids)
        valid_indices.append(i)

    print(f'  {len(prompts)} valid prompts, {too_long} too long', flush=True)

    # Load vLLM
    print(f'\nLoading vLLM ({DRIPPER_MODEL}, max_len={args.max_model_len})...', flush=True)
    llm = LLM(
        model=DRIPPER_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )

    # Build sampling params with structured output per page
    print(f'Running inference with guided decoding...', flush=True)
    sampling_params_list = []
    for item_ids in item_ids_list:
        item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
        pattern = f'<answer>\\s*{item_pattern}\\s*</answer>'
        structured_outputs_params = StructuredOutputsParams(regex=pattern)
        sampling_params_list.append(SamplingParams(
            structured_outputs=structured_outputs_params,
            top_k=1, top_p=0.95, temperature=0, max_tokens=MAX_TOKENS,
        ))

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params_list)
    gen_time = time.time() - t0
    print(f'  Generated {len(outputs)} responses in {gen_time:.0f}s ({len(outputs)/gen_time:.1f} pg/s)', flush=True)

    # Parse and score
    print(f'\nParsing and scoring...', flush=True)
    scores = [0.0] * len(pages)
    scores_by_level = {}
    parse_fail = 0
    map_fail = 0
    empty = 0

    for out_idx, page_idx in enumerate(valid_indices):
        page = simplified_pages[page_idx][2]
        map_html = simplified_pages[page_idx][1]
        reference = page.get('convert_main_content', '')
        level = page.get('meta', {}).get('level', 'unknown')

        response_text = outputs[out_idx].outputs[0].text

        try:
            labels = parse_llm_response(response_text)
        except Exception:
            parse_fail += 1
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        n_main = sum(1 for v in labels.values() if v == 'main')
        if n_main == 0:
            empty += 1
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        try:
            main_html = extract_main_html(map_html, labels)
            pred_text = html_to_text(main_html).strip()
        except Exception:
            map_fail += 1
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        if not pred_text:
            empty += 1
            r5 = 0.0
        elif not reference:
            r5 = 0.0
        else:
            r5 = rouge_n_f1(reference, pred_text, n=5)

        scores[page_idx] = r5
        scores_by_level.setdefault(level, []).append(r5)

    # Also score pages that failed simplification as 0
    for i, (simplified, _, page) in enumerate(simplified_pages):
        if simplified is None:
            level = page.get('meta', {}).get('level', 'unknown')
            scores_by_level.setdefault(level, []).append(0.0)

    # Report
    n = len(pages)
    # Only count valid pages for average
    valid_scores = [scores[i] for i in valid_indices]
    avg_valid = sum(valid_scores) / max(len(valid_scores), 1)
    avg_all = sum(scores) / max(n, 1)

    print(f'\n{"="*70}')
    print(f'DRIPPER 0.6B (native vLLM) — WebMainBench ROUGE-5 F1 (English, {n} pages)')
    print(f'{"="*70}')
    print(f'  Prompt: {PROMPT_VERSION}')
    print(f'  Max model len: {args.max_model_len}')
    print(f'  Valid pages: {len(valid_indices)}/{n} (too_long={too_long}, simp_fail={simp_fail})')
    print(f'  Parse failures: {parse_fail}')
    print(f'  Map failures: {map_fail}')
    print(f'  Empty extractions: {empty}')
    print(f'  Throughput: {len(outputs)/gen_time:.1f} pg/s')

    print(f'\n  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}')
    print(f'  {"-"*67}')

    level_avgs = {}
    for lev in ['simple', 'mid', 'hard']:
        vals = scores_by_level.get(lev, [])
        level_avgs[lev] = sum(vals) / max(len(vals), 1)

    comparisons = [
        (f'** Dripper native (valid only) **', avg_valid,
         level_avgs.get('simple', 0), level_avgs.get('mid', 0), level_avgs.get('hard', 0)),
        (f'** Dripper native (all {n} pages) **', avg_all, 0, 0, 0),
        ('Dripper 0.6B (paper)', 0.878, 0.921, 0.880, 0.831),
        ('DeepSeek V3.2 (short_compact)', 0.865, 0.932, 0.875, 0.786),
        ('Pulpie Orange Large (2.1B)', 0.862, 0.928, 0.856, 0.807),
        ('Pulpie Orange Base (0.6B)', 0.847, 0.907, 0.848, 0.787),
        ('DeepSeek V3.2 (v0, Bedrock)', 0.840, 0.930, 0.823, 0.774),
    ]
    comparisons.sort(key=lambda x: -x[1])
    for name, r_all, r_s, r_m, r_h in comparisons:
        if r_s == 0 and r_m == 0:
            print(f'  {name:<35} {r_all:>8.4f}')
        else:
            print(f'  {name:<35} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}')


if __name__ == '__main__':
    main()
