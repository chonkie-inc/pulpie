"""Comprehensive WebMainBench benchmark: all extraction methods.

Evaluates ROUGE-5 F1 on the English-only subset (6,647 pages) for:
  - Pulpie orange-small (210M), orange-base (610M), orange-large (2.1B)
  - Dripper 0.6B (inline vLLM with guided decoding)
  - Trafilatura
  - magic-html
  - Raw html2text (lower bound)

Usage:
  python eval/bench_all_methods.py --methods pulpie-small,trafilatura,raw-h2t --limit 50
  python eval/bench_all_methods.py --methods all --limit 0
  python eval/bench_all_methods.py --methods dripper --limit 200 --device cuda:0
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter

import html2text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
MINERU_PATH = os.path.join(ROOT_DIR, '..', 'MinerU-HTML')

ALL_METHODS = [
    'dripper', 'pulpie-large', 'pulpie-base', 'pulpie-small',
    'trafilatura', 'magic-html', 'raw-h2t',
]

# Local model paths (fallback when HF repos are private/unavailable)
LOCAL_MODELS = {
    'orange-small': os.path.join(DATA_DIR, 'block_classifier_eurobert_210m_distill', 'final'),
    'orange-base': os.path.join(DATA_DIR, 'block_classifier_eurobert_610m_distill', 'final'),
    'orange-large': os.path.join(DATA_DIR, 'block_classifier_eurobert_2.1B', 'checkpoint-5250'),
}


# ── Metrics ──

def html_to_text(html_str):
    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge5_scores(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return {'f1': 0.0, 'p': 0.0, 'r': 0.0}
    ref_ng = Counter(ngrams(ref_tokens, 5))
    pred_ng = Counter(ngrams(pred_tokens, 5))
    if not ref_ng or not pred_ng:
        return {'f1': 0.0, 'p': 0.0, 'r': 0.0}
    overlap = sum((ref_ng & pred_ng).values())
    p = overlap / max(sum(pred_ng.values()), 1)
    r = overlap / max(sum(ref_ng.values()), 1)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'f1': f1, 'p': p, 'r': r}


# ── Dataset ──

def load_pages(limit=0):
    pages = []
    with open(WMB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                pages.append(rec)
    if limit > 0:
        pages = pages[:limit]
    return pages


# ── Cache ──

def load_cache(output_dir, method_name):
    path = os.path.join(output_dir, f'{method_name}_scores.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(output_dir, method_name, scores):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{method_name}_scores.json')
    with open(path, 'w') as f:
        json.dump(scores, f)


# ── MinerU-HTML module patching (for Dripper) ──

def setup_mineru_modules():
    import importlib.util
    from dataclasses import dataclass, field
    from typing import Optional
    from enum import Enum

    def _make_module(name):
        mod = type(sys)(name)
        sys.modules[name] = mod
        return mod

    if 'mineru_html' in sys.modules:
        return

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


# ── Method runners ──

def run_pulpie(pages, model_name, device, output_dir, no_cache=False):
    display_name = f'pulpie-{model_name.split("-")[-1]}'
    cache = {} if no_cache else load_cache(output_dir, display_name)
    pending = [i for i in range(len(pages)) if str(i) not in cache]

    if not pending:
        print(f'  All {len(pages)} pages cached.', flush=True)
        return cache

    print(f'  {len(pending)} pages to process ({len(cache)} cached)...', flush=True)

    # Use MinerU-HTML simplify + reconstruct (what the model was trained on).
    # Pulpie's built-in simplify/reconstruct differ and produce lower scores.
    setup_mineru_modules()
    simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
    extract_main_html_fn = sys.modules['mineru_html.process.map_to_main'].extract_main_html

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from pulpie.chunker import extract_blocks, tokenize_blocks, pack_chunks, SEP_TOKEN

    # Load model directly (same as old eval that got 0.864)
    model_path = LOCAL_MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [SEP_TOKEN]})
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    t0 = time.time()
    for count, i in enumerate(pending):
        html_content = pages[i].get('html', '')
        reference = pages[i].get('convert_main_content', '')

        try:
            simplified, map_html = simplify_html(html_content)
            blocks = extract_blocks(simplified)
            if not blocks:
                raise ValueError('no blocks')

            item_ids = []
            for block in blocks:
                m = re.search(r'_item_id="(\d+)"', block)
                item_ids.append(m.group(1) if m else None)

            block_token_ids = tokenize_blocks(blocks, tokenizer)
            chunks = pack_chunks(block_token_ids, max_tokens=8192,
                                 sep_token_id=sep_token_id,
                                 bos_token_id=tokenizer.bos_token_id,
                                 eos_token_id=tokenizer.eos_token_id)

            predictions = [0] * len(blocks)
            with torch.no_grad():
                for chunk_ids, block_indices in chunks:
                    input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
                    attn = torch.ones_like(input_ids)
                    outputs = model(input_ids=input_ids, attention_mask=attn)
                    logits = outputs.logits[0]
                    sep_pos = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
                    preds = logits[sep_pos].argmax(dim=-1).cpu().tolist()
                    for j, bi in enumerate(block_indices):
                        if j < len(preds):
                            predictions[bi] = preds[j]

            labels = {}
            for idx, item_id in enumerate(item_ids):
                if item_id is not None:
                    labels[item_id] = 'main' if predictions[idx] == 1 else 'other'

            main_html = extract_main_html_fn(map_html, labels)
            pred_text = html_to_text(main_html).strip()
        except Exception:
            pred_text = ''

        scores = rouge5_scores(reference, pred_text)
        scores['empty'] = len(pred_text.strip()) == 0
        cache[str(i)] = scores

        if (count + 1) % 100 == 0:
            elapsed = time.time() - t0
            save_cache(output_dir, display_name, cache)
            print(f'    {count+1}/{len(pending)} ({elapsed:.0f}s, '
                  f'{(count+1)/elapsed:.1f} pg/s)', flush=True)

    elapsed = time.time() - t0
    save_cache(output_dir, display_name, cache)
    print(f'  Done: {len(pending)} pages in {elapsed:.0f}s '
          f'({len(pending)/elapsed:.1f} pg/s)', flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return cache


def run_dripper(pages, device, output_dir, max_model_len=32768, no_cache=False):
    cache = {} if no_cache else load_cache(output_dir, 'dripper')
    pending = [i for i in range(len(pages)) if str(i) not in cache]

    if not pending:
        print(f'  All {len(pages)} pages cached.', flush=True)
        return cache

    print(f'  {len(pending)} pages to process ({len(cache)} cached)...', flush=True)

    setup_mineru_modules()
    simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
    get_full_prompt = sys.modules['mineru_html.process.build_prompt'].get_full_prompt
    parse_llm_response = sys.modules['mineru_html.process.parse_result'].parse_llm_response
    extract_main_html_fn = sys.modules['mineru_html.process.map_to_main'].extract_main_html

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer

    dripper_model = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
    tokenizer = AutoTokenizer.from_pretrained(dripper_model)

    # Simplify and build prompts for pending pages
    print(f'  Simplifying HTML...', flush=True)
    prompts = []
    item_ids_list = []
    valid_pending = []
    too_long = 0
    simp_fail = 0

    for i in pending:
        html = pages[i].get('html', '')
        try:
            simplified, map_html = simplify_html(html)
        except Exception:
            simp_fail += 1
            cache[str(i)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        try:
            prompt = get_full_prompt(simplified, version='short_compact')
        except Exception:
            cache[str(i)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        item_ids = [int(m) for m in re.findall(r'_item_id="(\d+)"', simplified)]
        if not item_ids:
            cache[str(i)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=True,
        )

        token_ids = tokenizer.encode(chat_prompt)
        if len(token_ids) > max_model_len - 1000:
            too_long += 1
            cache[str(i)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        prompts.append(chat_prompt)
        item_ids_list.append(item_ids)
        valid_pending.append((i, map_html))

    print(f'  {len(prompts)} valid prompts, {too_long} too long, {simp_fail} simp fail', flush=True)

    if not prompts:
        save_cache(output_dir, 'dripper', cache)
        return cache

    # Extract GPU index from device string
    gpu_idx = 0
    if 'cuda:' in device:
        gpu_idx = int(device.split(':')[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)

    print(f'  Loading vLLM (max_len={max_model_len})...', flush=True)
    llm = LLM(
        model=dripper_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        max_model_len=max_model_len,
    )

    sampling_params_list = []
    for item_ids in item_ids_list:
        item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
        pattern = f'<answer>\\s*{item_pattern}\\s*</answer>'
        structured_outputs_params = StructuredOutputsParams(regex=pattern)
        sampling_params_list.append(SamplingParams(
            structured_outputs=structured_outputs_params,
            top_k=1, top_p=0.95, temperature=0, max_tokens=16384,
        ))

    print(f'  Running inference...', flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params_list)
    gen_time = time.time() - t0
    print(f'  Generated {len(outputs)} in {gen_time:.0f}s ({len(outputs)/gen_time:.1f} pg/s)', flush=True)

    # Parse and score
    parse_fail = 0
    for out_idx, (page_idx, map_html) in enumerate(valid_pending):
        reference = pages[page_idx].get('convert_main_content', '')
        response_text = outputs[out_idx].outputs[0].text

        try:
            labels = parse_llm_response(response_text)
        except Exception:
            parse_fail += 1
            cache[str(page_idx)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        try:
            main_html = extract_main_html_fn(map_html, labels)
            pred_text = html_to_text(main_html).strip()
        except Exception:
            cache[str(page_idx)] = {'f1': 0.0, 'p': 0.0, 'r': 0.0, 'empty': True}
            continue

        scores = rouge5_scores(reference, pred_text)
        scores['empty'] = len(pred_text) == 0
        cache[str(page_idx)] = scores

    if parse_fail:
        print(f'  {parse_fail} parse failures', flush=True)

    save_cache(output_dir, 'dripper', cache)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return cache


def run_trafilatura(pages, output_dir, no_cache=False):
    cache = {} if no_cache else load_cache(output_dir, 'trafilatura')
    pending = [i for i in range(len(pages)) if str(i) not in cache]

    if not pending:
        print(f'  All {len(pages)} pages cached.', flush=True)
        return cache

    print(f'  {len(pending)} pages to process ({len(cache)} cached)...', flush=True)

    try:
        import trafilatura
    except ImportError:
        print('  [SKIP] trafilatura not installed (pip install trafilatura)', flush=True)
        return None

    t0 = time.time()
    for count, i in enumerate(pending):
        html = pages[i].get('html', '')
        reference = pages[i].get('convert_main_content', '')

        try:
            pred_text = trafilatura.extract(
                html, include_tables=True, include_comments=False
            ) or ''
        except Exception:
            pred_text = ''

        scores = rouge5_scores(reference, pred_text)
        scores['empty'] = len(pred_text.strip()) == 0
        cache[str(i)] = scores

        if (count + 1) % 500 == 0:
            elapsed = time.time() - t0
            save_cache(output_dir, 'trafilatura', cache)
            print(f'    {count+1}/{len(pending)} ({elapsed:.0f}s)', flush=True)

    elapsed = time.time() - t0
    save_cache(output_dir, 'trafilatura', cache)
    print(f'  Done: {len(pending)} pages in {elapsed:.0f}s '
          f'({len(pending)/elapsed:.1f} pg/s)', flush=True)
    return cache


def run_magic_html(pages, output_dir, no_cache=False):
    cache = {} if no_cache else load_cache(output_dir, 'magic-html')
    pending = [i for i in range(len(pages)) if str(i) not in cache]

    if not pending:
        print(f'  All {len(pages)} pages cached.', flush=True)
        return cache

    print(f'  {len(pending)} pages to process ({len(cache)} cached)...', flush=True)

    try:
        from magic_html import GeneralExtractor
    except ImportError:
        print('  [SKIP] magic-html not installed (pip install magic-html)', flush=True)
        return None

    extractor = GeneralExtractor()

    t0 = time.time()
    for count, i in enumerate(pending):
        html = pages[i].get('html', '')
        reference = pages[i].get('convert_main_content', '')

        try:
            result = extractor.extract(html, base_url='')
            extracted_html = result.get('html', '') if isinstance(result, dict) else (result or '')
            pred_text = html_to_text(extracted_html).strip()
        except Exception:
            pred_text = ''

        scores = rouge5_scores(reference, pred_text)
        scores['empty'] = len(pred_text.strip()) == 0
        cache[str(i)] = scores

        if (count + 1) % 500 == 0:
            elapsed = time.time() - t0
            save_cache(output_dir, 'magic-html', cache)
            print(f'    {count+1}/{len(pending)} ({elapsed:.0f}s)', flush=True)

    elapsed = time.time() - t0
    save_cache(output_dir, 'magic-html', cache)
    print(f'  Done: {len(pending)} pages in {elapsed:.0f}s '
          f'({len(pending)/elapsed:.1f} pg/s)', flush=True)
    return cache


def run_raw_h2t(pages, output_dir, no_cache=False):
    cache = {} if no_cache else load_cache(output_dir, 'raw-h2t')
    pending = [i for i in range(len(pages)) if str(i) not in cache]

    if not pending:
        print(f'  All {len(pages)} pages cached.', flush=True)
        return cache

    print(f'  {len(pending)} pages to process ({len(cache)} cached)...', flush=True)

    t0 = time.time()
    for count, i in enumerate(pending):
        html = pages[i].get('html', '')
        reference = pages[i].get('convert_main_content', '')

        try:
            pred_text = html_to_text(html).strip()
        except Exception:
            pred_text = ''

        scores = rouge5_scores(reference, pred_text)
        scores['empty'] = len(pred_text.strip()) == 0
        cache[str(i)] = scores

    elapsed = time.time() - t0
    save_cache(output_dir, 'raw-h2t', cache)
    print(f'  Done: {len(pending)} pages in {elapsed:.0f}s', flush=True)
    return cache


# ── Reporting ──

def report_results(all_results, pages):
    levels = ['simple', 'mid', 'hard']
    n = len(pages)

    print(f'\n{"="*80}')
    print(f'WebMainBench ROUGE-5 F1 (English-only, {n} pages)')
    print(f'{"="*80}')
    print(f'  {"Method":<28} {"All":>7} {"Simple":>7} {"Mid":>7} '
          f'{"Hard":>7} {"Empty":>6} {"P":>6} {"R":>6}')
    print(f'  {"-"*78}')

    rows = []
    for method_name, scores in all_results.items():
        if scores is None:
            continue

        f1_all = []
        p_all = []
        r_all = []
        f1_by_level = {l: [] for l in levels}
        empty_count = 0

        for i in range(n):
            s = scores.get(str(i))
            if s is None:
                continue
            f1_all.append(s['f1'])
            p_all.append(s['p'])
            r_all.append(s['r'])
            if s.get('empty', False):
                empty_count += 1
            level = pages[i].get('meta', {}).get('level', '')
            if level in f1_by_level:
                f1_by_level[level].append(s['f1'])

        if not f1_all:
            continue

        avg_f1 = sum(f1_all) / len(f1_all)
        avg_p = sum(p_all) / len(p_all)
        avg_r = sum(r_all) / len(r_all)
        level_f1 = {}
        for l in levels:
            if f1_by_level[l]:
                level_f1[l] = sum(f1_by_level[l]) / len(f1_by_level[l])
            else:
                level_f1[l] = 0.0

        rows.append({
            'name': method_name, 'f1': avg_f1, 'p': avg_p, 'r': avg_r,
            'simple': level_f1['simple'], 'mid': level_f1['mid'],
            'hard': level_f1['hard'], 'empty': empty_count,
            'scored': len(f1_all),
        })

    rows.sort(key=lambda x: -x['f1'])

    for row in rows:
        scored_note = f' ({row["scored"]})' if row['scored'] < n else ''
        print(f'  {row["name"]:<28} {row["f1"]:>7.3f} {row["simple"]:>7.3f} '
              f'{row["mid"]:>7.3f} {row["hard"]:>7.3f} {row["empty"]:>6}{scored_note}'
              f' {row["p"]:>6.3f} {row["r"]:>6.3f}')

    # F1 distribution for top method
    if rows:
        print(f'\n  F1 distribution ({rows[0]["name"]}):')
        top_scores = all_results[rows[0]['name']]
        bins = [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
        for lo, hi in bins:
            count = sum(1 for i in range(n) if str(i) in top_scores
                        and lo <= top_scores[str(i)]['f1'] < hi)
            pct = count / n * 100
            label = f'[{lo:.1f}, {hi:.1f})'
            bar = '#' * int(pct / 2)
            print(f'    {label}: {count:>5} ({pct:>5.1f}%) {bar}')


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Comprehensive WMB benchmark')
    parser.add_argument('--methods', type=str, default='all',
                        help=f'Comma-separated methods or "all". Options: {",".join(ALL_METHODS)}')
    parser.add_argument('--limit', type=int, default=0, help='0 = all English pages')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--max-model-len', type=int, default=32768,
                        help='vLLM max_model_len for Dripper')
    args = parser.parse_args()

    if args.methods == 'all':
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(',')]

    print(f'Loading WebMainBench (English only)...', flush=True)
    pages = load_pages(args.limit)
    print(f'  {len(pages)} pages', flush=True)

    all_results = {}

    for method in methods:
        print(f'\n{"─"*60}', flush=True)
        print(f'Running: {method}', flush=True)
        print(f'{"─"*60}', flush=True)

        if method == 'dripper':
            try:
                import vllm  # noqa: F401
                result = run_dripper(pages, args.device, args.output_dir,
                                     max_model_len=args.max_model_len,
                                     no_cache=args.no_cache)
            except ImportError:
                print('  [SKIP] vllm not installed', flush=True)
                result = None
            all_results['Dripper 0.6B'] = result

        elif method == 'pulpie-small':
            result = run_pulpie(pages, 'orange-small', args.device,
                                args.output_dir, no_cache=args.no_cache)
            all_results['Pulpie Small (210M)'] = result

        elif method == 'pulpie-base':
            result = run_pulpie(pages, 'orange-base', args.device,
                                args.output_dir, no_cache=args.no_cache)
            all_results['Pulpie Base (610M)'] = result

        elif method == 'pulpie-large':
            result = run_pulpie(pages, 'orange-large', args.device,
                                args.output_dir, no_cache=args.no_cache)
            all_results['Pulpie Large (2.1B)'] = result

        elif method == 'trafilatura':
            result = run_trafilatura(pages, args.output_dir, no_cache=args.no_cache)
            all_results['Trafilatura'] = result

        elif method == 'magic-html':
            result = run_magic_html(pages, args.output_dir, no_cache=args.no_cache)
            all_results['magic-html'] = result

        elif method == 'raw-h2t':
            result = run_raw_h2t(pages, args.output_dir, no_cache=args.no_cache)
            all_results['Raw html2text'] = result

        else:
            print(f'  [SKIP] Unknown method: {method}', flush=True)

    report_results(all_results, pages)


if __name__ == '__main__':
    main()
