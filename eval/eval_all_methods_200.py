"""Compare all methods on same 200 WMB English pages.

Runs: Dripper 0.6B, Orange Base (0.6B), Orange Large (2.1B)
DeepSeek results already obtained separately.
"""

import json
import os
import re
import sys
import time
from collections import Counter

import html2text
import requests
import torch
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
    _load_file('mineru_html.process.build_prompt', 'build_prompt.py')
    _load_file('mineru_html.process.parse_result', 'parse_result.py')
    _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
get_full_prompt = sys.modules['mineru_html.process.build_prompt'].get_full_prompt
parse_llm_response = sys.modules['mineru_html.process.parse_result'].parse_llm_response
extract_main_html = sys.modules['mineru_html.process.map_to_main'].extract_main_html

from pulpie.chunker import extract_blocks, tokenize_blocks, pack_chunks, SEP_TOKEN

# ── Config ──
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
VLLM_URL = "http://localhost:8235/v1/chat/completions"
DRIPPER_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
ORANGE_BASE_PATH = os.path.join(DATA_DIR, 'block_classifier_0.6B', 'final')
ORANGE_LARGE_PATH = os.path.join(DATA_DIR, 'block_classifier_eurobert_2.1B', 'checkpoint-5250')
BLOCK_TOKEN = "[BLOCK]"
MAX_TOKENS_DRIPPER = 4096
ORANGE_BASE_MAX_LENGTH = 32768
ORANGE_LARGE_MAX_TOKENS = 8192


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


def extract_item_ids(html_str):
    return [int(m) for m in re.findall(r'_item_id="(\d+)"', html_str)]


def build_guided_regex(item_ids):
    item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
    return f'<answer>\\s*{item_pattern}\\s*</answer>'


# ── Dripper ──

def extract_with_dripper(simplified, map_html):
    prompt = get_full_prompt(simplified, version='short_compact')
    item_ids = extract_item_ids(simplified)
    if not item_ids:
        return None

    body = {
        "model": DRIPPER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS_DRIPPER,
        "temperature": 0,
        "guided_regex": build_guided_regex(item_ids),
    }
    try:
        resp = requests.post(VLLM_URL, json=body, timeout=120)
        resp.raise_for_status()
        text = resp.json()['choices'][0]['message']['content']
    except Exception:
        return None

    try:
        labels = parse_llm_response(text)
    except Exception:
        return None

    try:
        main_html = extract_main_html(map_html, labels)
        return html_to_text(main_html).strip()
    except Exception:
        return None


# ── Orange Base (0.6B, [BLOCK] marker) ──

def insert_block_markers(simplified_html):
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
    return ''.join(parts), item_ids


@torch.no_grad()
def classify_orange_base(model, tokenizer, block_token_id, simplified, device):
    marked_html, item_ids = insert_block_markers(simplified)
    if marked_html is None:
        return {}
    encoding = tokenizer(
        marked_html, truncation=True, max_length=ORANGE_BASE_MAX_LENGTH,
        add_special_tokens=True, padding=False, return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    outputs = model(input_ids=input_ids, attention_mask={'full_attention': None})
    logits = outputs.logits[0]
    block_positions = (input_ids[0] == block_token_id).nonzero(as_tuple=True)[0]
    preds = logits[block_positions].argmax(dim=-1).cpu().tolist()
    labels = {}
    for i, item_id in enumerate(item_ids):
        labels[item_id] = 'main' if (i < len(preds) and preds[i] == 1) else 'other'
    return labels


# ── Orange Large (2.1B, <|sep|> chunking) ──

@torch.no_grad()
def classify_orange_large(model, tokenizer, sep_token_id, simplified, device):
    blocks = extract_blocks(simplified)
    if not blocks:
        return {}

    item_id_pattern = re.compile(r'_item_id="(\d+)"')
    block_item_ids = []
    for block in blocks:
        m = item_id_pattern.search(block)
        block_item_ids.append(m.group(1) if m else None)

    block_token_ids = tokenize_blocks(blocks, tokenizer)
    chunks = pack_chunks(
        block_token_ids, max_tokens=ORANGE_LARGE_MAX_TOKENS,
        sep_token_id=sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    labels = {}
    for chunk_ids, chunk_block_indices in chunks:
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits[0]
        sep_positions = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
        preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()
        for sep_idx, block_idx in enumerate(chunk_block_indices):
            if sep_idx < len(preds) and block_idx < len(block_item_ids):
                item_id = block_item_ids[block_idx]
                if item_id is not None:
                    labels[item_id] = 'main' if preds[sep_idx] == 1 else 'other'
    return labels


def labels_to_text(labels, map_html):
    n_main = sum(1 for v in labels.values() if v == 'main')
    if n_main == 0:
        return ''
    try:
        main_html = extract_main_html(map_html, labels)
        return html_to_text(main_html).strip()
    except Exception:
        return ''


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

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

    # Load Orange Base
    print(f'\nLoading Orange Base (0.6B) on {device}...', flush=True)
    base_tokenizer = AutoTokenizer.from_pretrained(ORANGE_BASE_PATH, trust_remote_code=True)
    base_model = AutoModelForTokenClassification.from_pretrained(
        ORANGE_BASE_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()
    for m in base_model.modules():
        if hasattr(m, 'is_causal'):
            m.is_causal = False
    block_token_id = base_tokenizer.convert_tokens_to_ids(BLOCK_TOKEN)
    print(f'  [BLOCK] id = {block_token_id}', flush=True)

    # Load Orange Large
    print(f'Loading Orange Large (2.1B) on {device}...', flush=True)
    large_tokenizer = AutoTokenizer.from_pretrained(ORANGE_LARGE_PATH, trust_remote_code=True)
    large_model = AutoModelForTokenClassification.from_pretrained(
        ORANGE_LARGE_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()
    sep_token_id = large_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f'  <|sep|> id = {sep_token_id}', flush=True)

    # Run all methods
    print(f'\nRunning all methods...', flush=True)
    dripper_scores, base_scores, large_scores = [], [], []
    dripper_by_level, base_by_level, large_by_level = {}, {}, {}
    dripper_fail, base_fail, large_fail = 0, 0, 0
    t0 = time.time()

    for i, page in enumerate(pages):
        html_content = page.get('html', '')
        reference = page.get('convert_main_content', '')
        level = page.get('meta', {}).get('level', 'unknown')

        if not html_content or not reference:
            for scores, by_level in [(dripper_scores, dripper_by_level),
                                      (base_scores, base_by_level),
                                      (large_scores, large_by_level)]:
                scores.append(0.0)
                by_level.setdefault(level, []).append(0.0)
            continue

        try:
            simplified, map_html = simplify_html(html_content)
        except Exception:
            for scores, by_level in [(dripper_scores, dripper_by_level),
                                      (base_scores, base_by_level),
                                      (large_scores, large_by_level)]:
                scores.append(0.0)
                by_level.setdefault(level, []).append(0.0)
            continue

        # Dripper
        drip_text = extract_with_dripper(simplified, map_html)
        if drip_text is None:
            drip_text = ''
            dripper_fail += 1
        r5 = rouge_n_f1(reference, drip_text) if drip_text else 0.0
        dripper_scores.append(r5)
        dripper_by_level.setdefault(level, []).append(r5)

        # Orange Base
        base_labels = classify_orange_base(base_model, base_tokenizer, block_token_id, simplified, device)
        base_text = labels_to_text(base_labels, map_html)
        if not base_text:
            base_fail += 1
        r5 = rouge_n_f1(reference, base_text) if base_text else 0.0
        base_scores.append(r5)
        base_by_level.setdefault(level, []).append(r5)

        # Orange Large
        large_labels = classify_orange_large(large_model, large_tokenizer, sep_token_id, simplified, device)
        large_text = labels_to_text(large_labels, map_html)
        if not large_text:
            large_fail += 1
        r5 = rouge_n_f1(reference, large_text) if large_text else 0.0
        large_scores.append(r5)
        large_by_level.setdefault(level, []).append(r5)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pages)} ({elapsed:.0f}s) '
                  f'dripper_fail={dripper_fail} base_fail={base_fail} large_fail={large_fail}',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\n  Done in {elapsed:.0f}s', flush=True)

    # Report
    n = len(pages)
    print(f'\n{"="*70}')
    print(f'ALL METHODS — WebMainBench ROUGE-5 F1 (English, {n} pages, same sample)')
    print(f'{"="*70}')

    methods = [
        ('Dripper 0.6B (local vLLM)', dripper_scores, dripper_by_level, dripper_fail),
        ('Pulpie Orange Base (0.6B)', base_scores, base_by_level, base_fail),
        ('Pulpie Orange Large (2.1B)', large_scores, large_by_level, large_fail),
    ]

    print(f'\n  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8} {"Empty":>6}')
    print(f'  {"-"*73}')

    rows = []
    for name, scores, by_level, fail in methods:
        avg = sum(scores) / max(len(scores), 1)
        avgs = {}
        for lev in ['simple', 'mid', 'hard']:
            vals = by_level.get(lev, [])
            avgs[lev] = sum(vals) / max(len(vals), 1)
        rows.append((name, avg, avgs['simple'], avgs['mid'], avgs['hard'], fail))

    # Add reference numbers
    rows.append(('DeepSeek V3.2 (v0, Bedrock)', 0.840, 0.930, 0.823, 0.774, 6))
    rows.append(('DeepSeek V3.2 (short_compact)', 0.865, 0.932, 0.875, 0.786, 0))
    rows.append(('DeepSeek V3.2 (paper)', 0.910, 0.942, 0.910, 0.877, 0))
    rows.append(('Dripper 0.6B (paper)', 0.878, 0.921, 0.880, 0.831, 0))

    rows.sort(key=lambda x: -x[1])
    for name, avg, s, m, h, fail in rows:
        print(f'  {name:<35} {avg:>8.4f} {s:>8.4f} {m:>8.4f} {h:>8.4f} {fail:>6}')


if __name__ == '__main__':
    main()
