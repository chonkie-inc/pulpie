"""Compare qrater clean rates: pulpie vs Dripper vs Orange vs raw html2text.

Runs all four extraction methods on the same WMB pages:
1. Raw html2text (no extraction)
2. Pulpie GBM (espresso)
3. Dripper 0.6B (via local vLLM)
4. Pulpie Orange (encoder classifier)

Scores each output with qrater EuroBERT-210m classifier.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

import html2text
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')
HBIRD_BIN = os.path.join(SCRIPT_DIR, '..', 'target', 'release', 'pulpie')
QRATER_MODEL = os.path.join(SCRIPT_DIR, '..', '..', 'gym', 'qrater',
                             'models', 'encoder-distill',
                             'eurobert-210m_0.6b-labels', 'final')
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')

VLLM_URL = "http://localhost:8235/v1/chat/completions"
MODEL_NAME = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
ORANGE_MODEL_PATH = os.path.join(DATA_DIR, 'block_classifier_0.6B', 'final')
MAX_TOKENS = 4096
MAX_TEXT_CHARS = 10000
BLOCK_TOKEN = "[BLOCK]"
ORANGE_MAX_LENGTH = 32768

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


def html_to_text(html_str):
    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def extract_with_pulpie(html_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, tmp_path], capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ''
    finally:
        os.unlink(tmp_path)


def extract_item_ids(html_str):
    return [int(m) for m in re.findall(r'_item_id="(\d+)"', html_str)]


def build_guided_regex(item_ids):
    item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
    return f'<answer>\\s*{item_pattern}\\s*</answer>'


def extract_with_dripper(html_content):
    """Full Dripper pipeline: simplify → prompt → LLM → parse → reconstruct → html2text."""
    try:
        simplified, map_html = simplify_html(html_content)
    except Exception:
        return ''

    prompt = get_full_prompt(simplified, version='short_compact')
    item_ids = extract_item_ids(simplified)
    if not item_ids:
        return ''

    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "guided_regex": build_guided_regex(item_ids),
    }
    try:
        resp = requests.post(VLLM_URL, json=body, timeout=120)
        resp.raise_for_status()
        text = resp.json()['choices'][0]['message']['content']
    except Exception:
        return ''

    try:
        labels = parse_llm_response(text)
    except Exception:
        return ''

    try:
        main_html = extract_main_html(map_html, labels)
        return html_to_text(main_html).strip()
    except Exception:
        return ''


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


def load_orange_model(device):
    tok = AutoTokenizer.from_pretrained(ORANGE_MODEL_PATH, trust_remote_code=True)
    mdl = AutoModelForTokenClassification.from_pretrained(
        ORANGE_MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()
    for m in mdl.modules():
        if hasattr(m, 'is_causal'):
            m.is_causal = False
    block_token_id = tok.convert_tokens_to_ids(BLOCK_TOKEN)
    return mdl, tok, block_token_id


@torch.no_grad()
def extract_with_orange(html_content, latte_model, latte_tokenizer, block_token_id, device):
    try:
        simplified, map_html = simplify_html(html_content)
    except Exception:
        return ''

    marked_html, item_ids = insert_block_markers(simplified)
    if marked_html is None:
        return ''

    encoding = latte_tokenizer(
        marked_html, truncation=True, max_length=ORANGE_MAX_LENGTH,
        add_special_tokens=True, padding=False, return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    outputs = latte_model(input_ids=input_ids, attention_mask={'full_attention': None})
    logits = outputs.logits[0]

    block_positions = (input_ids[0] == block_token_id).nonzero(as_tuple=True)[0]
    preds = logits[block_positions].argmax(dim=-1).cpu().tolist()

    labels = {}
    for i, item_id in enumerate(item_ids):
        if i < len(preds):
            labels[item_id] = 'main' if preds[i] == 1 else 'other'
        else:
            labels[item_id] = 'other'

    n_main = sum(1 for v in labels.values() if v == 'main')
    if n_main == 0:
        return ''

    try:
        main_html = extract_main_html(map_html, labels)
        return html_to_text(main_html).strip()
    except Exception:
        return ''


def classify_batch(texts, model, tokenizer, batch_size=32):
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, truncation=True, max_length=4096,
                padding=True, return_tensors='pt',
            ).to(model.device)
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            probs = torch.softmax(logits, dim=-1).cpu()
            for j, pred in enumerate(preds):
                results.append({
                    'label': 'clean' if pred == 1 else 'dirty',
                    'confidence': probs[j][pred].item(),
                })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--skip-dripper', action='store_true', help='Skip Dripper (requires vLLM)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    print(f'Loading qrater model on {device}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QRATER_MODEL, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        QRATER_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    print(f'Loading Orange encoder on {device}...', flush=True)
    latte_model, latte_tokenizer, block_token_id = load_orange_model(device)

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

    # Extract with all four methods
    raw_texts, hbird_texts, dripper_texts, latte_texts = [], [], [], []
    dripper_fail = 0
    latte_fail = 0
    t0 = time.time()

    for i, page in enumerate(pages):
        html = page.get('html', '')

        # Raw
        raw_md = html_to_text(html)[:MAX_TEXT_CHARS]
        raw_texts.append(raw_md if raw_md.strip() else '')

        # Pulpie GBM (espresso)
        hbird_md = extract_with_pulpie(html)[:MAX_TEXT_CHARS]
        hbird_texts.append(hbird_md if hbird_md.strip() else '')

        # Dripper
        if not args.skip_dripper:
            drip_md = extract_with_dripper(html)[:MAX_TEXT_CHARS]
        else:
            drip_md = ''
        dripper_texts.append(drip_md if drip_md.strip() else '')
        if not drip_md.strip():
            dripper_fail += 1

        # Orange (encoder classifier)
        latte_md = extract_with_orange(html, latte_model, latte_tokenizer, block_token_id, device)[:MAX_TEXT_CHARS]
        latte_texts.append(latte_md if latte_md.strip() else '')
        if not latte_md.strip():
            latte_fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pages) - i - 1) / rate
            print(f'  {i+1}/{len(pages)} ({elapsed:.0f}s, {rate:.1f}pg/s, ETA={eta:.0f}s) '
                  f'dripper_fail={dripper_fail} latte_fail={latte_fail}', flush=True)

    print(f'\n  Done extracting in {time.time() - t0:.0f}s', flush=True)
    print(f'  Raw empty: {sum(1 for t in raw_texts if not t)}', flush=True)
    print(f'  Hbird empty: {sum(1 for t in hbird_texts if not t)}', flush=True)
    print(f'  Dripper empty: {sum(1 for t in dripper_texts if not t)}', flush=True)
    print(f'  Latte empty: {sum(1 for t in latte_texts if not t)}', flush=True)

    # Score with qrater
    print(f'\nScoring with qrater...', flush=True)

    def safe_texts(texts):
        return [t if t.strip() else '[empty]' for t in texts]

    raw_results = classify_batch(safe_texts(raw_texts), model, tokenizer)
    hbird_results = classify_batch(safe_texts(hbird_texts), model, tokenizer)
    dripper_results = classify_batch(safe_texts(dripper_texts), model, tokenizer)
    latte_results = classify_batch(safe_texts(latte_texts), model, tokenizer)

    # Force empty to dirty
    for texts_list, results_list in [
        (raw_texts, raw_results),
        (hbird_texts, hbird_results),
        (dripper_texts, dripper_results),
        (latte_texts, latte_results),
    ]:
        for i, t in enumerate(texts_list):
            if not t.strip():
                results_list[i] = {'label': 'dirty', 'confidence': 1.0}

    # Report
    n = len(pages)
    methods = [
        ('Raw html2text', raw_results),
        ('Pulpie Espresso (GBM)', hbird_results),
        ('Dripper 0.6B', dripper_results),
        ('Pulpie Orange (encoder)', latte_results),
    ]

    print(f'\n{"="*60}')
    print(f'QRATER CLEAN RATE (WMB English, {n} pages)')
    print(f'{"="*60}')
    print(f'  {"Method":<25} {"Clean":>6} {"Dirty":>6} {"Clean%":>8}')
    print(f'  {"-"*47}')
    for name, results in methods:
        clean = sum(1 for r in results if r['label'] == 'clean')
        print(f'  {name:<25} {clean:>6} {n - clean:>6} {clean/n*100:>7.1f}%')

    # By difficulty
    print(f'\n  By difficulty:')
    print(f'  {"Level":>7}  {"Raw":>10}  {"Espresso":>10}  {"Dripper":>10}  {"Orange":>10}')
    print(f'  {"-"*53}')
    for level in ['simple', 'mid', 'hard']:
        idx = [i for i, p in enumerate(pages) if p.get('meta', {}).get('level') == level]
        if not idx:
            continue
        vals = []
        for _, results in methods:
            c = sum(1 for i in idx if results[i]['label'] == 'clean')
            vals.append(f'{c}/{len(idx)} ({c/len(idx)*100:.0f}%)')
        print(f'  {level:>7}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}  {vals[3]:>10}')

    # ROUGE-5 scoring (if pages have reference text)
    from collections import Counter

    def _ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def rouge_n_f1(reference, prediction, n=5):
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        if not ref_tokens or not pred_tokens:
            return 0.0
        ref_ng = Counter(_ngrams(ref_tokens, n))
        pred_ng = Counter(_ngrams(pred_tokens, n))
        if not ref_ng or not pred_ng:
            return 0.0
        overlap = sum((ref_ng & pred_ng).values())
        prec = overlap / max(sum(pred_ng.values()), 1)
        rec = overlap / max(sum(ref_ng.values()), 1)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    has_ref = any(p.get('convert_main_content') for p in pages)
    if has_ref:
        print(f'\n{"="*60}')
        print(f'ROUGE-5 F1 (WMB English, {n} pages)')
        print(f'{"="*60}')

        text_methods = [
            ('Raw html2text', raw_texts),
            ('Pulpie Espresso (GBM)', hbird_texts),
            ('Dripper 0.6B', dripper_texts),
            ('Pulpie Orange (encoder)', latte_texts),
        ]

        print(f'  {"Method":<30} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}')
        print(f'  {"-"*62}')
        for name, texts in text_methods:
            scores_all = []
            scores_by_level = {}
            for i, page in enumerate(pages):
                ref = page.get('convert_main_content', '')
                level = page.get('meta', {}).get('level', 'unknown')
                r5 = rouge_n_f1(ref, texts[i]) if ref and texts[i] else 0.0
                scores_all.append(r5)
                scores_by_level.setdefault(level, []).append(r5)
            avg_all = sum(scores_all) / max(len(scores_all), 1)
            avgs = {}
            for lev in ['simple', 'mid', 'hard']:
                vals = scores_by_level.get(lev, [])
                avgs[lev] = sum(vals) / max(len(vals), 1)
            print(f'  {name:<30} {avg_all:>8.4f} {avgs["simple"]:>8.4f} {avgs["mid"]:>8.4f} {avgs["hard"]:>8.4f}')


if __name__ == '__main__':
    main()
