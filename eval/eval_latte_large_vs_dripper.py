"""Compare Pulpie Orange Large (2.1B) vs Dripper on WMB with ROUGE-5 + qrater.

Runs both methods on same pages, scores with:
1. ROUGE-5 F1 (whitespace tokenized)
2. Qrater clean rate (EuroBERT-210m classifier)

Usage:
  python eval/eval_latte_large_vs_dripper.py --limit 500 --gpu 0
"""

import json
import os
import re
import sys
import time
from collections import Counter

import html2text
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

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
DRIPPER_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
ORANGE_LARGE_PATH = os.path.join(DATA_DIR, 'block_classifier_eurobert_2.1B', 'checkpoint-5250')
QRATER_MODEL = os.path.join(SCRIPT_DIR, '..', '..', 'gym', 'qrater',
                             'models', 'encoder-distill',
                             'eurobert-210m_0.6b-labels', 'final')
ORANGE_LARGE_MAX_TOKENS = 8192
DRIPPER_MAX_MODEL_LEN = 32768
DRIPPER_MAX_TOKENS = 16384
MAX_TEXT_CHARS = 10000


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


def classify_batch_qrater(texts, model, tokenizer, batch_size=32):
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
            for pred in preds:
                results.append('clean' if pred == 1 else 'dirty')
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-dripper', action='store_true')
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

    # Simplify all pages
    print(f'\nSimplifying HTML...', flush=True)
    simplified_pages = []
    simp_fail = 0
    for page in pages:
        try:
            simplified, map_html = simplify_html(page['html'])
            simplified_pages.append((simplified, map_html))
        except Exception:
            simplified_pages.append((None, None))
            simp_fail += 1
    print(f'  {len(pages) - simp_fail} ok, {simp_fail} failed', flush=True)

    # ── Orange Large ──
    print(f'\nLoading Orange Large (2.1B) on {device}...', flush=True)
    large_tokenizer = AutoTokenizer.from_pretrained(ORANGE_LARGE_PATH, trust_remote_code=True)
    large_model = AutoModelForTokenClassification.from_pretrained(
        ORANGE_LARGE_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()
    sep_token_id = large_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    print(f'  <|sep|> id = {sep_token_id}', flush=True)

    print(f'\nRunning Orange Large...', flush=True)
    latte_texts = []
    latte_fail = 0
    t0 = time.time()
    for i, (simplified, map_html) in enumerate(simplified_pages):
        if simplified is None:
            latte_texts.append('')
            latte_fail += 1
            continue
        labels = classify_orange_large(large_model, large_tokenizer, sep_token_id, simplified, device)
        text = labels_to_text(labels, map_html)
        latte_texts.append(text[:MAX_TEXT_CHARS])
        if not text:
            latte_fail += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pages)} ({elapsed:.0f}s, {(i+1)/elapsed:.1f}pg/s) fail={latte_fail}', flush=True)
    latte_time = time.time() - t0
    print(f'  Done in {latte_time:.0f}s ({len(pages)/latte_time:.1f} pg/s), fail={latte_fail}', flush=True)

    # Free Orange Large from GPU
    del large_model
    torch.cuda.empty_cache()

    # ── Dripper (native vLLM) ──
    dripper_texts = [''] * len(pages)
    if not args.skip_dripper:
        print(f'\nLoading Dripper (native vLLM, {DRIPPER_MAX_MODEL_LEN} context)...', flush=True)
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        dripper_tokenizer = AutoTokenizer.from_pretrained(DRIPPER_MODEL)

        # Build prompts
        prompts = []
        item_ids_list = []
        valid_indices = []
        too_long = 0

        for i, (simplified, map_html) in enumerate(simplified_pages):
            if simplified is None:
                continue
            try:
                prompt = get_full_prompt(simplified, version='short_compact')
            except Exception:
                continue
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ]
            chat_prompt = dripper_tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False, add_generation_prompt=True,
            )
            item_ids = [int(m) for m in re.findall(r'_item_id="(\d+)"', simplified)]
            if not item_ids:
                continue
            token_ids = dripper_tokenizer.encode(chat_prompt)
            if len(token_ids) > DRIPPER_MAX_MODEL_LEN - 1000:
                too_long += 1
                continue
            prompts.append(chat_prompt)
            item_ids_list.append(item_ids)
            valid_indices.append(i)

        print(f'  {len(prompts)} valid prompts, {too_long} too long', flush=True)

        llm = LLM(
            model=DRIPPER_MODEL,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            max_model_len=DRIPPER_MAX_MODEL_LEN,
        )

        sampling_params_list = []
        for item_ids in item_ids_list:
            item_pattern = ''.join(f'{i}(main|other)' for i in item_ids)
            pattern = f'<answer>\\s*{item_pattern}\\s*</answer>'
            sampling_params_list.append(SamplingParams(
                structured_outputs=StructuredOutputsParams(regex=pattern),
                top_k=1, top_p=0.95, temperature=0, max_tokens=DRIPPER_MAX_TOKENS,
            ))

        print(f'  Running inference...', flush=True)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling_params_list)
        dripper_time = time.time() - t0
        print(f'  Generated {len(outputs)} in {dripper_time:.0f}s ({len(outputs)/dripper_time:.1f} pg/s)', flush=True)

        dripper_fail = 0
        for out_idx, page_idx in enumerate(valid_indices):
            map_html = simplified_pages[page_idx][1]
            response_text = outputs[out_idx].outputs[0].text
            try:
                labels = parse_llm_response(response_text)
                text = labels_to_text(labels, map_html)
                dripper_texts[page_idx] = text[:MAX_TEXT_CHARS]
                if not text:
                    dripper_fail += 1
            except Exception:
                dripper_fail += 1

        print(f'  Dripper fail={dripper_fail}', flush=True)
        del llm
        torch.cuda.empty_cache()

    # ── Qrater scoring ──
    print(f'\nLoading qrater model on {device}...', flush=True)
    qrater_tokenizer = AutoTokenizer.from_pretrained(QRATER_MODEL, trust_remote_code=True)
    qrater_model = AutoModelForSequenceClassification.from_pretrained(
        QRATER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    def safe_texts(texts):
        return [t if t.strip() else '[empty]' for t in texts]

    print(f'Scoring with qrater...', flush=True)
    latte_qrater = classify_batch_qrater(safe_texts(latte_texts), qrater_model, qrater_tokenizer)
    dripper_qrater = classify_batch_qrater(safe_texts(dripper_texts), qrater_model, qrater_tokenizer)

    # Force empty to dirty
    for texts_list, results_list in [(latte_texts, latte_qrater), (dripper_texts, dripper_qrater)]:
        for i, t in enumerate(texts_list):
            if not t.strip():
                results_list[i] = 'dirty'

    # ── Report ──
    n = len(pages)
    print(f'\n{"="*70}')
    print(f'ORANGE LARGE vs DRIPPER — WMB English, {n} pages')
    print(f'{"="*70}')

    # ROUGE-5
    print(f'\n  ROUGE-5 F1:')
    print(f'  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}')
    print(f'  {"-"*67}')

    for name, texts in [('Pulpie Orange Large (2.1B)', latte_texts),
                        ('Dripper 0.6B (native vLLM)', dripper_texts)]:
        scores_by_level = {}
        scores_all = []
        for i, page in enumerate(pages):
            ref = page.get('convert_main_content', '')
            level = page.get('meta', {}).get('level', 'unknown')
            r5 = rouge_n_f1(ref, texts[i]) if ref and texts[i] else 0.0
            scores_all.append(r5)
            scores_by_level.setdefault(level, []).append(r5)
        avg = sum(scores_all) / max(len(scores_all), 1)
        avgs = {}
        for lev in ['simple', 'mid', 'hard']:
            vals = scores_by_level.get(lev, [])
            avgs[lev] = sum(vals) / max(len(vals), 1)
        print(f'  {name:<35} {avg:>8.4f} {avgs["simple"]:>8.4f} {avgs["mid"]:>8.4f} {avgs["hard"]:>8.4f}')

    # Qrater
    print(f'\n  Qrater Clean Rate:')
    print(f'  {"Method":<35} {"All":>8} {"Simple":>8} {"Mid":>8} {"Hard":>8}')
    print(f'  {"-"*67}')

    for name, results in [('Pulpie Orange Large (2.1B)', latte_qrater),
                          ('Dripper 0.6B (native vLLM)', dripper_qrater)]:
        all_clean = sum(1 for r in results if r == 'clean') / n * 100
        level_clean = {}
        for lev in ['simple', 'mid', 'hard']:
            idx = [i for i, p in enumerate(pages) if p.get('meta', {}).get('level') == lev]
            if idx:
                level_clean[lev] = sum(1 for i in idx if results[i] == 'clean') / len(idx) * 100
            else:
                level_clean[lev] = 0
        print(f'  {name:<35} {all_clean:>7.1f}% {level_clean["simple"]:>7.1f}% {level_clean["mid"]:>7.1f}% {level_clean["hard"]:>7.1f}%')


if __name__ == '__main__':
    main()
