"""Evaluate pulpie output quality using qrater clean/dirty classifier.

Runs pulpie on HTML pages (WebMainBench or CC), then scores
the extracted markdown with the qrater EuroBERT-210m model.
Reports clean rate before (raw html2text) and after pulpie extraction.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import html2text
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, '..', 'data')
HBIRD_BIN = os.path.join(EVAL_DIR, '..', 'target', 'release', 'pulpie')
QRATER_MODEL = os.path.join(EVAL_DIR, '..', '..', 'gym', 'qrater',
                             'models', 'encoder-distill',
                             'eurobert-210m_0.6b-labels', 'final')

WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')
CC_PATH = os.path.join(DATA_DIR, 'cc_sampled.jsonl')

MAX_LENGTH = 4096
MAX_TEXT_CHARS = 10000


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
            [HBIRD_BIN, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ''
    finally:
        os.unlink(tmp_path)


def classify_batch(texts, model, tokenizer, batch_size=32):
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH,
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
    parser.add_argument('--source', choices=['wmb', 'cc'], default='wmb')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    # Load qrater model
    device = f'cuda:{args.gpu}'
    print(f'Loading qrater model on {device}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(QRATER_MODEL, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        QRATER_MODEL, dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load pages
    if args.source == 'wmb':
        print(f'Loading WebMainBench (English only)...', flush=True)
        pages = []
        with open(WMB_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('meta', {}).get('language') == 'en':
                    pages.append(rec)
        print(f'  {len(pages)} English pages', flush=True)
    else:
        print(f'Loading CC sampled...', flush=True)
        pages = []
        with open(CC_PATH) as f:
            for line in f:
                pages.append(json.loads(line))
        print(f'  {len(pages)} pages', flush=True)

    if args.limit > 0:
        pages = pages[:args.limit]
        print(f'  Limited to {len(pages)}', flush=True)

    # Process: raw html2text vs pulpie extraction
    print(f'\nExtracting with pulpie...', flush=True)
    raw_texts = []
    hbird_texts = []
    t0 = time.time()

    for i, page in enumerate(pages):
        html = page.get('html', '')

        # Raw: just html2text the whole page
        raw_md = html_to_text(html)[:MAX_TEXT_CHARS]
        raw_texts.append(raw_md if raw_md.strip() else '')

        # Pulpie: extract then score
        hbird_md = extract_with_pulpie(html)[:MAX_TEXT_CHARS]
        hbird_texts.append(hbird_md if hbird_md.strip() else '')

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(pages)} ({elapsed:.0f}s)', flush=True)

    print(f'  Done extracting in {time.time() - t0:.0f}s', flush=True)
    print(f'  Pulpie empty: {sum(1 for t in hbird_texts if not t)}', flush=True)
    print(f'  Raw empty: {sum(1 for t in raw_texts if not t)}', flush=True)

    # Score with qrater
    print(f'\nScoring with qrater...', flush=True)

    # Replace empty strings with a placeholder so tokenizer doesn't crash
    raw_for_score = [t if t.strip() else '[empty]' for t in raw_texts]
    hbird_for_score = [t if t.strip() else '[empty]' for t in hbird_texts]

    raw_results = classify_batch(raw_for_score, model, tokenizer)
    hbird_results = classify_batch(hbird_for_score, model, tokenizer)

    # Force empty extractions to dirty
    for i, t in enumerate(raw_texts):
        if not t.strip():
            raw_results[i] = {'label': 'dirty', 'confidence': 1.0}
    for i, t in enumerate(hbird_texts):
        if not t.strip():
            hbird_results[i] = {'label': 'dirty', 'confidence': 1.0}

    # Report
    raw_clean = sum(1 for r in raw_results if r['label'] == 'clean')
    hbird_clean = sum(1 for r in hbird_results if r['label'] == 'clean')
    n = len(pages)

    print(f'\n{"="*60}')
    print(f'QRATER CLEAN RATE ({args.source.upper()}, {n} pages)')
    print(f'{"="*60}')
    print(f'  {"Method":<30} {"Clean":>6} {"Dirty":>6} {"Clean%":>8}')
    print(f'  {"-"*52}')
    print(f'  {"Raw html2text":<30} {raw_clean:>6} {n - raw_clean:>6} {raw_clean/n*100:>7.1f}%')
    print(f'  {"Pulpie (combined)":<30} {hbird_clean:>6} {n - hbird_clean:>6} {hbird_clean/n*100:>7.1f}%')
    print(f'  {"Improvement":<30} {"":>6} {"":>6} {(hbird_clean - raw_clean)/n*100:>+7.1f}pp')

    # Transition matrix
    both_clean = sum(1 for r, h in zip(raw_results, hbird_results) if r['label'] == 'clean' and h['label'] == 'clean')
    raw_clean_hbird_dirty = sum(1 for r, h in zip(raw_results, hbird_results) if r['label'] == 'clean' and h['label'] == 'dirty')
    raw_dirty_hbird_clean = sum(1 for r, h in zip(raw_results, hbird_results) if r['label'] == 'dirty' and h['label'] == 'clean')
    both_dirty = sum(1 for r, h in zip(raw_results, hbird_results) if r['label'] == 'dirty' and h['label'] == 'dirty')

    print(f'\n  Transition matrix:')
    print(f'  {"":>25} {"Pulpie Clean":>12} {"Pulpie Dirty":>12}')
    print(f'  {"Raw Clean":<25} {both_clean:>12} {raw_clean_hbird_dirty:>12}')
    print(f'  {"Raw Dirty":<25} {raw_dirty_hbird_clean:>12} {both_dirty:>12}')

    # By difficulty level (WMB only)
    if args.source == 'wmb':
        print(f'\n  By difficulty:')
        for level in ['simple', 'mid', 'hard']:
            idx = [i for i, p in enumerate(pages) if p.get('meta', {}).get('level') == level]
            if not idx:
                continue
            lev_raw = sum(1 for i in idx if raw_results[i]['label'] == 'clean')
            lev_hbird = sum(1 for i in idx if hbird_results[i]['label'] == 'clean')
            print(f'    {level:>7}: raw={lev_raw}/{len(idx)} ({lev_raw/len(idx)*100:.1f}%)  '
                  f'pulpie={lev_hbird}/{len(idx)} ({lev_hbird/len(idx)*100:.1f}%)  '
                  f'delta={((lev_hbird-lev_raw)/len(idx)*100):+.1f}pp')


if __name__ == '__main__':
    main()
