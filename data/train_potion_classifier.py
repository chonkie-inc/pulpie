"""Train a potion-base-8M classifier for block-level keep/discard.

Uses model2vec's StaticModelForClassification (same approach as potion-edu-classifier).
Trains on WebMainBench English pages, holds out 200 pages for validation.
Reports block-level accuracy + page-level ROUGE-5.
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from bs4 import BeautifulSoup, Tag

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PULPIE_BIN = os.path.join(SCRIPT_DIR, "..", "target", "release", "export_features")
WEBMAINBENCH = os.path.join(SCRIPT_DIR, "webmainbench.jsonl")

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body"}
MIN_TEXT_LEN = 5
SEED = 42
VAL_PAGES = 200


# ── DOM helpers (from generate_training_data_dom.py) ──

def has_cc_select(element):
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def has_block_descendants(element):
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.name in BLOCK_TAGS:
            return True
    return False


def walk_dom(element, blocks):
    if not isinstance(element, Tag):
        return
    tag = element.name
    if tag in BLOCK_TAGS:
        text = element.get_text().strip()
        if len(text) >= MIN_TEXT_LEN:
            blocks.append(element)
        return
    if tag in CONTAINER_TAGS or tag not in BLOCK_TAGS:
        if has_block_descendants(element):
            for child in element.children:
                if isinstance(child, Tag):
                    walk_dom(child, blocks)
        else:
            text = element.get_text().strip()
            if len(text) >= MIN_TEXT_LEN:
                blocks.append(element)


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def strip_annotations(html):
    html = re.sub(r'\s+cc-select="[^"]*"', '', html)
    html = re.sub(r'\s+data-anno-uid="[^"]*"', '', html)
    html = re.sub(r'\s+class="mark-selected"', '', html)
    html = re.sub(r'</?marked-text[^>]*>', '', html)
    html = re.sub(r'</?marked-tail[^>]*>', '', html)
    return html


def run_export_features(html):
    result = subprocess.run(
        [PULPIE_BIN], input=html, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def extract_page_blocks(html):
    """Extract block texts + labels from an annotated WebMainBench page.

    Returns list of (text, label) tuples, or None on failure.
    """
    # DOM labeling
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    dom_blocks = []
    walk_dom(body, dom_blocks)

    dom_labels = []
    for block in dom_blocks:
        text = block.get_text()
        norm = normalize(text)
        label = 1 if has_cc_select(block) else 0
        dom_labels.append((norm, label))

    if not dom_labels:
        return None

    # Rust segmenter for matching
    clean_html = strip_annotations(html)
    rust_blocks = run_export_features(clean_html)
    if not rust_blocks:
        return None

    # Match
    label_map = {}
    for norm_text, label in dom_labels:
        if norm_text not in label_map:
            label_map[norm_text] = []
        label_map[norm_text].append(label)

    matched = []
    for rb in rust_blocks:
        norm = normalize(rb["text"])
        if norm in label_map and label_map[norm]:
            label = label_map[norm].pop(0)
            matched.append((rb["text"], label))

    return matched if matched else None


def calc_rouge5_f1(reference, prediction):
    """Whitespace-tokenized ROUGE-5 F1."""
    ref_tokens = reference.strip().split()
    pred_tokens = prediction.strip().split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    n = 5
    def ngrams(tokens, n):
        counts = {}
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    ref_ng = ngrams(ref_tokens, n)
    pred_ng = ngrams(pred_tokens, n)
    if not ref_ng or not pred_ng:
        return 0.0

    overlap = 0
    for ng, count in pred_ng.items():
        overlap += min(count, ref_ng.get(ng, 0))

    prec = overlap / sum(pred_ng.values()) if pred_ng else 0
    rec = overlap / sum(ref_ng.values()) if ref_ng else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='minishlab/potion-base-8M')
    parser.add_argument('--val-pages', type=int, default=VAL_PAGES)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    # ── Step 1: Extract all English pages with block texts + labels ──
    print("Loading WebMainBench (English only)...", flush=True)
    pages = []  # list of {html, reference, blocks: [(text, label)], meta}
    skipped = 0

    with open(WEBMAINBENCH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') != 'en':
                continue
            html = rec.get('html', '')
            reference = rec.get('convert_main_content', '')
            if not html or len(html) < 100:
                skipped += 1
                continue

            blocks = extract_page_blocks(html)
            if not blocks:
                skipped += 1
                continue

            pages.append({
                'reference': reference,
                'blocks': blocks,
                'level': rec.get('meta', {}).get('level', '?'),
            })

            if len(pages) % 500 == 0:
                print(f"  {len(pages)} pages extracted ({skipped} skipped)", flush=True)

    print(f"  Total: {len(pages)} pages, {skipped} skipped", flush=True)
    total_blocks = sum(len(p['blocks']) for p in pages)
    total_keep = sum(sum(1 for _, l in p['blocks'] if l == 1) for p in pages)
    print(f"  Blocks: {total_blocks} ({total_keep} keep, {total_blocks - total_keep} discard)", flush=True)

    # ── Step 2: Page-level train/val split ──
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(pages))
    val_idx = set(indices[:args.val_pages])

    train_texts, train_labels = [], []
    val_pages_data = []

    for i, page in enumerate(pages):
        if i in val_idx:
            val_pages_data.append(page)
        else:
            for text, label in page['blocks']:
                train_texts.append(text)
                train_labels.append(label)

    val_texts, val_labels = [], []
    for page in val_pages_data:
        for text, label in page['blocks']:
            val_texts.append(text)
            val_labels.append(label)

    print(f"\n  Train: {len(pages) - args.val_pages} pages, {len(train_texts)} blocks", flush=True)
    print(f"  Val:   {args.val_pages} pages, {len(val_texts)} blocks", flush=True)

    # ── Step 3: Train StaticModelForClassification ──
    from model2vec.train import StaticModelForClassification

    print(f"\nTraining potion classifier ({args.model})...", flush=True)
    t0 = time.time()

    classifier = StaticModelForClassification.from_pretrained(model_name=args.model)
    classifier.fit(train_texts, train_labels)

    train_time = time.time() - t0
    print(f"  Training took {train_time:.1f}s", flush=True)

    # Save model
    import torch
    save_path = os.path.join(SCRIPT_DIR, "potion_block_classifier.pt")
    torch.save(classifier.state_dict(), save_path)
    print(f"  Saved to {save_path}", flush=True)

    # ── Step 4: Evaluate — block-level metrics ──
    print(f"\nEvaluating...", flush=True)
    val_preds = classifier.predict(val_texts)
    val_preds = np.array(val_preds)
    val_labels_arr = np.array(val_labels)

    acc = (val_preds == val_labels_arr).mean()
    tp = ((val_preds == 1) & (val_labels_arr == 1)).sum()
    fp = ((val_preds == 1) & (val_labels_arr == 0)).sum()
    fn = ((val_preds == 0) & (val_labels_arr == 1)).sum()
    tn = ((val_preds == 0) & (val_labels_arr == 0)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{'='*60}")
    print(f"BLOCK-LEVEL METRICS ({len(val_texts)} blocks)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # ── Step 5: Evaluate — page-level ROUGE-5 ──
    print(f"\n{'='*60}")
    print(f"PAGE-LEVEL ROUGE-5 ({len(val_pages_data)} pages)")
    print(f"{'='*60}")

    block_offset = 0
    rouge_scores = []
    level_scores = {}

    for page in val_pages_data:
        n_blocks = len(page['blocks'])
        page_preds = val_preds[block_offset:block_offset + n_blocks]
        page_texts = [t for t, _ in page['blocks']]

        # Reconstruct: concatenate text of blocks predicted as keep
        kept = [page_texts[j] for j in range(n_blocks) if page_preds[j] == 1]
        prediction = '\n'.join(kept)
        reference = page['reference']

        r5 = calc_rouge5_f1(reference, prediction)
        rouge_scores.append(r5)

        level = page['level']
        if level not in level_scores:
            level_scores[level] = []
        level_scores[level].append(r5)

        block_offset += n_blocks

    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0

    print(f"  Overall ROUGE-5 F1: {avg_rouge:.4f}")
    for level in ['simple', 'mid', 'hard']:
        if level in level_scores:
            scores = level_scores[level]
            print(f"  {level:>7}: {np.mean(scores):.4f} ({len(scores)} pages)")

    # Distribution
    bins = [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)]
    print(f"\n  F1 distribution:")
    for lo, hi in bins:
        count = sum(1 for s in rouge_scores if lo <= s < hi)
        label = f"{lo:.1f}-{hi:.2f}" if hi > 1 else f"{lo:.1f}-{hi:.1f}"
        print(f"    {label}: {count:>4} ({count/len(rouge_scores)*100:.1f}%)")
    empty = sum(1 for s in rouge_scores if s == 0)
    print(f"    empty:  {empty:>4} ({empty/len(rouge_scores)*100:.1f}%)")

    # ── Comparison ──
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'ROUGE-5 F1':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Potion classifier':<30} {avg_rouge:>10.4f}")
    print(f"  {'Pulpie GBM (h2t)':<30} {'0.8059':>10}")
    print(f"  {'Dripper 0.6B (paper)':<30} {'0.8780':>10}")


if __name__ == '__main__':
    main()
