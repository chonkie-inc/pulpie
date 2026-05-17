"""Diagnose why ROUGE-5 scores are low. Check:
1. Empty extractions (score 0)
2. Markdown formatting mismatch vs ground truth
3. Content missed vs boilerplate leaked
4. Score distribution — is it a few bad pages or uniformly bad?
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")
HBIRD_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "pulpie")


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0  # f1, precision, recall
    ref_ng = Counter(ngrams(ref_tokens, n))
    pred_ng = Counter(ngrams(pred_tokens, n))
    if not ref_ng or not pred_ng:
        return 0.0, 0.0, 0.0
    overlap = sum((ref_ng & pred_ng).values())
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def extract(html):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    try:
        r = subprocess.run([HBIRD_BIN, tmp], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except:
        return ""
    finally:
        os.unlink(tmp)


def main():
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    print(f"Analyzing {len(lines)} pages...\n", flush=True)

    all_f1 = []
    all_prec = []
    all_recall = []
    empty_pages = []
    worst_pages = []
    best_pages = []
    by_level = {}

    for i, line in enumerate(lines):
        rec = json.loads(line)
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        level = rec.get("meta", {}).get("level", "unknown")
        url = rec.get("url", "")

        if not html or not reference:
            continue

        pred = extract(html)

        if not pred:
            f1, prec, recall = 0.0, 0.0, 0.0
            empty_pages.append({"url": url, "level": level, "ref_len": len(reference)})
        else:
            f1, prec, recall = rouge_n_f1(reference, pred, n=5)

        all_f1.append(f1)
        all_prec.append(prec)
        all_recall.append(recall)

        if level not in by_level:
            by_level[level] = {"f1": [], "prec": [], "recall": []}
        by_level[level]["f1"].append(f1)
        by_level[level]["prec"].append(prec)
        by_level[level]["recall"].append(recall)

        entry = {"url": url[:60], "level": level, "f1": f1, "prec": prec,
                 "recall": recall, "pred_len": len(pred), "ref_len": len(reference)}

        if f1 < 0.3 and pred:  # bad but not empty
            worst_pages.append(entry)
        if f1 > 0.9:
            best_pages.append(entry)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(lines)}...", flush=True)

    n = len(all_f1)
    avg_f1 = sum(all_f1) / n
    avg_prec = sum(all_prec) / n
    avg_recall = sum(all_recall) / n

    print(f"\n{'='*70}")
    print(f"DIAGNOSIS ({n} pages)")
    print(f"{'='*70}")

    # 1. Overall precision vs recall
    print(f"\n1. PRECISION vs RECALL (is the problem missing content or extra boilerplate?)")
    print(f"   Average ROUGE-5 F1:        {avg_f1:.4f}")
    print(f"   Average ROUGE-5 Precision: {avg_prec:.4f}  (high = we don't include junk)")
    print(f"   Average ROUGE-5 Recall:    {avg_recall:.4f}  (high = we don't miss content)")
    if avg_prec > avg_recall:
        print(f"   --> RECALL is lower: we're MISSING CONTENT (too aggressive filtering)")
    else:
        print(f"   --> PRECISION is lower: we're LEAKING BOILERPLATE (too permissive)")

    # By level
    print(f"\n   By difficulty:")
    for level in ["simple", "mid", "hard"]:
        if level in by_level:
            d = by_level[level]
            lf1 = sum(d["f1"]) / len(d["f1"])
            lp = sum(d["prec"]) / len(d["prec"])
            lr = sum(d["recall"]) / len(d["recall"])
            print(f"   {level:>8}: F1={lf1:.4f}  P={lp:.4f}  R={lr:.4f}  (n={len(d['f1'])})")

    # 2. Empty extractions impact
    print(f"\n2. EMPTY EXTRACTIONS")
    print(f"   {len(empty_pages)} pages produced no output ({len(empty_pages)/n*100:.1f}%)")
    non_empty_f1 = [f for f in all_f1 if f > 0 or True]  # recalc without empties
    non_empty = [f for i, f in enumerate(all_f1) if all_prec[i] > 0 or all_recall[i] > 0 or f > 0]
    # Actually just filter properly
    ne_scores = [(f, p, r) for f, p, r in zip(all_f1, all_prec, all_recall) if not (f == 0 and p == 0 and r == 0)]
    if ne_scores:
        ne_f1 = sum(s[0] for s in ne_scores) / len(ne_scores)
        print(f"   Avg ROUGE-5 F1 excluding empty: {ne_f1:.4f} (vs {avg_f1:.4f} with empty)")
        print(f"   Impact of empties: {ne_f1 - avg_f1:+.4f}")

    if empty_pages:
        by_lvl = Counter(e["level"] for e in empty_pages)
        print(f"   Empty by level: {dict(by_lvl)}")
        avg_ref = sum(e["ref_len"] for e in empty_pages) / len(empty_pages)
        print(f"   Avg reference length of empty pages: {avg_ref:.0f} chars")

    # 3. Score distribution
    print(f"\n3. SCORE DISTRIBUTION")
    brackets = [(0, 0.01), (0.01, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    for lo, hi in brackets:
        count = sum(1 for f in all_f1 if lo <= f < hi)
        pct = count / n * 100
        bar = "#" * int(pct)
        label = f"[{lo:.2f}-{hi:.2f})"
        print(f"   {label:>14}: {count:>5} ({pct:>5.1f}%) {bar}")

    # 4. Worst non-empty pages
    print(f"\n4. WORST NON-EMPTY PAGES (F1 < 0.3, showing top 15)")
    worst_pages.sort(key=lambda x: x["f1"])
    for p in worst_pages[:15]:
        print(f"   F1={p['f1']:.3f} P={p['prec']:.3f} R={p['recall']:.3f} "
              f"pred={p['pred_len']:>6} ref={p['ref_len']:>6} [{p['level']}] {p['url']}")

    # 5. Length ratio analysis
    print(f"\n5. OUTPUT LENGTH vs REFERENCE LENGTH")
    ratios = []
    for i, line in enumerate(lines[:500]):  # sample for speed
        rec = json.loads(line)
        html = rec.get("html", "")
        ref = rec.get("convert_main_content", "")
        if not html or not ref:
            continue
        pred = extract(html)
        if pred:
            ratio = len(pred) / max(len(ref), 1)
            ratios.append(ratio)
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        median_ratio = sorted(ratios)[len(ratios)//2]
        print(f"   Avg pred/ref length ratio: {avg_ratio:.2f}")
        print(f"   Median pred/ref length ratio: {median_ratio:.2f}")
        too_short = sum(1 for r in ratios if r < 0.5) / len(ratios)
        too_long = sum(1 for r in ratios if r > 2.0) / len(ratios)
        print(f"   Too short (<0.5x): {too_short:.1%}")
        print(f"   Too long (>2.0x): {too_long:.1%}")


if __name__ == "__main__":
    main()
