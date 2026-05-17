"""Evaluate pulpie on WebMainBench using ROUGE-5 F1 (same as Dripper paper).

Fast implementation: custom n-gram ROUGE to avoid slow rouge_scorer library.
"""

import json
import os
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
    """Compute ROUGE-N F1 between two strings."""
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


def extract_with_pulpie(html_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def main():
    print("Loading WebMainBench...", flush=True)
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    total = len(lines)
    print(f"  {total} pages total", flush=True)

    scores_all = []
    scores_by_level = {}
    empty_count = 0

    for i, line in enumerate(lines):
        rec = json.loads(line)
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        level = rec.get("meta", {}).get("level", "unknown")

        if not html or not reference:
            continue

        prediction = extract_with_pulpie(html)

        if not prediction:
            empty_count += 1
            r5 = 0.0
        else:
            r5 = rouge_n_f1(reference, prediction, n=5)

        scores_all.append(r5)
        if level not in scores_by_level:
            scores_by_level[level] = []
        scores_by_level[level].append(r5)

        if (i + 1) % 500 == 0:
            avg = sum(scores_all) / len(scores_all)
            print(f"  {i+1}/{total} pages, avg ROUGE-5 F1: {avg:.4f}, empty: {empty_count}", flush=True)

    # Results
    n = len(scores_all)
    avg_r5 = sum(scores_all) / max(n, 1)

    print(f"\n{'='*70}", flush=True)
    print(f"PULPIE — WebMainBench ROUGE-5 F1 ({n} pages)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Empty extractions: {empty_count} ({empty_count/max(n,1)*100:.1f}%)", flush=True)

    print(f"\n  {'':>25} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*57}", flush=True)

    by_level_avgs = {}
    for level in ["simple", "mid", "hard"]:
        if level in scores_by_level:
            vals = scores_by_level[level]
            by_level_avgs[level] = sum(vals) / max(len(vals), 1)
        else:
            by_level_avgs[level] = 0.0

    print(f"  {'ROUGE-5 F1':>25} {avg_r5:>8.4f} {by_level_avgs['simple']:>8.4f} {by_level_avgs['mid']:>8.4f} {by_level_avgs['hard']:>8.4f}", flush=True)

    # Comparison
    print(f"\n  {'='*57}", flush=True)
    print(f"  COMPARISON (from Dripper paper)", flush=True)
    print(f"  {'='*57}", flush=True)
    print(f"  {'Tool':<25} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*57}", flush=True)

    comparisons = [
        ("DeepSeek-V3.2 (LLM)", 0.9098, 0.9415, 0.9104, 0.8771),
        ("GPT-4 (LLM)", 0.9024, 0.9382, 0.9042, 0.8638),
        ("Dripper 0.6B", 0.8779, 0.9205, 0.8804, 0.8313),
        ("magic-html", 0.7138, 0.7857, 0.7121, 0.6434),
        ("Readability", 0.6543, 0.7415, 0.6550, 0.5652),
        ("Trafilatura", 0.6402, 0.7309, 0.6417, 0.5466),
    ]

    hbird = ("** PULPIE **", avg_r5, by_level_avgs["simple"], by_level_avgs["mid"], by_level_avgs["hard"])
    all_entries = comparisons + [hbird]
    all_entries.sort(key=lambda x: -x[1])

    for name, r_all, r_s, r_m, r_h in all_entries:
        marker = " <--" if "PULPIE" in name else ""
        print(f"  {name:<25} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}{marker}", flush=True)

    # Save
    out_path = os.path.join(DATA_DIR, "bench_rouge_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "total_pages": n,
            "empty_count": empty_count,
            "rouge5_all": avg_r5,
            "rouge5_simple": by_level_avgs["simple"],
            "rouge5_mid": by_level_avgs["mid"],
            "rouge5_hard": by_level_avgs["hard"],
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
