"""Error analysis of pulpie pipeline on WebMainBench (English only).

Categorizes pages by failure mode:
  - Empty extraction (pulpie produces nothing)
  - Low precision (boilerplate leaking into output)
  - Low recall (missing content)
  - Formatting mismatch (content correct but formatting differs)
  - Good (ROUGE >= 0.8)

Also reports:
  - P/R distribution
  - Worst pages with examples
  - Error breakdown by difficulty level
  - Common patterns in failures
"""

import json
import os
import subprocess
import tempfile
from collections import Counter

import html2text

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")
HBIRD_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "pulpie")


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n_pr(reference, prediction, n=5):
    """Return (precision, recall, f1) for ROUGE-N."""
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    pred_ngrams = Counter(ngrams(pred_tokens, n))
    if not ref_ngrams or not pred_ngrams:
        return 0.0, 0.0, 0.0
    overlap = sum((ref_ngrams & pred_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def strip_formatting(text):
    """Remove markdown formatting to isolate content vs formatting errors."""
    import re
    t = text
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)  # headings
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)  # bold
    t = re.sub(r'\*(.+?)\*', r'\1', t)  # italic
    t = re.sub(r'_(.+?)_', r'\1', t)  # italic
    t = re.sub(r'`(.+?)`', r'\1', t)  # inline code
    t = re.sub(r'^\s*[\*\-\+]\s+', '', t, flags=re.MULTILINE)  # list markers
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)  # numbered lists
    t = re.sub(r'>\s*', '', t)  # blockquote
    t = re.sub(r'\|', ' ', t)  # table pipes
    t = re.sub(r'-{3,}', '', t)  # horizontal rules
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_h2t(html_content):
    """Extract via pulpie --html then html2text."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, "--html", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        raw_html = result.stdout.strip()
        if not raw_html:
            return ""
        h = html2text.HTML2Text(bodywidth=0)
        h.ignore_links = True
        h.ignore_images = True
        return h.handle(raw_html).strip()
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=500)
    args = parser.parse_args()

    print("Loading WebMainBench (English-only)...", flush=True)
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    records = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("meta", {}).get("language") == "en":
            records.append(rec)

    limit = min(args.pages, len(records))
    print(f"  Analyzing {limit} English pages\n", flush=True)

    # Collect per-page stats
    pages = []
    for i, rec in enumerate(records[:limit]):
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        level = rec.get("meta", {}).get("level", "unknown")
        url = rec.get("url", "?")
        style = rec.get("meta", {}).get("style", "?")

        if not html or not reference:
            continue

        pred = extract_h2t(html)

        if not pred:
            pages.append({
                "idx": i, "url": url, "level": level, "style": style,
                "p": 0, "r": 0, "f1": 0,
                "p_stripped": 0, "r_stripped": 0, "f1_stripped": 0,
                "category": "empty",
                "ref_len": len(reference), "pred_len": 0,
            })
        else:
            p, r, f1 = rouge_n_pr(reference, pred)

            # Also compute on stripped text to isolate formatting
            ref_stripped = strip_formatting(reference)
            pred_stripped = strip_formatting(pred)
            ps, rs, fs = rouge_n_pr(ref_stripped, pred_stripped)

            # Categorize
            if f1 >= 0.8:
                cat = "good"
            elif p < 0.4 and r >= 0.5:
                cat = "low_precision"
            elif r < 0.4 and p >= 0.5:
                cat = "low_recall"
            elif p < 0.4 and r < 0.4:
                cat = "both_low"
            elif fs - f1 > 0.1:
                cat = "formatting"
            elif r < p - 0.15:
                cat = "low_recall"
            elif p < r - 0.15:
                cat = "low_precision"
            else:
                cat = "moderate"

            pages.append({
                "idx": i, "url": url, "level": level, "style": style,
                "p": p, "r": r, "f1": f1,
                "p_stripped": ps, "r_stripped": rs, "f1_stripped": fs,
                "category": cat,
                "ref_len": len(reference), "pred_len": len(pred),
            })

        if (i + 1) % 100 == 0:
            avg_f1 = sum(pg["f1"] for pg in pages) / len(pages)
            print(f"  {i+1}/{limit}: avg_f1={avg_f1:.4f}", flush=True)

    # === REPORT ===
    n = len(pages)
    avg_f1 = sum(pg["f1"] for pg in pages) / n
    avg_p = sum(pg["p"] for pg in pages) / n
    avg_r = sum(pg["r"] for pg in pages) / n

    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS — {n} English pages")
    print(f"{'='*70}")

    # Overall stats
    print(f"\n  Overall: P={avg_p:.4f}  R={avg_r:.4f}  F1={avg_f1:.4f}")
    avg_fs = sum(pg["f1_stripped"] for pg in pages) / n
    print(f"  Stripped: F1={avg_fs:.4f}  (formatting gap: {avg_fs - avg_f1:+.4f})")

    # Category breakdown
    print(f"\n  {'Category':<18} {'Count':>6} {'%':>6} {'Avg F1':>8} {'Avg P':>8} {'Avg R':>8}")
    print(f"  {'-'*56}")
    cats = Counter(pg["category"] for pg in pages)
    for cat in ["good", "moderate", "formatting", "low_precision", "low_recall", "both_low", "empty"]:
        count = cats.get(cat, 0)
        if count == 0:
            continue
        subset = [pg for pg in pages if pg["category"] == cat]
        avg_f = sum(pg["f1"] for pg in subset) / len(subset)
        avg_pp = sum(pg["p"] for pg in subset) / len(subset)
        avg_rr = sum(pg["r"] for pg in subset) / len(subset)
        print(f"  {cat:<18} {count:>6} {count/n*100:>5.1f}% {avg_f:>8.4f} {avg_pp:>8.4f} {avg_rr:>8.4f}")

    # By difficulty level
    print(f"\n  {'Level':<10} {'Count':>6} {'Avg F1':>8} {'Avg P':>8} {'Avg R':>8} {'Empty':>6}")
    print(f"  {'-'*48}")
    for level in ["simple", "mid", "hard"]:
        subset = [pg for pg in pages if pg["level"] == level]
        if not subset:
            continue
        avg_f = sum(pg["f1"] for pg in subset) / len(subset)
        avg_pp = sum(pg["p"] for pg in subset) / len(subset)
        avg_rr = sum(pg["r"] for pg in subset) / len(subset)
        empty = sum(1 for pg in subset if pg["category"] == "empty")
        print(f"  {level:<10} {len(subset):>6} {avg_f:>8.4f} {avg_pp:>8.4f} {avg_rr:>8.4f} {empty:>6}")

    # By content style
    print(f"\n  {'Style':<25} {'Count':>6} {'Avg F1':>8}")
    print(f"  {'-'*41}")
    styles = Counter(pg["style"] for pg in pages)
    for style, count in styles.most_common(10):
        subset = [pg for pg in pages if pg["style"] == style]
        avg_f = sum(pg["f1"] for pg in subset) / len(subset)
        print(f"  {style:<25} {count:>6} {avg_f:>8.4f}")

    # F1 distribution
    print(f"\n  F1 distribution:")
    buckets = {"0.0": 0, "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0}
    for pg in pages:
        f = pg["f1"]
        if f == 0: buckets["0.0"] += 1
        elif f < 0.2: buckets["0.0-0.2"] += 1
        elif f < 0.4: buckets["0.2-0.4"] += 1
        elif f < 0.6: buckets["0.4-0.6"] += 1
        elif f < 0.8: buckets["0.6-0.8"] += 1
        elif f < 0.9: buckets["0.8-0.9"] += 1
        else: buckets["0.9-1.0"] += 1
    for bucket, count in buckets.items():
        bar = "#" * (count * 40 // n)
        print(f"    {bucket:>8}: {count:>4} ({count/n*100:>5.1f}%) {bar}")

    # Worst 20 pages
    print(f"\n  WORST 20 PAGES:")
    print(f"  {'Idx':>4} {'F1':>6} {'P':>6} {'R':>6} {'Cat':<15} {'Level':<7} {'Style':<20} URL")
    print(f"  {'-'*100}")
    worst = sorted(pages, key=lambda x: x["f1"])[:20]
    for pg in worst:
        print(f"  {pg['idx']:>4} {pg['f1']:>6.3f} {pg['p']:>6.3f} {pg['r']:>6.3f} {pg['category']:<15} {pg['level']:<7} {pg['style']:<20} {pg['url'][:50]}")

    # Biggest formatting gaps (content correct but formatting hurts)
    fmt_gap = sorted(pages, key=lambda x: -(x["f1_stripped"] - x["f1"]))[:10]
    print(f"\n  BIGGEST FORMATTING GAPS (stripped F1 - F1):")
    print(f"  {'Idx':>4} {'F1':>6} {'F1s':>6} {'Gap':>6} {'Level':<7} URL")
    print(f"  {'-'*70}")
    for pg in fmt_gap:
        gap = pg["f1_stripped"] - pg["f1"]
        if gap < 0.01:
            break
        print(f"  {pg['idx']:>4} {pg['f1']:>6.3f} {pg['f1_stripped']:>6.3f} {gap:>+6.3f} {pg['level']:<7} {pg['url'][:50]}")

    # Pages where precision is much lower than recall (boilerplate leak)
    print(f"\n  WORST PRECISION (boilerplate leaking, P << R):")
    p_gap = sorted(pages, key=lambda x: x["p"] - x["r"])[:10]
    for pg in p_gap:
        if pg["category"] == "empty":
            continue
        print(f"  idx={pg['idx']} P={pg['p']:.3f} R={pg['r']:.3f} F1={pg['f1']:.3f} ref={pg['ref_len']} pred={pg['pred_len']} {pg['url'][:60]}")

    # Pages where recall is much lower than precision (missing content)
    print(f"\n  WORST RECALL (missing content, R << P):")
    r_gap = sorted(pages, key=lambda x: x["r"] - x["p"])[:10]
    for pg in r_gap:
        if pg["category"] == "empty":
            continue
        print(f"  idx={pg['idx']} P={pg['p']:.3f} R={pg['r']:.3f} F1={pg['f1']:.3f} ref={pg['ref_len']} pred={pg['pred_len']} {pg['url'][:60]}")

    # Save detailed results
    out_path = os.path.join(DATA_DIR, "error_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(pages, f, indent=2)
    print(f"\n  Detailed results saved to {out_path}")


if __name__ == "__main__":
    main()
