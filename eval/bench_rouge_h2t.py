"""Evaluate pulpie on WebMainBench (English-only) using html2text canonicalization.

Matches the official WebMainBench eval pipeline:
  - Extract content HTML via pulpie --html
  - Convert through html2text (bodywidth=0, ignore_links, ignore_images)
  - Compare against convert_main_content (also html2text output)
  - ROUGE-5 F1 with whitespace tokenization (English-only, so jieba not needed)
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


def html_to_text(html_str):
    """Convert HTML to text using html2text with official WebMainBench settings.
    Fresh instance each call to avoid statefulness bugs."""
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


def extract_html(html_content):
    """Extract content HTML via pulpie --html."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, "--html", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def extract_md(html_content):
    """Extract markdown via pulpie (normal mode)."""
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
    print("Loading WebMainBench (English-only)...", flush=True)
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    # Filter to English
    records = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("meta", {}).get("language") == "en":
            records.append(rec)
    print(f"  {len(records)} English pages (of {len(lines)} total)", flush=True)

    scores_h2t = []       # pulpie --html → html2text
    scores_native = []    # pulpie native markdown
    scores_by_level_h2t = {}
    scores_by_level_native = {}
    empty_h2t = 0
    empty_native = 0

    for i, rec in enumerate(records):
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        level = rec.get("meta", {}).get("level", "unknown")

        if not html or not reference:
            continue

        # Path 1: --html mode → html2text
        extracted_html = extract_html(html)
        if extracted_html:
            pred_h2t = html_to_text(extracted_html).strip()
        else:
            pred_h2t = ""

        if not pred_h2t:
            empty_h2t += 1
            r5_h2t = 0.0
        else:
            r5_h2t = rouge_n_f1(reference, pred_h2t, n=5)

        scores_h2t.append(r5_h2t)
        scores_by_level_h2t.setdefault(level, []).append(r5_h2t)

        # Path 2: native markdown
        pred_native = extract_md(html)
        if not pred_native:
            empty_native += 1
            r5_native = 0.0
        else:
            r5_native = rouge_n_f1(reference, pred_native, n=5)

        scores_native.append(r5_native)
        scores_by_level_native.setdefault(level, []).append(r5_native)

        if (i + 1) % 500 == 0:
            n = len(scores_h2t)
            avg_h = sum(scores_h2t) / n
            avg_n = sum(scores_native) / n
            print(f"  {i+1}/{len(records)}: h2t={avg_h:.4f}  native={avg_n:.4f}  empty_h2t={empty_h2t}  empty_native={empty_native}", flush=True)

    n = len(scores_h2t)
    avg_h2t = sum(scores_h2t) / max(n, 1)
    avg_native = sum(scores_native) / max(n, 1)

    print(f"\n{'='*70}", flush=True)
    print(f"PULPIE — WebMainBench ROUGE-5 F1 (English only, {n} pages)", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n  {'':>30} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*62}", flush=True)

    for label, scores_by_level, empty_count in [
        ("html2text canon.", scores_by_level_h2t, empty_h2t),
        ("native markdown", scores_by_level_native, empty_native),
    ]:
        avgs = {}
        for lev in ["simple", "mid", "hard"]:
            vals = scores_by_level.get(lev, [])
            avgs[lev] = sum(vals) / max(len(vals), 1)
        total = sum(sum(v) for v in scores_by_level.values()) / max(sum(len(v) for v in scores_by_level.values()), 1)
        print(f"  {label:>30} {total:>8.4f} {avgs['simple']:>8.4f} {avgs['mid']:>8.4f} {avgs['hard']:>8.4f}  (empty={empty_count})", flush=True)

    print(f"\n  Difference (h2t - native):   {avg_h2t - avg_native:+.4f}", flush=True)

    # Comparison
    print(f"\n  {'='*62}", flush=True)
    print(f"  COMPARISON (Dripper paper, full dataset incl. non-English)", flush=True)
    print(f"  {'='*62}", flush=True)
    print(f"  {'Tool':<30} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*62}", flush=True)

    comparisons = [
        ("DeepSeek-V3.2 (LLM)", 0.9098, 0.9415, 0.9104, 0.8771),
        ("GPT-4 (LLM)", 0.9024, 0.9382, 0.9042, 0.8638),
        ("Dripper 0.6B", 0.8779, 0.9205, 0.8804, 0.8313),
        ("magic-html", 0.7138, 0.7857, 0.7121, 0.6434),
        ("Readability", 0.6543, 0.7415, 0.6550, 0.5652),
        ("Trafilatura", 0.6402, 0.7309, 0.6417, 0.5466),
    ]

    h2t_avgs = {}
    for lev in ["simple", "mid", "hard"]:
        vals = scores_by_level_h2t.get(lev, [])
        h2t_avgs[lev] = sum(vals) / max(len(vals), 1)

    hbird = ("** PULPIE (h2t) **", avg_h2t, h2t_avgs["simple"], h2t_avgs["mid"], h2t_avgs["hard"])
    all_entries = comparisons + [hbird]
    all_entries.sort(key=lambda x: -x[1])

    for name, r_all, r_s, r_m, r_h in all_entries:
        marker = " <--" if "PULPIE" in name else ""
        print(f"  {name:<30} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}{marker}", flush=True)

    print(f"\n  NOTE: Dripper paper numbers include non-English pages.", flush=True)
    print(f"  Our English-only scores are not directly comparable.", flush=True)

    out_path = os.path.join(DATA_DIR, "bench_rouge_h2t_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "english_only": True,
            "total_pages": n,
            "h2t_rouge5_all": avg_h2t,
            "native_rouge5_all": avg_native,
            "h2t_empty": empty_h2t,
            "native_empty": empty_native,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
