"""Sample 50 pages from WebMainBench, extract with pulpie, score with qrater."""

import json
import os
import random
import subprocess
import tempfile

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")
HBIRD_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "pulpie")
QRATER_PATH = "/home/bhavnick/workspace/gym/qrater/models/qwen-0.6b-distill/seed42_lr5e-05_ep3_T1.0_a0.5/final"

N_SAMPLES = 50
SEED = 42


def load_qrater():
    tokenizer = AutoTokenizer.from_pretrained(QRATER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        QRATER_PATH, torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def score_text(text, tokenizer, model):
    if not text.strip():
        return "dirty", 0.0
    inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = logits.argmax(dim=-1).item()
        label = "clean" if pred == 1 else "dirty"
        clean_prob = probs[1].item()
    return label, clean_prob


def extract_with_pulpie(html_content):
    """Write HTML to temp file, run pulpie, return markdown."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        return ""
    finally:
        os.unlink(tmp_path)


def main():
    # Load sample
    print(f"Loading {N_SAMPLES} random pages from WebMainBench...")
    random.seed(SEED)
    with open(BENCH_PATH) as f:
        lines = f.readlines()
    indices = random.sample(range(len(lines)), N_SAMPLES)
    pages = [json.loads(lines[i]) for i in indices]

    # Load qrater
    print("Loading qrater model...")
    tokenizer, model = load_qrater()
    print("Model loaded.\n")

    print(f"{'#':<4} {'URL':<60} {'Label':>6} {'P(clean)':>9} {'Chars':>7}")
    print("-" * 90)

    results = []
    clean_count = 0
    for i, page in enumerate(pages):
        url = page["url"]
        html = page["html"]
        md = extract_with_pulpie(html)
        md_chars = len(md.strip())

        if md_chars < 10:
            label, clean_prob = "empty", 0.0
        else:
            label, clean_prob = score_text(md, tokenizer, model)

        if label == "clean":
            clean_count += 1

        url_short = url[:58] if len(url) > 58 else url
        print(f"{i+1:<4} {url_short:<60} {label:>6} {clean_prob:>9.3f} {md_chars:>7}")

        results.append({
            "track_id": page.get("track_id", ""),
            "url": url,
            "level": page.get("meta", {}).get("level", ""),
            "label": label,
            "clean_prob": round(clean_prob, 4),
            "md_chars": md_chars,
        })

    print(f"\n{'='*90}")
    non_empty = [r for r in results if r["label"] != "empty"]
    clean_of_non_empty = sum(1 for r in non_empty if r["label"] == "clean")
    empty_count = sum(1 for r in results if r["label"] == "empty")
    print(f"Total: {len(results)} pages")
    print(f"Empty extractions: {empty_count}")
    print(f"Non-empty: {len(non_empty)}, Clean: {clean_of_non_empty}/{len(non_empty)} ({clean_of_non_empty/max(len(non_empty),1)*100:.0f}%)")
    avg_clean = sum(r["clean_prob"] for r in non_empty) / max(len(non_empty), 1)
    print(f"Avg P(clean) on non-empty: {avg_clean:.3f}")

    # By difficulty level
    for level in ["simple", "mid", "hard"]:
        subset = [r for r in non_empty if r["level"] == level]
        if subset:
            c = sum(1 for r in subset if r["label"] == "clean")
            avg = sum(r["clean_prob"] for r in subset) / len(subset)
            print(f"  {level}: {c}/{len(subset)} clean ({c/len(subset)*100:.0f}%), avg P(clean)={avg:.3f}")

    # Save
    out_path = os.path.join(DATA_DIR, "eval_sample50_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
