"""Score pulpie outputs and API reference texts with qrater."""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/home/bhavnick/workspace/gym/qrater/models/qwen-0.6b-distill/seed42_lr5e-05_ep3_T1.0_a0.5/final"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model

def score_text(text, tokenizer, model):
    if not text.strip():
        return "dirty", 0.0, 1.0
    inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = logits.argmax(dim=-1).item()
        label = "clean" if pred == 1 else "dirty"
        confidence = probs[pred].item()
        clean_prob = probs[1].item()
    return label, clean_prob, confidence

def main():
    print("Loading qrater model (0.6B)...")
    tokenizer, model = load_model()
    print("Model loaded.\n")

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(eval_dir, "output")
    ref_dir = os.path.join(eval_dir, "reference")

    # Collect all page IDs
    pages = sorted(
        f.replace(".md", "")
        for f in os.listdir(output_dir)
        if f.endswith(".md")
    )

    print(f"{'Page':<40} {'Hbird Label':>12} {'Hbird P(clean)':>15} {'Ref Label':>10} {'Ref P(clean)':>14}")
    print("-" * 95)

    results = []
    for page in pages:
        # Score hummingbird output
        hbird_path = os.path.join(output_dir, f"{page}.md")
        with open(hbird_path) as f:
            hbird_text = f.read()
        h_label, h_clean, h_conf = score_text(hbird_text, tokenizer, model)

        # Score reference if exists
        ref_path = os.path.join(ref_dir, f"{page}.txt")
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                ref_text = f.read()
            r_label, r_clean, r_conf = score_text(ref_text, tokenizer, model)
        else:
            r_label, r_clean, r_conf = "-", "-", "-"

        r_clean_str = f"{r_clean:.3f}" if isinstance(r_clean, float) else r_clean
        print(f"{page:<40} {h_label:>12} {h_clean:>15.3f} {r_label:>10} {r_clean_str:>14}")

        results.append({
            "page": page,
            "hbird_label": h_label, "hbird_clean_prob": round(h_clean, 4),
            "ref_label": r_label if isinstance(r_label, str) and r_label != "-" else None,
            "ref_clean_prob": round(r_clean, 4) if isinstance(r_clean, float) else None,
        })

    # Summary
    h_clean_count = sum(1 for r in results if r["hbird_label"] == "clean")
    h_total = len(results)
    ref_results = [r for r in results if r["ref_label"] is not None]
    r_clean_count = sum(1 for r in ref_results if r["ref_label"] == "clean")
    r_total = len(ref_results)

    print(f"\n{'Summary':=^95}")
    print(f"Hummingbird: {h_clean_count}/{h_total} clean ({h_clean_count/h_total*100:.0f}%)")
    if r_total:
        print(f"Reference:   {r_clean_count}/{r_total} clean ({r_clean_count/r_total*100:.0f}%)")

    # Save results
    results_path = os.path.join(eval_dir, "qrater_scores.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
