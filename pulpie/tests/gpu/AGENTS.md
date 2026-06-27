# GPU acceptance runbook — simplify-port ROUGE-5 gate

You are an agent on a GPU box. Your job: confirm that pulpie's **own** pipeline —
after the MinerU-HTML simplify/reconstruct port — reaches the quality the Orange
models were trained for. No prior session context is needed; everything is here.

## Background (why this exists)

pulpie's `simplify.py` was rewritten to faithfully reproduce MinerU-HTML's
`simplify_html` output (the format the Orange models were distilled on). Offline
byte-parity tests already pass (`pytest pulpie/tests -n auto`, CPU-only). This
runbook is the **end-to-end quality gate**: run pulpie standalone and verify the
score moved from the old **0.731 ROUGE-5** to **~0.862** (parity with MinerU's own
pipeline). Hitting ~0.862 means the port closed the gap and pulpie no longer needs
MinerU as an external dependency.

## Environment

- A CUDA GPU (the Orange models are small; a single 16GB+ GPU is ample).
- This repo checked out on branch `worktree-simplify-mineru-port` (or wherever it
  was merged).

```bash
cd <repo-root>
python -m venv .venv && source .venv/bin/activate
uv pip install -e "pulpie[markdown]"   # pulls torch, transformers, lxml,
                                       # selectolax, beautifulsoup4, html2text
uv pip install huggingface_hub          # for the dataset download
```

## Step 1 — Get the eval data

Download the 545-sample calibrated subset (has `convert_main_content` ground
truth; ~109 MB — do NOT pull the 1.4 GB full `webmainbench.jsonl`):

```bash
hf download opendatalab/WebMainBench WebMainBench_545.jsonl \
    --repo-type dataset --local-dir ./data
```

Each JSONL record has (verify field names on first run — print one record):
- `html` — raw input HTML.
- `convert_main_content` — ground-truth main content (markdown-ish text), the
  ROUGE-5 reference.
- `meta.level` — difficulty bucket (`simple` / `middle` / `hard`), optional.

## Step 2 — Run pulpie end-to-end and score

The script below uses pulpie's **public** `Pipeline` (which now calls the ported
`simplify`/`reconstruct`), then scores ROUGE-5 with the same whitespace-tokenized
F1 used in `eval/eval_latte_large_vs_dripper.py` (`rouge_n_f1`, n=5).

```python
# eval_simplify_port_gate.py
import json
from collections import Counter

from pulpie import Pipeline, PageInput


def ngrams(toks, n):
    return [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    ref, pred = reference.split(), prediction.split()
    if not ref or not pred:
        return 0.0
    r, p = Counter(ngrams(ref, n)), Counter(ngrams(pred, n))
    if not r or not p:
        return 0.0
    overlap = sum((r & p).values())
    precision = overlap / max(sum(p.values()), 1)
    recall = overlap / max(sum(r.values()), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main():
    pages, refs = [], []
    with open("data/WebMainBench_545.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            html = rec.get("html", "")
            ref = rec.get("convert_main_content", "")
            if not html or not ref:
                continue
            pages.append(PageInput(html=html, page_id=i))
            refs.append(ref)

    pipeline = Pipeline(model="orange-small")     # or orange-base / orange-large
    results = pipeline.extract_batch(pages)

    scores = [rouge_n_f1(ref, r.markdown) for ref, r in zip(refs, results)]
    avg = sum(scores) / max(len(scores), 1)
    print(f"pages={len(scores)}  mean ROUGE-5 F1 = {avg:.4f}")


if __name__ == "__main__":
    main()
```

```bash
python eval_simplify_port_gate.py
```

## Step 3 — The gate

- **PASS:** mean ROUGE-5 ≈ **0.86** (small model may sit slightly lower than
  large; `orange-large` is the 0.864 reference). Anything **≥ ~0.85** confirms the
  port worked — a large jump from the old 0.731.
- **FAIL / regressed (~0.73):** the simplify port is not being exercised. Sanity
  checks:
  - `python -c "from pulpie.simplify import simplify; print(simplify('<html><body><nav>x</nav><p>hi</p></body>')[0])"`
    — output should drop `<nav>` and show `_item_id` on the `<p>`.
  - Confirm `pytest pulpie/tests -n auto` is green (byte-parity holds).
  - Confirm the installed `pulpie` is this branch (not a stale PyPI build).

Report back: model used, page count, mean ROUGE-5, and (if available) the
per-`level` breakdown. Compare against the old 0.731 baseline and the 0.862 target.

## Notes
- Reference scoring matches the repo's existing harness; if `convert_main_content`
  is absent under that exact key in your copy, print one record and adjust the key.
- This is quality validation only — it does not need to pass in CI (no GPU there).
  The CPU byte-parity tests are the CI gate.
