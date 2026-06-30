# Pulpie Evaluation Set

HTML pages sourced from qrater datasets (exa, brave_jina, firecrawl APIs) for comparing pulpie's extraction against API baselines.

## Pages

| ID | Domain | Source | Status | Notes |
|----|--------|--------|--------|-------|
| 01_article_pg | paulgraham.com | manual | OK | Long-form essay |
| 02_nodejs_dev | nodejs.dev | exa | OK | Documentation |
| 03_numberanalytics_com | numberanalytics.com | exa | OK | Blog/article |
| 04_evisional_com | evisional.com | exa | OK | Technical blog |
| 05_protobuf_dev | protobuf.dev | exa | OK | Documentation/tutorial |
| 06_cobra_dev | cobra.dev | exa | OK | Project landing page |
| 07_how2_sh | how2.sh | exa | FAIL | JS-rendered SPA |
| 08_u11d_com | u11d.com | exa | OK | E-commerce tutorial |
| 09_istio_io | istio.io | exa | OK | Documentation |
| 10_immunologyexplained | aai.org | exa | FAIL | JS-rendered SPA |
| 11_tutorialspoint_com | tutorialspoint.com | brave_jina | OK | Tutorial with code |
| 12_builtin_com | builtin.com | brave_jina | OK | Technical article |
| 13_omnipod_com | omnipod.com | brave_jina | OK | Product/health info |

## Directories

- `html/` — Raw HTML pages (curl'd)
- `output/` — Pulpie extraction results
- `reference/` — API-extracted text (from qrater datasets) for comparison
- `results/` — Per-method benchmark scores backing `BENCHMARKS.md`
- `manifest.json` — Metadata for all pages

## Harness scripts

The numbers in the project README / `BENCHMARKS.md` are reproduced by:

- `eval_latte_large_vs_dripper.py` — ROUGE-5 + qrater clean-rate, Pulpie Orange
  vs Dripper on WebMainBench (the canonical quality harness; see also
  `tests/gpu/AGENTS.md`).
- `bench_all_methods.py` — multi-method comparison (Pulpie vs Dripper,
  Trafilatura, magic-html, raw html2text) for throughput + quality.
- `eval_all_methods_200.py` — the 200-page variant of the above.

These run against the WebMainBench dataset + Orange model checkpoints, which live
in a gitignored `data/` dir on the GPU box (not in the repo). They import the
chunker from the installed `pulpie` package.

## Known Failures

Pages 07 and 10 are JS-rendered SPAs with no static content — expected failure for any non-headless extraction pipeline.
