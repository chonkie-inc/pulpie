# Design: Port MinerU-HTML simplification into pulpie

**Date:** 2026-06-27
**Branch:** `worktree-simplify-mineru-port`
**Status:** Approved (pending final spec review)

## Problem

pulpie's `simplify.py` (369 lines) produces fundamentally different block
segmentation than MinerU-HTML's `simplify_html.py` (1172 lines), which the
`opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact` model — and pulpie's distilled
Orange models — were trained on. Using pulpie's pipeline scores **0.731 ROUGE-5**
vs **0.862** using MinerU's pipeline on WebMainBench: a 13pp quality gap.

Documented root causes (from `work.log`):
1. **Block segmentation** — MinerU does deep recursive splitting with data-table
   vs layout-table analysis and splits content lists into individual items.
   pulpie treats lists as atomic blocks (71 blocks where MinerU produces 92).
2. **Tag removal** — MinerU removes `<nav>`; pulpie does not.
3. **Attribute handling** — MinerU keeps an allow-list (`class`, `id`, `_item_id`,
   `alt`/`src` for img); pulpie uses a deny-list and leaks attributes.
4. **Compounding** — fixing any one issue in isolation made scores *worse*
   (0.731 → 0.708/0.727), because the model expects MinerU's exact segmentation
   boundaries. Partial parity is worse than none.

## Strategy: Port-with-oracle

Bring MinerU's functions into pulpie essentially verbatim, adapt them to pulpie's
surface (lxml-native, return pulpie's `(simplified, map_html)` tuple, drop the
`MinerUHTMLCase`/exception scaffolding), and **prove byte-for-byte that the port
equals the upstream original** using MinerU's real code as a golden oracle.

This is *porting*, not *vendoring a dependency*: the code lives in pulpie's `src/`,
in pulpie's style, with no `mineru_html` import and an Apache-2.0 attribution header.
We own it; the oracle keeps us honest.

Rejected alternatives:
- **From-scratch reimplementation** — high risk of subtle parity bugs; the
  compounding failure mode makes "close but not exact" actively harmful.
- **Vendor `mineru_html` as a runtime dependency** — drags in `MinerUHTMLCase`
  scaffolding and an external package; less ownership of pulpie's core path.

## Module structure

All inside `pulpie/src/pulpie/`, lxml-native, no `mineru_html` imports.

| File | Change | Ported from |
|------|--------|-------------|
| `simplify.py` | **Rewritten.** Public `simplify(raw_html, cutoff_length=500) -> (simplified_html, map_html)` (unchanged signature). Internally: the full MinerU algorithm — `add_data_uids`, `is_data_table`, `is_content_list`, `extract_paragraphs`, `process_paragraphs`, `clean_attributes`, `simplify_list`, `truncate_html_element_selective`, `cc-alg-uc-text` tail wrappers. | `simplify_html.py` (1172 lines) |
| `reconstruct.py` | **Rewritten** to MinerU's *keep-main* semantics: keep main + ancestors + descendants, `<br>` recall adjacent to main, drop `cc-alg-uc-text` wrappers, decode http URLs. Replaces current *remove-other* logic. Same `extract_main_html(map_html, labels)` signature. | `map_to_main.py` |
| `_html_utils.py` | **New** (private). `html_to_element`, `element_to_html`, `decode_http_urls_only`. | `html_utils.py` |

### Stable contracts (rest of pipeline untouched)
- `simplify()` keeps its `(str, str)` return → `pipeline._cpu_prepare` and
  `chunker.extract_blocks` need **no changes**. The chunker splits on
  `_item_id="(\d+)"` via regex — format-agnostic, verified.
- `reconstruct.extract_main_html(map_html, labels)` keeps its signature; only the
  internals flip from remove-other to keep-main. `pipeline._postprocess` unchanged.

### Adaptations from upstream
1. **Parse fix-up:** port MinerU's `selectolax.HTMLParser` → BeautifulSoup fallback
   that repairs malformed HTML before lxml. Adds `selectolax` + `beautifulsoup4`
   as runtime deps. (Required for parity on real-world broken HTML — approved.)
2. **`map_html` format:** now carries `cc-alg-uc-text` wrappers and MinerU's
   `_item_id` placement. This is why `reconstruct` must be ported in lockstep —
   simplify and reconstruct are a matched pair.

## Dependencies

Add to `pulpie/pyproject.toml` runtime deps:
- `selectolax`
- `beautifulsoup4`

## Testing strategy (offline, zero-GPU)

### Oracle
Vendor a pinned copy of upstream `mineru_html` under
`pulpie/tests/_oracle/MinerU-HTML/` (test-only, not shipped). A `conftest.py`
loads the real `simplify_html` / `map_to_main` via the synthetic-module shim
pattern used in `eval/eval_latte_large_vs_dripper.py`, so the oracle is
reproducible and not dependent on `/tmp`.

### Three layers (all on the 13 `eval/html/*.html` fixtures, parametrized)
1. **Byte-parity (primary, strict):** `pulpie.simplify(html)` equals
   `oracle.simplify_html(html)` for both `simplified_html` and `map_html`, after a
   shared whitespace normalization. Strongest "faithfully reproduce" check.
2. **Block-sequence parity (diagnostic):** on failure, compare `extract_blocks()`
   sequences (count, order, per-block normalized text) to localize the divergent
   block.
3. **Reconstruct parity:** feed a synthetic label dict through both pulpie's and
   the oracle's `extract_main_html(map_html, labels)`; assert equal output.

### Round-trip smoke test
`simplify → extract_blocks → (fake labels) → reconstruct` runs without error and
preserves all `_item_id`s.

### Runner
`pytest -n auto`, one parametrized case per fixture. TDD: tests written first,
fail against current `simplify.py`, port until byte-parity is green on all 13.

## GPU acceptance gate (deferred, delegated)

Offline byte-parity proves we reproduce the trained format. Final quality
confirmation needs a GPU + the Orange model and is delegated to a GPU-box agent
via an **`AGENTS.md`** runbook (self-contained, zero session context required):
- Download `WebMainBench_545.jsonl` from `opendatalab/WebMainBench` (the 545-sample
  calibrated subset with `groundtruth_content`; ~109 MB — not the 1.4 GB full file).
- Load an Orange model, run the existing ROUGE-5 harness.
- **Acceptance gate:** pulpie's own pipeline reaches ~0.862 ROUGE-5 (parity with
  MinerU's pipeline), up from 0.731.

This is post-merge validation, not part of the offline TDD loop.

## Out of scope
- Branding/naming/visual-identity work (tracked separately; note that upstream
  already landed a "Rename Hummingbird → Pulpie" commit).
- Any change to the chunker, model loading, or pipeline orchestration.
