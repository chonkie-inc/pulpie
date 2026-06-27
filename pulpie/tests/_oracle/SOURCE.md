# Vendored MinerU-HTML oracle (test-only)

This directory contains a pinned copy of MinerU-HTML's `mineru_html` package,
used **only** as a golden-master oracle for pulpie's simplify/reconstruct parity
tests. It is not shipped in the pulpie wheel (the package is scoped to `src/` via
`[tool.setuptools.packages.find] where = ["src"]`).

- **Upstream:** https://github.com/opendatalab/MinerU-HTML
- **Commit:** `73cf266690befd209cae7e6fdff9716d5b31a976`
- **License:** Apache-2.0

Only `mineru_html/process/{html_utils,simplify_html,map_to_main}.py` and their
light dependencies (`base.py`, `constants.py`, `exceptions.py`) are exercised; the
test conftest loads them via a synthetic-module shim (see
`pulpie/tests/conftest.py`) to avoid importing the full package's vllm/transformers
inference stack.

To refresh: re-clone upstream, copy `mineru_html/` here, update the commit SHA above.
