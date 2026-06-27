"""Byte/block parity of pulpie's simplify + reconstruct against the MinerU oracle.

pulpie's Orange models were distilled to match MinerU-HTML's simplification format
exactly. These tests assert that pulpie reproduces MinerU's output on a corpus of
real pages, using MinerU's own code (vendored under ``tests/_oracle/``) as a golden
master. Parametrized over ``eval/html/*.html`` via ``conftest.pytest_generate_tests``.
"""

from __future__ import annotations

import re

from pulpie.chunker import extract_blocks
from pulpie.reconstruct import extract_main_html
from pulpie.simplify import simplify


def _norm(s: str) -> str:
    """Collapse insignificant whitespace so only substantive diffs fail.

    Mirrors MinerU's ``post_process_html`` intent: whitespace *between* tags is not
    semantically meaningful, but whitespace inside text is. We collapse runs of
    whitespace globally and strip — enough to absorb serializer cosmetics (newlines,
    indentation) while still catching real text/structure differences.
    """
    return re.sub(r"\s+", " ", s).strip()


def _block_texts(simplified_html: str) -> list[str]:
    """Per-block normalized visible text, keyed implicitly by position/_item_id."""
    blocks = extract_blocks(simplified_html)
    texts = []
    for b in blocks:
        # strip tags, keep text — segmentation parity is about text grouping
        text = re.sub(r"<[^>]+>", " ", b)
        texts.append(_norm(text))
    return texts


def test_simplified_byte_parity(fixture_html, oracle_simplify):
    """pulpie.simplify()[0] reproduces MinerU's simplified_html (normalized)."""
    ours, _ = simplify(fixture_html)
    theirs, _ = oracle_simplify(fixture_html)
    assert _norm(ours) == _norm(theirs)


def test_map_html_byte_parity(fixture_html, oracle_simplify):
    """pulpie.simplify()[1] reproduces MinerU's map_html (normalized)."""
    _, ours = simplify(fixture_html)
    _, theirs = oracle_simplify(fixture_html)
    assert _norm(ours) == _norm(theirs)


def test_block_sequence_parity(fixture_html, oracle_simplify):
    """Block count, order, and per-block text match MinerU (diagnostic localizer)."""
    ours, _ = simplify(fixture_html)
    theirs, _ = oracle_simplify(fixture_html)
    ours_blocks = _block_texts(ours)
    theirs_blocks = _block_texts(theirs)

    assert len(ours_blocks) == len(theirs_blocks), (
        f"block count: ours={len(ours_blocks)} vs oracle={len(theirs_blocks)}"
    )
    for i, (a, b) in enumerate(zip(ours_blocks, theirs_blocks)):
        assert a == b, f"block {i} of {len(theirs_blocks)} differs:\n ours={a!r}\n  ref={b!r}"


def test_reconstruct_parity(fixture_html, oracle_simplify, oracle_extract_main):
    """extract_main_html() matches MinerU given the same map_html + labels.

    Uses MinerU's own map_html as the shared input so this isolates reconstruct
    semantics (keep-main) from any simplify divergence.
    """
    _, map_html = oracle_simplify(fixture_html)
    ids = sorted(set(re.findall(r'_item_id="(\d+)"', map_html)), key=int)
    # Deterministic synthetic labels: alternate main/other.
    labels = {item_id: ("main" if i % 2 == 0 else "other") for i, item_id in enumerate(ids)}

    ours = extract_main_html(map_html, labels)
    theirs = oracle_extract_main(map_html, labels)
    assert _norm(ours) == _norm(theirs)


def test_roundtrip_smoke(fixture_html):
    """simplify -> extract_blocks -> labels -> reconstruct runs and preserves ids."""
    simplified, map_html = simplify(fixture_html)
    blocks = extract_blocks(simplified)
    ids = re.findall(r'_item_id="(\d+)"', simplified)
    assert len(blocks) == len(ids)

    labels = {item_id: "main" for item_id in ids}
    out = extract_main_html(map_html, labels)
    assert isinstance(out, str)
