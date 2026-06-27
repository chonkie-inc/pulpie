"""Test fixtures for simplify/reconstruct parity against the MinerU-HTML oracle.

The oracle is a pinned copy of MinerU-HTML's ``mineru_html`` package, vendored
under ``tests/_oracle/`` (see ``tests/_oracle/SOURCE.md``). Its real top-level
``__init__`` pulls in a vllm/transformers inference stack we don't want in unit
tests, so we load only the three ``process`` modules we need via a synthetic-module
shim — the same pattern used in ``eval/eval_latte_large_vs_dripper.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

import pytest

_ORACLE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_oracle", "MinerU-HTML"
)
_PROCESS_DIR = os.path.join(_ORACLE_ROOT, "mineru_html", "process")


def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod


def _install_oracle_shim():
    """Build stub mineru_html.* modules and exec the real process files."""
    if "mineru_html.process.simplify_html" in sys.modules:
        return

    _make_module("mineru_html")

    constants = _make_module("mineru_html.constants")
    constants.ITEM_ID_ATTR = "_item_id"
    constants.TAIL_BLOCK_TAG = "cc-alg-uc-text"
    constants.SELECT_ATTR = "cc-select"
    constants.CLASS_ATTR = "mark-selected"

    class TagType(Enum):
        Main = "main"
        Other = "other"

    constants.TagType = TagType

    exc = _make_module("mineru_html.exceptions")

    class MinerUHTMLError(Exception):
        def set_case_id(self, case_id):
            self.case_id = case_id

    exc.MinerUHTMLError = MinerUHTMLError
    for cn in [
        "MinerUHTMLPreprocessError",
        "MinerUHTMLPromptError",
        "MinerUHTMLResponseParseError",
        "MinerUHTMLMapToMainError",
        "MinerUHTMLFallbackError",
    ]:
        setattr(exc, cn, type(cn, (MinerUHTMLError,), {}))

    base = _make_module("mineru_html.base")

    @dataclass
    class MinerUHTMLProcessData:
        simpled_html: str = ""
        map_html: str = ""

    @dataclass
    class MinerUHTMLOutput:
        main_html: str = ""

    @dataclass
    class MinerUHTMLInput:
        raw_html: str = ""

    @dataclass
    class MinerUHTMLCase:
        case_id: str = ""
        input_data: MinerUHTMLInput = field(default_factory=MinerUHTMLInput)
        process_data: MinerUHTMLProcessData = field(default_factory=MinerUHTMLProcessData)
        parse_result: object = None
        output_data: MinerUHTMLOutput = field(default_factory=MinerUHTMLOutput)

    for cls in (
        MinerUHTMLProcessData,
        MinerUHTMLOutput,
        MinerUHTMLInput,
        MinerUHTMLCase,
    ):
        setattr(base, cls.__name__, cls)

    _make_module("mineru_html.process")

    def _load(mod_name, filename):
        path = os.path.join(_PROCESS_DIR, filename)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("mineru_html.process.html_utils", "html_utils.py")
    _load("mineru_html.process.simplify_html", "simplify_html.py")
    _load("mineru_html.process.map_to_main", "map_to_main.py")


@pytest.fixture(scope="session")
def oracle_simplify():
    """Return MinerU's real ``simplify_html(raw_html, cutoff_length=500)``."""
    _install_oracle_shim()
    return sys.modules["mineru_html.process.simplify_html"].simplify_html


@pytest.fixture(scope="session")
def oracle_extract_main():
    """Return MinerU's real ``extract_main_html(map_html, labels)``."""
    _install_oracle_shim()
    return sys.modules["mineru_html.process.map_to_main"].extract_main_html


def _fixture_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    html_dir = os.path.normpath(os.path.join(here, "..", "..", "eval", "html"))
    return sorted(
        os.path.join(html_dir, f) for f in os.listdir(html_dir) if f.endswith(".html")
    )


def pytest_generate_tests(metafunc):
    """Parametrize any test taking ``fixture_html`` over the eval/html corpus."""
    if "fixture_html" in metafunc.fixturenames:
        paths = _fixture_paths()
        ids = [os.path.basename(p) for p in paths]
        contents = [open(p, encoding="utf-8", errors="replace").read() for p in paths]
        metafunc.parametrize("fixture_html", contents, ids=ids)
