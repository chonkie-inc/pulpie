"""Pulpie: fast content extraction from HTML using encoder models."""

from pulpie.extractor import ExtractionResult, Extractor
from pulpie.pipeline import PageInput, PageResult, Pipeline
from pulpie.simplify import simplify

__version__ = "0.0.1"
__all__ = ["ExtractionResult", "Extractor", "PageInput", "PageResult", "Pipeline", "simplify"]
