"""Low-level HTML element <-> string helpers.

Ported from MinerU-HTML (https://github.com/opendatalab/MinerU-HTML,
commit 73cf266, Apache-2.0) ``mineru_html/process/html_utils.py``. Kept
behaviorally identical so pulpie's simplified/reconstructed output matches the
format the Orange models were trained on.
"""

from __future__ import annotations

import html as _html
import re

from lxml import html as lxml_html
from lxml.etree import ParserError


def html_to_element(html_str: str) -> lxml_html.HtmlElement:
    """Parse an HTML string into an lxml element (MinerU-compatible parser opts)."""
    parser = lxml_html.HTMLParser(
        collect_ids=False,
        encoding="utf-8",
        remove_blank_text=True,
        remove_comments=True,
        remove_pis=True,
    )
    if isinstance(html_str, str) and (
        "<?xml" in html_str or "<meta charset" in html_str or "encoding=" in html_str
    ):
        html_input = html_str.encode("utf-8")
    else:
        html_input = html_str

    try:
        root = lxml_html.fromstring(html_input, parser=parser)
    except ParserError as e:
        if "Document is empty" in str(e):
            return lxml_html.HtmlElement()
        raise
    return root


def element_to_html(root: lxml_html.HtmlElement, pretty_print: bool = False) -> str:
    """Serialize an lxml element back to an HTML string."""
    html_bytes = lxml_html.tostring(root, pretty_print=pretty_print, encoding="utf-8")
    return html_bytes.decode("utf-8") if isinstance(html_bytes, bytes) else html_bytes


def decode_http_urls_only(html_str: str) -> str:
    """Unescape HTML entities only inside http(s)/ftp/protocol-relative href/src URLs."""

    def decode_match(match):
        prefix = match.group(1)
        url = match.group(2)
        suffix = match.group(3)
        if url.startswith(
            ("http://", "https://", "ftp://", "HTTP://", "HTTPS://", "FTP://", "//")
        ):
            return f"{prefix}{_html.unescape(url)}{suffix}"
        return match.group(0)

    pattern = r'(href="|src=")(.*?)(")'
    return re.sub(pattern, decode_match, html_str, flags=re.IGNORECASE | re.DOTALL)
