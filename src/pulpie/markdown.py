"""HTML -> Markdown conversion shared by Extractor and Pipeline.

Strips tracking-pixel / spacer images before conversion so they don't leak into
the output as empty `![](...)` tags, then renders with html2text. If html2text
isn't installed (it's an optional `[markdown]` extra), the cleaned HTML is
returned unchanged.
"""

from __future__ import annotations

import re

# <img> tags that carry no real content: 1x1 spacers, blank/transparent gifs,
# or data-URI placeholders. Matched case-insensitively on the whole tag.
_SPACER_IMG = re.compile(
    r"<img\b[^>]*?"
    r'(?:(?:width|height)\s*=\s*["\']?\s*1\b'  # width=1 / height=1
    r'|src\s*=\s*["\'][^"\']*'
    r"(?:trans(?:parent)?|spacer|blank|pixel|1x1|clear)[^\"']*[\"']"
    r"|src\s*=\s*[\"']data:image[^\"']*[\"'])"
    r"[^>]*>",
    re.IGNORECASE,
)


def strip_spacer_images(html: str) -> str:
    """Remove zero-content / tracking-pixel `<img>` tags."""
    return _SPACER_IMG.sub("", html)


def to_markdown(html: str) -> str:
    """Convert main-content HTML to Markdown.

    Returns the (spacer-stripped) HTML unchanged if html2text isn't available.
    """
    cleaned = strip_spacer_images(html)
    try:
        import html2text
    except ImportError:
        return cleaned

    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = False
    h.ignore_images = False
    return h.handle(cleaned).strip()
