"""Reconstruct main-content HTML from classifier labels.

Given the ``map_html`` produced by :func:`pulpie.simplify.simplify` and a dict of
``_item_id -> "main"|"other"`` labels, keep the main-labeled elements (plus their
ancestors and descendants) and drop everything else.

Faithful port of MinerU-HTML's ``extract_main_html``
(https://github.com/opendatalab/MinerU-HTML, commit 73cf266, Apache-2.0,
``mineru_html/process/map_to_main.py``), adapted to pulpie's ``dict[str, str]``
label API. Must stay paired with ``simplify`` — it relies on the ``_item_id``
placement and ``cc-alg-uc-text`` tail wrappers that simplify emits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pulpie._html_utils import decode_http_urls_only, element_to_html, html_to_element

if TYPE_CHECKING:
    from lxml import html

ITEM_ID_ATTR = "_item_id"
TAIL_BLOCK_TAG = "cc-alg-uc-text"


def _remove_recursive_by_condition(
    root: html.HtmlElement, remove_condition: Callable[[html.HtmlElement], bool]
) -> html.HtmlElement:
    """Remove elements matching ``remove_condition``; recurse only into survivors."""
    current_removed = False
    if remove_condition(root):
        parent = root.getparent()
        if parent is not None:
            parent.remove(root)
            current_removed = True
    if not current_removed:
        for child in root.iterchildren():
            _remove_recursive_by_condition(child, remove_condition)
    return root


def extract_main_html(map_html: str, labels: dict[str, str]) -> str:
    """Keep main-labeled content from ``map_html``, dropping the rest.

    Args:
        map_html: Original HTML with ``_item_id`` markers (from ``simplify``).
        labels: Dict mapping ``_item_id`` -> ``"main"`` or ``"other"``.

    Returns:
        Main-content HTML string.
    """
    if not map_html:
        return ""

    root = html_to_element(map_html)

    # Map each _item_id to its first element in document order (matches the
    # previous per-id ``xpath(...)[0]`` lookup, in a single tree traversal).
    id_to_element: dict[str, html.HtmlElement] = {}
    for el in root.iter():
        iid = el.get(ITEM_ID_ATTR)
        if iid is not None and iid not in id_to_element:
            id_to_element[iid] = el

    elements_to_remain: set = set()
    for remained_id, label in labels.items():
        if label != "main":
            continue
        elem = id_to_element.get(remained_id)
        if elem is None:
            continue
        for child in elem.iter():
            elements_to_remain.add(child)
        for ancestor in elem.iterancestors():
            elements_to_remain.add(ancestor)

    # Recall <br> tags adjacent to kept (non-br) content.
    last_element: html.HtmlElement | None = None
    for element in root.iter():
        if last_element is not None:
            if element.tag == "br" and (
                last_element in elements_to_remain and last_element.tag != "br"
            ):
                elements_to_remain.add(element)
            if last_element.tag == "br" and (
                element in elements_to_remain and element.tag != "br"
            ):
                elements_to_remain.add(last_element)
        last_element = element

    _remove_recursive_by_condition(root, lambda x: x not in elements_to_remain)

    for tail_block in root.xpath(f"//{TAIL_BLOCK_TAG}"):
        tail_block.drop_tag()

    return decode_http_urls_only(element_to_html(root))
