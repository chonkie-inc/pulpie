"""HTML simplification for block-level classification.

Converts raw HTML into a simplified representation where each content block has a
unique ``_item_id``, and a parallel ``map_html`` (the original DOM marked with the
same ``_item_id``s) used to reconstruct main content after classification.

This is a faithful port of MinerU-HTML's ``simplify_html``
(https://github.com/opendatalab/MinerU-HTML, commit 73cf266, Apache-2.0,
``mineru_html/process/simplify_html.py``), adapted to pulpie's API. The Orange
models were distilled to match MinerU's exact segmentation, so this module
reproduces its output rather than reimplementing the idea. Credit: MinerU-HTML /
Dripper team (Ma et al., 2025).
"""

from __future__ import annotations

import contextlib
import copy
import re
import uuid
from urllib.parse import quote

from bs4 import BeautifulSoup
from lxml import etree, html
from selectolax.parser import HTMLParser

# `simplify` is the only public symbol; everything else is a port-internal helper.
__all__ = ["simplify"]

# Inline tags (do not start a new block).
inline_tags = {
    "map", "optgroup", "span", "input", "time", "u", "strong", "small", "sub",
    "samp", "blink", "b", "code", "nobr", "strike", "bdo", "basefont", "abbr",
    "var", "i", "cccode-inline", "s", "pic", "label", "mark", "object",
    "ccmath-inline", "svg", "button", "a", "font", "dfn", "sup", "kbd", "q",
    "script", "acronym", "option", "img", "big", "cite", "em", "marked-tail",
    "marked-text",
}

# Table-related tags that may live within tables.
table_tags_set = {
    "caption", "colgroup", "col", "thead", "tbody", "tfoot", "tr", "td", "th", "br",
}

# Tags removed wholesale before paragraph extraction.
tags_to_remove = {"title", "head", "style", "script", "link", "meta", "iframe", "frame", "nav"}

# Treated as block-level with no block children by default.
no_block_tags = {"math"}

# Tags whose text is excluded from truncation length accounting.
no_calc_text_tags = {"math", "table"}

# Standalone class/id values removed when a direct child of <body>.
ATTR_PATTERNS_TO_REMOVE = {"nav"}

# style declarations that mark an element invisible (-> removed).
ATTR_INVISIBLE = {
    "display": "none",
    "font-size": "0px",
    "color": "transparent",
    "visibility": "hidden",
    "opacity": "0",
}

# Custom wrapper tag used for unwrapped/inline text runs.
tail_block_tag = "cc-alg-uc-text"


# ── uid tracking ──────────────────────────────────────────────────────────


def add_data_uids(dom: html.HtmlElement) -> None:
    """Tag every node with a unique ``data-uid`` for cross-DOM mapping."""
    for node in dom.iter():
        with contextlib.suppress(TypeError):
            node.set("data-uid", str(uuid.uuid4()))


def remove_all_uids(dom: html.HtmlElement) -> None:
    """Strip all ``data-uid`` attributes."""
    for node in dom.iter():
        if "data-uid" in node.attrib:
            del node.attrib["data-uid"]


def build_uid_map(dom: html.HtmlElement) -> dict:
    """Map ``data-uid`` -> node."""
    return {node.get("data-uid"): node for node in dom.iter() if node.get("data-uid")}


# ── table / list classification ─────────────────────────────────────────────


def judge_table_parent(table_element, node_list) -> bool:
    """True if any node is a descendant of ``table_element`` (not a nested table)."""
    for node in node_list:
        ancestor = node.getparent()
        while ancestor is not None:
            if ancestor is table_element:
                return True
            elif ancestor.tag == "table":
                break
            ancestor = ancestor.getparent()
    return False


def is_data_table(table_element: html.HtmlElement) -> bool:
    """Distinguish a data table (content) from a layout table."""
    caption_nodes = table_element.xpath(".//caption")
    if judge_table_parent(table_element, caption_nodes):
        return True

    col_nodes = table_element.xpath(".//col")
    colgroup_nodes = table_element.xpath(".//colgroup")
    if judge_table_parent(table_element, col_nodes) or judge_table_parent(
        table_element, colgroup_nodes
    ):
        return True

    cell_nodes = table_element.xpath(".//*[self::td or self::th][@headers]")
    if judge_table_parent(table_element, cell_nodes):
        return True

    if table_element.get("role") == "table" or table_element.get("data-table"):
        return True

    for node in table_element.iterdescendants():
        if node.tag in table_tags_set:
            continue
        if node.tag not in inline_tags:
            return False

    return True


def has_non_listitem_children(list_element) -> bool:
    """True if a list has direct children that are not the expected item tags."""
    if list_element.tag in ("ul", "ol"):
        allowed_tags = {"li"}
    elif list_element.tag == "dl":
        allowed_tags = {"dt", "dd"}
    else:
        allowed_tags = set()

    exclude_conditions = " and ".join(f"name()!='{tag}'" for tag in allowed_tags)
    if exclude_conditions and list_element.xpath(f"./*[{exclude_conditions}]"):
        return True

    text_children = list_element.xpath("./text()")
    return any(text.strip() for text in text_children)


# ── paragraph extraction ─────────────────────────────────────────────────────


def extract_paragraphs(processing_dom, uid_map, include_parents: bool = True) -> list:
    """Walk the cleaned DOM and emit a flat, ordered list of content paragraphs."""
    table_types = {}
    for table in processing_dom.xpath(".//table"):
        table_types[table.get("data-uid")] = is_data_table(table)

    list_types: dict = {}

    def is_block_element(node) -> bool:
        def judge_special_case(node, expected_tags, types_map):
            ancestor = node
            while ancestor is not None and ancestor.tag not in expected_tags:
                ancestor = ancestor.getparent()
            if ancestor is not None:
                ancestor_uid = ancestor.get("data-uid")
                return not types_map.get(ancestor_uid, False)
            return None

        if node.tag in ("td", "th"):
            return judge_special_case(node, ["table"], table_types)
        if node.tag == "li":
            return judge_special_case(node, ["ul", "ol"], list_types)
        if node.tag in ("dt", "dd"):
            return judge_special_case(node, ["dl"], list_types)
        if node.tag in no_block_tags or node.tag in inline_tags:
            return False
        return isinstance(node, html.HtmlElement)

    def has_block_descendants(node) -> bool:
        if node.tag in no_block_tags:
            return False
        for child in node.iterdescendants():
            parent = child.getparent()
            if parent is not None and (
                parent.tag in no_block_tags or parent.get("cc-no-block") == "true"
            ):
                child.set("cc-no-block", "true")
            if child.get("cc-no-block") != "true" and is_block_element(child):
                if node.tag in inline_tags:
                    original_element = uid_map.get(node.get("data-uid"))
                    original_element.set("cc-block-type", "true")
                return True
        return False

    def is_content_list(list_element) -> bool:
        items = list_element.xpath("li | dt | dd")
        if len(items) == 0:
            return False
        if has_non_listitem_children(list_element):
            return False
        return all(not has_block_descendants(item) for item in items)

    for list_element in processing_dom.xpath(".//ul | .//ol | .//dl"):
        list_types[list_element.get("data-uid")] = is_content_list(list_element)

    def clone_structure(path):
        if not path:
            raise ValueError("Path cannot be empty")
        if not include_parents:
            last_node = html.Element(path[-1].tag)
            last_node.attrib.update(path[-1].attrib)
            return last_node, last_node

        root = html.Element(path[0].tag)
        root.attrib.update(path[0].attrib)
        current = root
        for node in path[1:-1]:
            new_node = html.Element(node.tag)
            new_node.attrib.update(node.attrib)
            current.append(new_node)
            current = new_node

        last_node = html.Element(path[-1].tag)
        last_node.attrib.update(path[-1].attrib)
        current.append(last_node)
        return root, last_node

    paragraphs = []

    def merge_inline_content(parent, content_list):
        last_inserted = None
        for idx, (item_type, item) in enumerate(content_list):
            if item_type in ("direct_text", "tail_text"):
                if last_inserted is None:
                    if not parent.text:
                        parent.text = item
                    else:
                        parent.text += " " + item
                else:
                    if last_inserted.tail is None:
                        last_inserted.tail = item
                    else:
                        last_inserted.tail += " " + item
            else:
                item_copy = copy.deepcopy(item)
                if idx == len(content_list) - 1 and item_copy.tag == "br":
                    item_copy.tail = None
                parent.append(item_copy)
                last_inserted = item

    def process_node(node, path):
        current_path = [*path, node]
        inline_content = []
        content_sources = []

        if node.text and node.text.strip():
            inline_content.append(("direct_text", node.text.strip()))
            content_sources.append("direct_text")

        for child in node:
            if is_block_element(child) or has_block_descendants(child):
                if child.tag == "br":
                    inline_content.append(("element", child))
                    content_sources.append("element")
                if inline_content:
                    try:
                        root, last_node = clone_structure(current_path)
                        merge_inline_content(last_node, inline_content)

                        content_type = "mixed"
                        if all(t == "direct_text" for t in content_sources):
                            content_type = "unwrapped_text"
                        elif all(t == "element" for t in content_sources):
                            content_type = "inline_elements"

                        original_element = uid_map.get(node.get("data-uid"))
                        paragraphs.append(
                            {
                                "html": etree.tostring(root, encoding="unicode").strip(),
                                "content_type": content_type,
                                "_original_element": original_element,
                            }
                        )
                    except ValueError:
                        pass
                    inline_content = []
                    content_sources = []
                if child.tag != "br":
                    if table_types.get(child.get("data-uid")) or (
                        not has_block_descendants(child)
                    ):
                        try:
                            root, last_node = clone_structure([*current_path, child])
                            last_node.text = child.text if child.text else None
                            for grandchild in child:
                                last_node.append(copy.deepcopy(grandchild))

                            original_element = uid_map.get(child.get("data-uid"))
                            paragraphs.append(
                                {
                                    "html": etree.tostring(root, encoding="unicode").strip(),
                                    "content_type": "block_element",
                                    "_original_element": original_element,
                                }
                            )
                        except ValueError:
                            pass
                    else:
                        process_node(child, current_path)

                if child.tail and child.tail.strip():
                    inline_content.append(("tail_text", child.tail.strip()))
                    content_sources.append("tail_text")
            else:
                inline_content.append(("element", child))
                content_sources.append("element")
                if child.tail and child.tail.strip():
                    inline_content.append(("tail_text", child.tail.strip()))
                    content_sources.append("tail_text")

        if inline_content:
            try:
                root, last_node = clone_structure(current_path)
                merge_inline_content(last_node, inline_content)

                content_type = "mixed"
                if all(t == "direct_text" for t in content_sources):
                    content_type = "unwrapped_text"
                elif all(t == "element" for t in content_sources):
                    content_type = "inline_elements"
                elif all(t in ("direct_text", "tail_text") for t in content_sources):
                    content_type = "unwrapped_text"

                original_element = uid_map.get(node.get("data-uid"))
                paragraphs.append(
                    {
                        "html": etree.tostring(root, encoding="unicode").strip(),
                        "content_type": content_type,
                        "_original_element": original_element,
                    }
                )
            except ValueError:
                pass

    process_node(processing_dom, [])

    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        if p["html"] not in seen:
            seen.add(p["html"])
            unique_paragraphs.append(p)
    return unique_paragraphs


# ── cleanup helpers ───────────────────────────────────────────────────────────


def remove_xml_declaration(html_string: str) -> str:
    return re.sub(r"<\?xml\s+.*?\??>", "", html_string, flags=re.DOTALL)


def post_process_html(html_content: str) -> str:
    if not html_content:
        return html_content

    def replace_outside_tag_space(match):
        if match.group(1):
            return match.group(1)
        elif match.group(2):
            return re.sub(r"\s+", " ", match.group(2))
        return match.group(0)

    html_content = re.sub(r"(<[^>]+>)|([^<]+)", replace_outside_tag_space, html_content)
    return html_content.strip()


def remove_tags(dom) -> None:
    for tag in tags_to_remove:
        for node in dom.xpath(f".//{tag}"):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)


def is_meaningful_content(element) -> bool:
    if element.text and element.text.strip():
        return True
    if element.tag == "img":
        src = element.get("src", "")
        return bool(src and src.strip())
    for child in element:
        if is_meaningful_content(child):
            return True
    return bool(element.tail and element.tail.strip())


def clean_attributes(element, short_src: bool = False) -> None:
    """Keep only class/id (and src/alt for img); drop everything else."""
    if element.tag == "img":
        src = element.get("src", "").strip()
        alt = element.get("alt", "").strip()
        class_attr = element.get("class", "").strip()
        id_attr = element.get("id", "").strip()
        element.attrib.clear()
        if src and not src.startswith("data:image/"):
            if short_src:
                if len(quote(src)) <= 10:
                    element.set("src", src)
                else:
                    element.set("src", src[:5] + "..." + src[-5:])
            else:
                element.set("src", src)
        if alt:
            element.set("alt", alt)
        if class_attr:
            element.set("class", class_attr)
        if id_attr:
            element.set("id", id_attr)
    else:
        class_attr = element.get("class", "").strip()
        id_attr = element.get("id", "").strip()
        element.attrib.clear()
        if class_attr:
            element.set("class", class_attr)
        if id_attr:
            element.set("id", id_attr)

    for child in element:
        clean_attributes(child)


def simplify_list(element) -> None:
    """Collapse long lists to first+last item with an ellipsis."""
    if element.tag in ("ul", "ol"):
        items = list(element.iterchildren())
        if len(items) > 2:
            for item in items[1:-1]:
                element.remove(item)
            ellipsis = etree.Element("span")
            ellipsis.text = "..."
            items[-1].addprevious(ellipsis)
    elif element.tag == "dl":
        items = list(element.iterchildren())
        if len(items) > 2:
            dts = [item for item in items if item.tag == "dt"]
            if len(dts) > 1:
                first_dt_index = items.index(dts[0])
                next_dt_index = items.index(dts[1])
                first_group = items[first_dt_index:next_dt_index]
                last_dt_index = items.index(dts[-1])
                last_group = items[last_dt_index:]
                for child in list(element.iterchildren()):
                    element.remove(child)
                for item in first_group:
                    element.append(item)
                ellipsis = etree.Element("span")
                ellipsis.text = "..."
                element.append(ellipsis)
                for item in last_group:
                    element.append(item)

    for child in element:
        simplify_list(child)


def should_remove_element(element) -> bool:
    class_name = element.get("class", "")
    id_name = element.get("id", "")

    if class_name in ATTR_PATTERNS_TO_REMOVE or id_name in ATTR_PATTERNS_TO_REMOVE:
        parent = element.getparent()
        if parent is not None and parent.tag == "body":
            return True

    style_attr = element.get("style", "")
    if style_attr:
        for attr in style_attr.split(";"):
            if ":" not in attr:
                continue
            split_items = attr.split(":")
            key = split_items[0]
            value = ":".join(split_items[1:])
            if ATTR_INVISIBLE.get(key.strip()) == value.strip():
                return True

    parent = element.getparent()
    if parent is not None and parent.tag == "details":
        if element.tag == "summary":
            return False
        else:
            return parent.get("open") is None

    return False


def remove_specific_elements(element) -> None:
    for child in list(element.iterchildren()):
        remove_specific_elements(child)

    if should_remove_element(element):
        parent = element.getparent()
        if parent is not None:
            tail_text = element.tail or ""
            element.tail = None
            prev_sibling = element.getprevious()
            if prev_sibling is not None:
                if prev_sibling.tail is not None:
                    prev_sibling.tail += tail_text
                else:
                    if prev_sibling.text is not None:
                        prev_sibling.text += tail_text
                    else:
                        prev_sibling.text = tail_text
            else:
                if parent.text is not None:
                    parent.text += tail_text
                else:
                    parent.text = tail_text
            parent.remove(element)


def truncate_html_element_selective(element, max_length, ellipsis="...", exclude_tags=None):
    """Truncate text to ``max_length`` chars, ignoring text inside ``exclude_tags``."""
    if exclude_tags is None:
        exclude_tags = set()

    def _is_excluded(node):
        current = node
        while current is not None:
            if current.tag in exclude_tags:
                return True
            current = current.getparent()
        return False

    def _is_inside_excluded_tag(node):
        return _is_excluded(node.getparent()) if node.getparent() is not None else False

    def _calculate_text_length(node):
        total_length = 0
        if node.text and not _is_excluded(node):
            total_length += len(node.text)
        for child in node:
            total_length += _calculate_text_length(child)
        if node.tail:
            total_length += len(node.tail)
        return total_length

    current_length = [0]
    ellipsis_added = [False]
    nodes_to_process = []

    def _collect_text_nodes(node):
        if node.text and not _is_excluded(node):
            nodes_to_process.append(
                {
                    "type": "text",
                    "node": node,
                    "original_text": node.text,
                    "can_modify": not _is_inside_excluded_tag(node),
                }
            )
        for child in node:
            _collect_text_nodes(child)
        if node.tail:
            nodes_to_process.append(
                {
                    "type": "tail",
                    "node": node,
                    "original_text": node.tail,
                    "can_modify": not _is_inside_excluded_tag(node),
                }
            )

    def _clean_ancestors_following_siblings(node):
        parent = node.getparent()
        if parent is None:
            return
        grandparent = parent.getparent()
        if grandparent is None:
            return
        children = list(grandparent)
        try:
            index = children.index(parent)
            for sibling in children[index + 1 :]:
                grandparent.remove(sibling)
        except ValueError:
            pass
        _clean_ancestors_following_siblings(parent)

    def _mark_truncation_point(truncate_node):
        parent = truncate_node.getparent()
        if parent is not None:
            children = list(parent)
            try:
                index = children.index(truncate_node)
                for sibling in children[index + 1 :]:
                    parent.remove(sibling)
            except ValueError:
                pass
        _clean_ancestors_following_siblings(truncate_node)

    def _process_text_nodes():
        for node_info in nodes_to_process:
            if ellipsis_added[0]:
                if node_info["type"] == "text":
                    node_info["node"].text = None
                else:
                    node_info["node"].tail = None
                continue
            text_len = len(node_info["original_text"])
            if current_length[0] + text_len <= max_length:
                current_length[0] += text_len
            else:
                if node_info["can_modify"]:
                    remaining = max_length - current_length[0]
                    truncated_text = node_info["original_text"][:remaining] + ellipsis
                    if node_info["type"] == "text":
                        node_info["node"].text = truncated_text
                    else:
                        node_info["node"].tail = truncated_text
                    current_length[0] = max_length
                    ellipsis_added[0] = True
                    _mark_truncation_point(node_info["node"])
                else:
                    current_length[0] += text_len

    total_text_length = _calculate_text_length(element)
    if total_text_length <= max_length:
        return element
    _collect_text_nodes(element)
    _process_text_nodes()
    return element


# ── paragraph -> simplified html + item-id mapping ───────────────────────────


def process_paragraphs(paragraphs, uid_map, cutoff_length: int = 500) -> str:
    """Clean each paragraph, assign ``_item_id``, and mark the original DOM."""
    result = []
    item_id = 1

    for para in paragraphs:
        try:
            root = html.fragment_fromstring(para["html"], create_parent=False)
            root_for_xpath = copy.deepcopy(root)
            content_type = para.get("content_type", "block_element")

            clean_attributes(root)
            simplify_list(root)

            if not is_meaningful_content(root):
                continue

            truncate_html_element_selective(
                root, max_length=cutoff_length, exclude_tags=no_calc_text_tags
            )

            current_id = str(item_id)
            root.set("_item_id", current_id)

            original_parent = para["_original_element"]
            if content_type != "block_element":
                if original_parent is not None:
                    original_element = uid_map.get(root_for_xpath.get("data-uid"))
                    if len(root_for_xpath) > 0:
                        if (
                            root_for_xpath.tag in inline_tags
                            and original_element.tag != "body"
                            and original_element.get("cc-block-type") != "true"
                        ):
                            original_element.set("_item_id", current_id)
                        else:
                            children_to_wrap = []
                            for child in root_for_xpath.iterchildren():
                                child_uid = child.get("data-uid")
                                if child_uid and child_uid in uid_map:
                                    children_to_wrap.append(uid_map[child_uid])

                            if children_to_wrap:
                                first_child = children_to_wrap[0]
                                last_child = children_to_wrap[-1]
                                start_idx = original_parent.index(first_child)
                                end_idx = original_parent.index(last_child)

                                nodes_to_wrap = [
                                    original_parent[i] for i in range(start_idx, end_idx + 1)
                                ]

                                leading_text = (
                                    original_parent.text
                                    if start_idx == 0
                                    else original_parent[start_idx - 1].tail
                                )
                                trailing_text = last_child.tail

                                wrapper = etree.Element(tail_block_tag)
                                wrapper.set("_item_id", current_id)
                                if original_parent.get("cc-select") is not None:
                                    wrapper.set("cc-select", original_parent.get("cc-select"))

                                if leading_text:
                                    wrapper.text = leading_text
                                    if start_idx == 0:
                                        original_parent.text = None
                                    else:
                                        original_parent[start_idx - 1].tail = None

                                for node in nodes_to_wrap:
                                    original_parent.remove(node)
                                    wrapper.append(node)

                                original_parent.insert(start_idx, wrapper)
                                if last_child.tag == "br" and trailing_text:
                                    wrapper.tail = trailing_text
                                    last_child.tail = None
                    else:
                        if content_type == "inline_elements":
                            original_element.set("_item_id", current_id)
                        else:
                            if root_for_xpath.text and root_for_xpath.text.strip():
                                found = False
                                if (
                                    original_parent.text
                                    and original_parent.text.strip()
                                    == root_for_xpath.text.strip()
                                ):
                                    wrapper = etree.Element(tail_block_tag)
                                    wrapper.set("_item_id", current_id)
                                    wrapper.text = original_parent.text
                                    if original_parent.get("cc-select") is not None:
                                        wrapper.set("cc-select", original_parent.get("cc-select"))
                                    original_parent.text = None
                                    if len(original_parent) > 0:
                                        original_parent.insert(0, wrapper)
                                    else:
                                        original_parent.append(wrapper)
                                    found = True

                                if not found:
                                    for child in original_parent.iterchildren():
                                        if (
                                            child.tail
                                            and child.tail.strip()
                                            == root_for_xpath.text.strip()
                                        ):
                                            wrapper = etree.Element(tail_block_tag)
                                            wrapper.set("_item_id", current_id)
                                            wrapper.text = child.tail
                                            if original_parent.get("cc-select") is not None:
                                                wrapper.set(
                                                    "cc-select",
                                                    original_parent.get("cc-select"),
                                                )
                                            child.tail = None
                                            parent = child.getparent()
                                            index = parent.index(child)
                                            parent.insert(index + 1, wrapper)
                                            break
            else:
                original_parent.set("_item_id", current_id)
                for child in original_parent.iterdescendants():
                    if child.get("cc-select") is not None:
                        original_parent.set("cc-select", child.get("cc-select"))
                        break

            item_id += 1

            cleaned_html = etree.tostring(root, method="html", encoding="unicode").strip()
            result.append({"html": cleaned_html, "_item_id": current_id, "content_type": content_type})

        except Exception:
            continue

    simplified_html = (
        '<html><head><meta charset="utf-8"></head><body>'
        + "".join(p["html"] for p in result)
        + "</body></html>"
    )
    return post_process_html(simplified_html)


def simplify(raw_html: str, cutoff_length: int = 500) -> tuple[str, str]:
    """Simplify raw HTML for block-level classification.

    Args:
        raw_html: Raw HTML string.
        cutoff_length: Maximum character length for text in each block.

    Returns:
        Tuple of ``(simplified_html, map_html)``:
        - ``simplified_html``: cleaned HTML, one block per ``_item_id``.
        - ``map_html``: the original DOM marked with matching ``_item_id``s, used
          by :func:`pulpie.reconstruct.extract_main_html`.
    """
    try:
        fixed_html = HTMLParser(raw_html).html
        if fixed_html is None:
            raise ValueError("selectolax produced no html")
    except Exception:
        fixed_html = str(BeautifulSoup(raw_html, "html.parser"))

    preprocessed_html = remove_xml_declaration(fixed_html)
    parser = html.HTMLParser(remove_comments=True)
    original_dom = html.fromstring(preprocessed_html, parser=parser)
    add_data_uids(original_dom)
    original_uid_map = build_uid_map(original_dom)

    processing_dom = copy.deepcopy(original_dom)
    remove_tags(processing_dom)
    remove_specific_elements(processing_dom)

    paragraphs = extract_paragraphs(processing_dom, original_uid_map, include_parents=False)
    simplified_html = process_paragraphs(paragraphs, original_uid_map, cutoff_length=cutoff_length)

    remove_all_uids(original_dom)
    map_html = etree.tostring(original_dom, pretty_print=False, method="html", encoding="unicode")

    return simplified_html, map_html
