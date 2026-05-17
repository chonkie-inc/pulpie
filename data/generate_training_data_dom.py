"""Generate training data using DOM-based label projection.

Instead of word overlap, we walk the annotated DOM the same way pulpie
does, and check if each block element contains cc-select="true" descendants.
Then we match those labels to pulpie's Rust-extracted features by text.
"""

import csv
import json
import os
import re
import subprocess
import sys

from bs4 import BeautifulSoup, Tag

PULPIE_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "target", "release", "export_features"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webmainbench.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data_dom.csv")

# Must match pulpie's segment.rs
BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}
CONTAINER_TAGS = {"div", "section", "article", "main", "body", "nav", "footer", "aside", "header"}
MIN_TEXT_LEN = 5

# Feature columns (must match Rust Features struct)
FEATURE_COLS = [
    "text_len", "word_count", "sentence_count", "comma_count",
    "avg_word_length", "stop_word_ratio", "capitalization_ratio",
    "punctuation_density", "has_copyright", "has_date_pattern",
    "link_len", "link_count", "link_ratio", "tag_count",
    "paragraph_count", "heading_count", "list_item_count", "image_count",
    "text_to_tag_ratio",
    "class_id_score", "parent_class_id_score", "tag_type", "tag_type_score",
    "dom_depth", "position", "distance_from_end",
    "is_first_10pct", "is_last_10pct", "has_boilerplate_class",
    "parent_tag_type", "semantic_ancestor",
    "prev_block_text_len", "prev_block_link_ratio",
    "next_block_text_len", "next_block_link_ratio",
    "blocks_since_heading", "blocks_until_heading",
    "section_heading_text_len", "section_block_count", "section_link_density",
    "page_total_blocks", "page_total_text_len", "page_total_link_ratio",
    "page_heading_count", "block_text_len_ratio",
    "in_nav", "in_footer", "in_aside", "in_header",
]

TAG_TYPE_MAP = {
    "Paragraph": 0, "Heading": 1, "ListItem": 2,
    "Preformatted": 3, "TableCell": 4, "Blockquote": 5, "Other": 6,
}


def has_cc_select(element):
    """Check if element or any descendant has cc-select='true'."""
    if not isinstance(element, Tag):
        return False
    if element.get("cc-select") == "true":
        return True
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.get("cc-select") == "true":
            return True
    return False


def get_text(element):
    """Get text content of element (matching pulpie's element.text().collect())."""
    return element.get_text()


def has_block_descendants(element):
    """Check if element has block-level tag descendants."""
    for desc in element.descendants:
        if isinstance(desc, Tag) and desc.name in BLOCK_TAGS:
            return True
    return False


def walk_dom(element, blocks):
    """Walk DOM same as pulpie's segment.rs walk()."""
    if not isinstance(element, Tag):
        return

    tag = element.name

    if tag in BLOCK_TAGS:
        text = get_text(element).strip()
        if len(text) >= MIN_TEXT_LEN:
            blocks.append(element)
        return

    if tag in CONTAINER_TAGS or tag not in BLOCK_TAGS:
        if has_block_descendants(element):
            for child in element.children:
                if isinstance(child, Tag):
                    walk_dom(child, blocks)
        else:
            text = get_text(element).strip()
            if len(text) >= MIN_TEXT_LEN:
                blocks.append(element)
        return


def label_blocks_dom(html):
    """Parse annotated HTML, walk DOM, label each block by cc-select presence.

    Returns list of (normalized_text, label) tuples.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    blocks = []
    walk_dom(body, blocks)

    labeled = []
    for block in blocks:
        text = get_text(block)
        norm = normalize(text)
        label = 1 if has_cc_select(block) else 0
        labeled.append((norm, label))

    return labeled


def normalize(text):
    """Normalize text for matching between Python and Rust outputs."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def strip_annotations(html):
    """Remove cc-select, annotation attrs, and marked-text/marked-tail wrappers."""
    html = re.sub(r'\s+cc-select="[^"]*"', '', html)
    html = re.sub(r'\s+data-anno-uid="[^"]*"', '', html)
    html = re.sub(r'\s+class="mark-selected"', '', html)
    html = re.sub(r'</?marked-text[^>]*>', '', html)
    html = re.sub(r'</?marked-tail[^>]*>', '', html)
    return html


def run_export_features(html):
    """Run pulpie export_features binary on HTML."""
    result = subprocess.run(
        [PULPIE_BIN],
        input=html, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def match_blocks(rust_blocks, dom_labels):
    """Match Rust blocks to DOM-labeled blocks by normalized text.

    Returns list of (rust_block, label) pairs.
    """
    # Build lookup from normalized text -> label
    # If multiple DOM blocks have the same text, store all labels
    label_map = {}
    for norm_text, label in dom_labels:
        if norm_text not in label_map:
            label_map[norm_text] = []
        label_map[norm_text].append(label)

    matched = []
    for rb in rust_blocks:
        norm = normalize(rb["text"])
        if norm in label_map and label_map[norm]:
            label = label_map[norm].pop(0)
            matched.append((rb, label))
        # If no match, skip this block (don't train on uncertain labels)

    return matched


def feature_row(block, label):
    """Convert a block's features dict to a flat row."""
    f = block["features"]
    row = []
    for col in FEATURE_COLS:
        val = f[col]
        if isinstance(val, bool):
            row.append(int(val))
        elif col == "tag_type":
            row.append(TAG_TYPE_MAP.get(val, 6))
        else:
            row.append(val)
    row.append(label)
    return row


def main():
    if not os.path.exists(PULPIE_BIN):
        print("ERROR: Build pulpie first: cargo build --release")
        sys.exit(1)

    print(f"Reading {DATA_PATH}...", flush=True)
    total_pages = 0
    total_blocks = 0
    total_matched = 0
    total_keep = 0
    total_discard = 0
    skipped = 0

    with open(DATA_PATH) as fin, open(OUTPUT_PATH, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(FEATURE_COLS + ["label"])

        for line_no, line in enumerate(fin, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            html = rec.get("html", "")
            if not html or len(html) < 100:
                skipped += 1
                continue

            # Step 1: Label blocks via DOM walk on annotated HTML
            dom_labels = label_blocks_dom(html)
            if not dom_labels:
                skipped += 1
                continue

            # Step 2: Strip annotations, run Rust segmenter for features
            clean_html = strip_annotations(html)
            rust_blocks = run_export_features(clean_html)
            if not rust_blocks:
                skipped += 1
                continue

            # Step 3: Match by text
            matched = match_blocks(rust_blocks, dom_labels)
            if not matched:
                skipped += 1
                continue

            for rb, label in matched:
                writer.writerow(feature_row(rb, label))
                if label == 1:
                    total_keep += 1
                else:
                    total_discard += 1

            total_pages += 1
            total_blocks += len(rust_blocks)
            total_matched += len(matched)

            if total_pages % 100 == 0:
                match_rate = total_matched / max(total_blocks, 1) * 100
                print(f"  {total_pages} pages, {total_matched}/{total_blocks} matched ({match_rate:.0f}%), "
                      f"{total_keep} keep / {total_discard} discard, "
                      f"{skipped} skipped", flush=True)

    match_rate = total_matched / max(total_blocks, 1) * 100
    print(f"\nDone: {total_pages} pages", flush=True)
    print(f"  Matched: {total_matched}/{total_blocks} blocks ({match_rate:.1f}%)")
    print(f"  KEEP: {total_keep} ({total_keep/max(total_matched,1)*100:.1f}%)")
    print(f"  DISCARD: {total_discard} ({total_discard/max(total_matched,1)*100:.1f}%)")
    print(f"  Skipped: {skipped} pages")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
