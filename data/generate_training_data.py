"""Generate training data from WebMainBench for pulpie's GBM classifier.

Reads webmainbench.jsonl, runs pulpie's segmenter + feature extraction
on each page, projects cc-select="true" annotations to block-level KEEP/DISCARD
labels via word overlap.

Outputs: training_data.csv (features + label per block)
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from html.parser import HTMLParser

PULPIE_BIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "target", "release", "export_features"
)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webmainbench.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")

OVERLAP_THRESHOLD = 0.5

# Feature columns (must match Rust Features struct field order)
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
    "prev_block_text_len", "prev_block_link_ratio",
    "next_block_text_len", "next_block_link_ratio",
    "blocks_since_heading", "blocks_until_heading",
    "page_total_blocks", "page_total_text_len", "page_total_link_ratio",
    "page_heading_count", "block_text_len_ratio",
]

TAG_TYPE_MAP = {
    "Paragraph": 0, "Heading": 1, "ListItem": 2,
    "Preformatted": 3, "TableCell": 4, "Blockquote": 5, "Other": 6,
}


class CCSelectExtractor(HTMLParser):
    """Extract text from elements with cc-select='true'."""

    def __init__(self):
        super().__init__()
        self._in_selected = 0
        self.texts = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if attrs_dict.get("cc-select") == "true":
            self._in_selected += 1

    def handle_endtag(self, tag):
        if self._in_selected > 0:
            self._in_selected -= 1

    def handle_data(self, data):
        if self._in_selected > 0:
            self.texts.append(data)


def extract_cc_select_words(html: str) -> set:
    """Extract word set from cc-select='true' annotated elements."""
    parser = CCSelectExtractor()
    parser.feed(html)
    text = " ".join(parser.texts)
    words = set(re.findall(r'\w+', text.lower()))
    return words


def strip_annotations(html: str) -> str:
    """Remove cc-select, data-anno-uid, and marked-text/marked-tail wrapper tags."""
    # Remove annotation attributes
    html = re.sub(r'\s+cc-select="[^"]*"', '', html)
    html = re.sub(r'\s+data-anno-uid="[^"]*"', '', html)
    html = re.sub(r'\s+class="mark-selected"', '', html)
    # Unwrap <marked-text> and <marked-tail> (keep content)
    html = re.sub(r'</?marked-text[^>]*>', '', html)
    html = re.sub(r'</?marked-tail[^>]*>', '', html)
    return html


def compute_overlap(block_text: str, gt_words: set) -> float:
    """Fraction of block's words that appear in ground truth."""
    block_words = re.findall(r'\w+', block_text.lower())
    if not block_words:
        return 0.0
    matches = sum(1 for w in block_words if w in gt_words)
    return matches / len(block_words)


def run_export_features(html: str) -> list:
    """Run pulpie export_features binary on HTML, return block list."""
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


def feature_row(block: dict, label: int) -> list:
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
        print(f"ERROR: Build pulpie first: cargo build --release")
        sys.exit(1)

    print(f"Reading {DATA_PATH}...")
    total_pages = 0
    total_blocks = 0
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

            # Extract ground truth words from cc-select annotations
            gt_words = extract_cc_select_words(html)
            if len(gt_words) < 5:
                skipped += 1
                continue

            # Strip annotation markup before feeding to pulpie
            clean_html = strip_annotations(html)

            # Run pulpie segmenter + feature extraction
            blocks = run_export_features(clean_html)
            if not blocks:
                skipped += 1
                continue

            # Label each block
            page_keep = 0
            page_discard = 0
            for block in blocks:
                overlap = compute_overlap(block["text"], gt_words)
                label = 1 if overlap >= OVERLAP_THRESHOLD else 0
                writer.writerow(feature_row(block, label))
                if label == 1:
                    page_keep += 1
                else:
                    page_discard += 1

            total_pages += 1
            total_blocks += len(blocks)
            total_keep += page_keep
            total_discard += page_discard

            if total_pages % 100 == 0:
                print(f"  {total_pages} pages, {total_blocks} blocks "
                      f"({total_keep} keep / {total_discard} discard), "
                      f"{skipped} skipped")

    print(f"\nDone: {total_pages} pages, {total_blocks} blocks")
    print(f"  KEEP: {total_keep} ({total_keep/max(total_blocks,1)*100:.1f}%)")
    print(f"  DISCARD: {total_discard} ({total_discard/max(total_blocks,1)*100:.1f}%)")
    print(f"  Skipped: {skipped} pages")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
