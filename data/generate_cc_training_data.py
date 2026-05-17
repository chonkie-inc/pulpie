"""Generate training data from CC labeled pages (parallelized).

Projects Dripper's _item_id labels onto Pulpie's block segmentation
via text overlap matching. Uses multiprocessing for speed.

Input: cc_labeled_*.jsonl (labels) + cc_sampled*.jsonl (HTML)
Output: training_data_cc.csv
"""

import csv
import json
import os
import re
import subprocess
import sys
import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from multiprocessing import Pool
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PULPIE_BIN = os.path.join(SCRIPT_DIR, "..", "target", "release", "export_features")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "training_data_cc.csv")
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')
NUM_WORKERS = 32

FEATURE_COLS = [
    "link_ratio", "page_total_blocks", "parent_tag_type", "page_total_link_ratio",
    "tag_type", "section_block_count", "page_total_text_len", "dom_depth",
    "section_link_density", "position", "tag_type_score", "section_heading_text_len",
    "blocks_since_heading", "word_count", "next_block_link_ratio", "prev_block_link_ratio",
    "text_to_tag_ratio", "semantic_ancestor", "blocks_until_heading", "page_heading_count",
    "parent_class_id_score", "next_block_text_len", "link_count", "text_len",
    "tag_count", "avg_word_length",
    "in_nav", "in_footer", "in_aside", "in_header",
]

TAG_TYPE_MAP = {
    "Paragraph": 0, "Heading": 1, "ListItem": 2,
    "Preformatted": 3, "TableCell": 4, "Blockquote": 5, "Other": 6,
}


def _init_mineru():
    """Initialize MinerU-HTML modules in worker process."""
    def _make_module(name):
        mod = type(sys)(name)
        sys.modules[name] = mod
        return mod

    if 'mineru_html' not in sys.modules:
        _make_module('mineru_html')
        _c = _make_module('mineru_html.constants')
        _c.ITEM_ID_ATTR = '_item_id'
        _c.TAIL_BLOCK_TAG = 'cc-alg-uc-text'
        _c.SELECT_ATTR = 'cc-select'
        _c.CLASS_ATTR = 'mark-selected'
        class TagType(Enum):
            Main = 'main'
            Other = 'other'
        _c.TagType = TagType
        _e = _make_module('mineru_html.exceptions')
        class MinerUHTMLError(Exception): pass
        for cn in ['MinerUHTMLPreprocessError', 'MinerUHTMLPromptError',
                    'MinerUHTMLResponseParseError', 'MinerUHTMLMapToMainError',
                    'MinerUHTMLFallbackError']:
            setattr(_e, cn, type(cn, (MinerUHTMLError,), {}))
        _e.MinerUHTMLError = MinerUHTMLError
        _b = _make_module('mineru_html.base')
        @dataclass
        class MinerUHTMLProcessData: simpled_html: str = ''; map_html: str = ''
        @dataclass
        class MinerUHTMLGenerateInput: full_prompt: str = ''
        @dataclass
        class MinerUHTMLParseResult: item_label: dict = field(default_factory=dict)
        @dataclass
        class MinerUHTMLOutput: main_html: str = ''
        @dataclass
        class MinerUHTMLInput: raw_html: str = ''
        @dataclass
        class MinerUHTMLCase:
            case_id: str = ''
            input_data: MinerUHTMLInput = field(default_factory=MinerUHTMLInput)
            process_data: MinerUHTMLProcessData = field(default_factory=MinerUHTMLProcessData)
            generate_input: MinerUHTMLGenerateInput = field(default_factory=MinerUHTMLGenerateInput)
            generate_output: Optional[object] = None
            parse_result: MinerUHTMLParseResult = field(default_factory=MinerUHTMLParseResult)
            output_data: MinerUHTMLOutput = field(default_factory=MinerUHTMLOutput)
        for cls in [MinerUHTMLCase, MinerUHTMLProcessData, MinerUHTMLGenerateInput,
                    MinerUHTMLParseResult, MinerUHTMLOutput, MinerUHTMLInput]:
            setattr(_b, cls.__name__, cls)
        _make_module('mineru_html.process')
        def _load_file(mod_name, filename):
            path = os.path.join(MINERU_PATH, 'mineru_html', 'process', filename)
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod
        _load_file('mineru_html.process.html_utils', 'html_utils.py')
        _load_file('mineru_html.process.simplify_html', 'simplify_html.py')


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def process_page(args):
    """Process a single page. Returns list of (feature_row, label) or None."""
    url, html, labels = args

    _init_mineru()
    simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
    from bs4 import BeautifulSoup

    # Step 1: Dripper simplification for label texts
    try:
        simplified, _ = simplify_html(html)
    except Exception:
        return None, 'simplify_error'

    soup = BeautifulSoup(simplified, 'html.parser')
    dripper_texts = {}
    for el in soup.find_all(attrs={'_item_id': True}):
        iid = el['_item_id']
        text = normalize(el.get_text())
        if text:
            dripper_texts[iid] = text

    if not dripper_texts:
        return None, 'no_dripper_texts'

    # Step 2: Pulpie features on raw HTML
    try:
        result = subprocess.run(
            [PULPIE_BIN], input=html, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None, 'pulpie_error'
        rust_blocks = json.loads(result.stdout)
    except Exception:
        return None, 'pulpie_error'

    if not rust_blocks:
        return None, 'no_blocks'

    # Step 3: Match by text overlap
    rows = []
    for rb in rust_blocks:
        rb_text = normalize(rb["text"])
        if not rb_text or len(rb_text) < 5:
            continue

        best_iid = None
        best_overlap = 0

        for iid, d_text in dripper_texts.items():
            if iid not in labels:
                continue
            if rb_text in d_text or d_text in rb_text:
                overlap = min(len(rb_text), len(d_text))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_iid = iid
            else:
                rb_words = set(rb_text.split())
                d_words = set(d_text.split())
                if not rb_words:
                    continue
                overlap_ratio = len(rb_words & d_words) / len(rb_words)
                if overlap_ratio > 0.7:
                    overlap = len(rb_words & d_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_iid = iid

        if best_iid is not None:
            label = 1 if labels[best_iid] == 'main' else 0
            f = rb["features"]
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
            rows.append(row)

    if not rows:
        return None, 'no_match'

    return rows, 'ok'


def main():
    if not os.path.exists(PULPIE_BIN):
        print("ERROR: Build first: cargo build --release")
        sys.exit(1)

    # Load all labels
    all_labels = {}
    for path in ["cc_labeled_filtered.jsonl", "cc_labeled_dripper_83k.jsonl"]:
        full_path = os.path.join(SCRIPT_DIR, path)
        if not os.path.exists(full_path):
            continue
        with open(full_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('status') == 'ok' and rec.get('labels'):
                    all_labels[rec['url']] = rec['labels']
    print(f"Loaded labels for {len(all_labels)} pages", flush=True)

    # Collect work items
    work = []
    for html_path in ["cc_sampled.jsonl", "cc_sampled_100k.jsonl"]:
        full_path = os.path.join(SCRIPT_DIR, html_path)
        if not os.path.exists(full_path):
            continue
        print(f"Reading {html_path}...", flush=True)
        with open(full_path) as fin:
            for line in fin:
                rec = json.loads(line)
                url = rec.get('url', '')
                html = rec.get('html', '')
                if url in all_labels and html and len(html) >= 100:
                    work.append((url, html, all_labels[url]))

    print(f"Processing {len(work)} pages with {NUM_WORKERS} workers...", flush=True)

    total_blocks = 0
    total_keep = 0
    total_discard = 0
    status_counts = {}

    with open(OUTPUT_PATH, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(FEATURE_COLS + ["label"])

        with Pool(NUM_WORKERS) as pool:
            for i, (rows, status) in enumerate(pool.imap_unordered(process_page, work, chunksize=16)):
                status_counts[status] = status_counts.get(status, 0) + 1

                if rows:
                    for row in rows:
                        writer.writerow(row)
                        if row[-1] == 1:
                            total_keep += 1
                        else:
                            total_discard += 1
                    total_blocks += len(rows)

                if (i + 1) % 1000 == 0:
                    pages_ok = status_counts.get('ok', 0)
                    print(f"  {i+1}/{len(work)} done, {pages_ok} ok, "
                          f"{total_blocks} blocks ({total_keep} keep / {total_discard} discard), "
                          f"errors: {dict((k,v) for k,v in status_counts.items() if k != 'ok')}",
                          flush=True)

    pages_ok = status_counts.get('ok', 0)
    print(f"\nDone: {pages_ok} pages, {total_blocks} blocks", flush=True)
    print(f"  KEEP: {total_keep} ({total_keep/max(total_blocks,1)*100:.1f}%)")
    print(f"  DISCARD: {total_discard} ({total_discard/max(total_blocks,1)*100:.1f}%)")
    print(f"  Status: {status_counts}")
    print(f"  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
