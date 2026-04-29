"""Diagnose the ROUGE-5 gap between our block classifier (0.687) and Dripper (0.878).

Tests on first 50 WMB English pages from wmb_eval_sample.jsonl and computes:
  1. Oracle ceiling — perfect cc-select labels through the pipeline
  2. Simplify coverage — fraction of reference text present in map_html
  3. html2text alignment — where oracle output diverges from reference
  4. All-main baseline — label every block as "main"

Pipeline: simplify_html(html) -> classify blocks -> extract_main_html(map_html, labels) -> html2text -> ROUGE-5
"""

import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

import html2text as html2text_lib

# ── MinerU-HTML module loading (same shim as eval_block_classifier.py) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


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
    class MinerUHTMLProcessData:
        simpled_html: str = ''
        map_html: str = ''

    @dataclass
    class MinerUHTMLGenerateInput:
        full_prompt: str = ''

    @dataclass
    class MinerUHTMLParseResult:
        item_label: dict = field(default_factory=dict)

    @dataclass
    class MinerUHTMLOutput:
        main_html: str = ''

    @dataclass
    class MinerUHTMLInput:
        raw_html: str = ''

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
    _load_file('mineru_html.process.map_to_main', 'map_to_main.py')

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html
extract_main_html = sys.modules['mineru_html.process.map_to_main'].extract_main_html

from lxml import html as lhtml


# ── Scoring utilities ──

def html_to_text(html_str):
    """Convert HTML to plain text, matching eval_block_classifier.py settings."""
    h = html2text_lib.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    """ROUGE-N F1 with whitespace tokenization. Returns (f1, precision, recall)."""
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    ref_ng = Counter(ngrams(ref_tokens, n))
    pred_ng = Counter(ngrams(pred_tokens, n))
    if not ref_ng or not pred_ng:
        return 0.0, 0.0, 0.0
    overlap = sum((ref_ng & pred_ng).values())
    precision = overlap / max(sum(pred_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# ── Oracle label construction ──

def get_oracle_labels(map_html):
    """Parse map_html and determine which _item_id blocks should be 'main'.

    Strategy: For every element with cc-select="true" in the map_html, walk up
    the DOM tree to find the nearest ancestor (or self) with an _item_id attribute.
    Those item_ids are the oracle "main" blocks.

    Returns:
        (oracle_labels, all_item_ids, cc_select_item_ids)
        oracle_labels: dict mapping item_id -> 'main' or 'other'
        all_item_ids: set of all item_ids found
        cc_select_item_ids: set of item_ids marked as main by oracle
    """
    root = lhtml.fromstring(map_html)

    # Collect all _item_id values
    item_elements = root.xpath('//*[@_item_id]')
    all_item_ids = set(e.get('_item_id') for e in item_elements)

    # For each cc-select="true" element, find the enclosing _item_id block
    cc_select_item_ids = set()

    for elem in root.xpath('//*[@cc-select="true"]'):
        node = elem
        while node is not None:
            iid = node.get('_item_id')
            if iid:
                cc_select_item_ids.add(iid)
                break
            node = node.getparent()

    # Also check _item_id elements that directly have cc-select, or whose
    # descendants have cc-select (covers cases where _item_id is on an ancestor
    # of the cc-select element — handled above — and cases where _item_id element
    # itself has cc-select)
    for elem in item_elements:
        iid = elem.get('_item_id')
        if elem.get('cc-select') == 'true':
            cc_select_item_ids.add(iid)
        else:
            for desc in elem.iterdescendants():
                if desc.get('cc-select') == 'true':
                    cc_select_item_ids.add(iid)
                    break

    oracle_labels = {}
    for iid in all_item_ids:
        oracle_labels[iid] = 'main' if iid in cc_select_item_ids else 'other'

    return oracle_labels, all_item_ids, cc_select_item_ids


# ── Coverage analysis ──

def char_coverage(reference_text, candidate_text):
    """What fraction of reference words appear in the candidate?

    Uses word-level set overlap for speed. Returns the fraction of reference
    word tokens that appear at least once in the candidate.
    """
    if not reference_text:
        return 1.0
    if not candidate_text:
        return 0.0

    ref_words = reference_text.split()
    cand_word_set = set(candidate_text.split())

    if not ref_words:
        return 1.0

    matched = sum(1 for w in ref_words if w in cand_word_set)
    return matched / len(ref_words)


def find_divergences(reference, prediction, context=40):
    """Find the first few points where reference and prediction diverge.

    Returns a list of dicts with info about each divergence point.
    Uses only the first 2000 tokens of each to keep runtime bounded.
    """
    ref_tokens = reference.split()[:2000]
    pred_tokens = prediction.split()[:2000]

    sm = SequenceMatcher(None, ref_tokens, pred_tokens)
    opcodes = sm.get_opcodes()

    divergences = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        ref_span = ' '.join(ref_tokens[i1:i2])
        pred_span = ' '.join(pred_tokens[j1:j2])
        # Context around the divergence
        ctx_before_ref = ' '.join(ref_tokens[max(0, i1 - 3):i1])
        ctx_after_ref = ' '.join(ref_tokens[i2:i2 + 3])

        divergences.append({
            'type': tag,
            'ref_pos': (i1, i2),
            'pred_pos': (j1, j2),
            'ref_span': ref_span[:context],
            'pred_span': pred_span[:context],
            'context_before': ctx_before_ref,
            'context_after': ctx_after_ref,
        })
        if len(divergences) >= 8:
            break
    return divergences


# ── Main ──

def process_page(rec):
    """Process a single WMB page. Returns a results dict or None on failure."""
    html_content = rec.get('html', '')
    ref_text = rec.get('convert_main_content', '')
    url = rec.get('url', '')
    level = rec.get('meta', {}).get('level', 'unknown')

    if not html_content or not ref_text:
        return None

    # Step 1: simplify_html
    try:
        simplified, map_html = simplify_html(html_content)
    except Exception as e:
        return {
            'url': url, 'level': level, 'error': f'simplify_html failed: {e}',
            'oracle_r5': 0.0, 'all_main_r5': 0.0,
            'n_blocks': 0, 'n_oracle_main': 0,
            'simplify_coverage': 0.0,
            'oracle_f1': 0.0, 'oracle_prec': 0.0, 'oracle_recall': 0.0,
            'all_main_f1': 0.0, 'all_main_prec': 0.0, 'all_main_recall': 0.0,
            'divergences': [],
            'ref_len': len(ref_text), 'oracle_len': 0, 'all_main_len': 0,
        }

    # Step 2: Get oracle labels from cc-select
    oracle_labels, all_item_ids, cc_select_item_ids = get_oracle_labels(map_html)

    n_blocks = len(all_item_ids)
    n_oracle_main = len(cc_select_item_ids)

    # Step 3: Extract main HTML with oracle labels
    if n_oracle_main > 0:
        try:
            oracle_main_html = extract_main_html(map_html, oracle_labels)
            oracle_text = html_to_text(oracle_main_html).strip()
        except Exception as e:
            oracle_text = ''
    else:
        oracle_text = ''

    # Step 4: Extract main HTML with all-main labels
    all_main_labels = {iid: 'main' for iid in all_item_ids}
    try:
        all_main_html = extract_main_html(map_html, all_main_labels)
        all_main_text = html_to_text(all_main_html).strip()
    except Exception:
        all_main_text = ''

    # Step 5: Compute ROUGE-5 scores
    oracle_f1, oracle_prec, oracle_recall = rouge_n_f1(ref_text, oracle_text)
    all_main_f1, all_main_prec, all_main_recall = rouge_n_f1(ref_text, all_main_text)

    # Step 6: Simplify coverage - what fraction of reference chars are in map_html text?
    map_html_text = html_to_text(map_html).strip()
    coverage = char_coverage(ref_text, map_html_text)

    # Step 7: Divergence analysis between oracle output and reference
    divergences = find_divergences(ref_text, oracle_text)

    return {
        'url': url,
        'level': level,
        'n_blocks': n_blocks,
        'n_oracle_main': n_oracle_main,
        'oracle_f1': oracle_f1,
        'oracle_prec': oracle_prec,
        'oracle_recall': oracle_recall,
        'all_main_f1': all_main_f1,
        'all_main_prec': all_main_prec,
        'all_main_recall': all_main_recall,
        'simplify_coverage': coverage,
        'divergences': divergences,
        'ref_len': len(ref_text),
        'oracle_len': len(oracle_text),
        'all_main_len': len(all_main_text),
        'map_html_len': len(map_html_text),
        'error': None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose ROUGE-5 gap on WMB')
    parser.add_argument('--limit', type=int, default=50,
                        help='Number of pages to process (default: 50)')
    parser.add_argument('--data', type=str,
                        default=os.path.join(DATA_DIR, 'wmb_eval_sample.jsonl'),
                        help='Path to WMB eval JSONL')
    args = parser.parse_args()

    # Load data
    records = []
    with open(args.data) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                records.append(rec)
            if len(records) >= args.limit:
                break

    print(f'Loaded {len(records)} English pages from {args.data}', flush=True)
    print(f'Processing {len(records)} pages...\n', flush=True)

    results = []
    errors = []

    for i, rec in enumerate(records):
        result = process_page(rec)
        if result is None:
            continue
        results.append(result)
        if result.get('error'):
            errors.append(result)
        if (i + 1) % 10 == 0:
            print(f'  Processed {i + 1}/{len(records)}...', flush=True)

    n = len(results)
    if n == 0:
        print('No results to analyze!')
        return

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 1: Per-page results (first 10)
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'PER-PAGE RESULTS (first 10 of {n})')
    print(f'{"=" * 90}')
    print(f'{"Page":>4} {"Level":>6} {"Blks":>5} {"Main":>5} {"OracR5":>8} {"AllMnR5":>8} '
          f'{"Cov%":>6} {"OrcP":>6} {"OrcR":>6} {"Ref":>6} {"Orc":>6}')
    print(f'{"-" * 90}')

    for i, r in enumerate(results[:10]):
        print(f'{i:>4} {r["level"]:>6} {r["n_blocks"]:>5} {r["n_oracle_main"]:>5} '
              f'{r["oracle_f1"]:>8.4f} {r["all_main_f1"]:>8.4f} '
              f'{r["simplify_coverage"] * 100:>6.1f} '
              f'{r["oracle_prec"]:>6.3f} {r["oracle_recall"]:>6.3f} '
              f'{r["ref_len"]:>6} {r["oracle_len"]:>6}')

        # Show divergences for pages with oracle < 1.0
        if r['oracle_f1'] < 0.99 and r['divergences']:
            for d in r['divergences'][:3]:
                ref_show = d['ref_span'][:50] if d['ref_span'] else '(empty)'
                pred_show = d['pred_span'][:50] if d['pred_span'] else '(empty)'
                print(f'       {d["type"]:>8}: ref="{ref_show}" | pred="{pred_show}"')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 2: Oracle ceiling
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'SECTION 1: ORACLE CEILING (perfect cc-select labels through pipeline)')
    print(f'{"=" * 90}')

    oracle_f1s = [r['oracle_f1'] for r in results]
    oracle_precs = [r['oracle_prec'] for r in results]
    oracle_recalls = [r['oracle_recall'] for r in results]
    avg_oracle = sum(oracle_f1s) / n
    avg_oracle_p = sum(oracle_precs) / n
    avg_oracle_r = sum(oracle_recalls) / n

    print(f'\n  Average Oracle ROUGE-5 F1:        {avg_oracle:.4f}')
    print(f'  Average Oracle ROUGE-5 Precision: {avg_oracle_p:.4f}')
    print(f'  Average Oracle ROUGE-5 Recall:    {avg_oracle_r:.4f}')

    # Distribution
    perfect = sum(1 for f in oracle_f1s if f >= 0.99)
    high = sum(1 for f in oracle_f1s if 0.95 <= f < 0.99)
    medium = sum(1 for f in oracle_f1s if 0.80 <= f < 0.95)
    low = sum(1 for f in oracle_f1s if f < 0.80)
    print(f'\n  Oracle score distribution:')
    print(f'    >= 0.99 (perfect):  {perfect:>3} ({perfect / n * 100:>5.1f}%)')
    print(f'    0.95 - 0.99:        {high:>3} ({high / n * 100:>5.1f}%)')
    print(f'    0.80 - 0.95:        {medium:>3} ({medium / n * 100:>5.1f}%)')
    print(f'    < 0.80:             {low:>3} ({low / n * 100:>5.1f}%)')

    # By level
    print(f'\n  Oracle by difficulty:')
    for level in ['simple', 'mid', 'hard']:
        level_results = [r for r in results if r['level'] == level]
        if level_results:
            avg_f1 = sum(r['oracle_f1'] for r in level_results) / len(level_results)
            avg_p = sum(r['oracle_prec'] for r in level_results) / len(level_results)
            avg_r = sum(r['oracle_recall'] for r in level_results) / len(level_results)
            print(f'    {level:>8}: F1={avg_f1:.4f}  P={avg_p:.4f}  R={avg_r:.4f}  (n={len(level_results)})')

    # Pages where oracle is not perfect
    bad_oracle = [(i, r) for i, r in enumerate(results) if r['oracle_f1'] < 0.95]
    if bad_oracle:
        print(f'\n  Pages where oracle < 0.95 ({len(bad_oracle)} pages):')
        for idx, r in sorted(bad_oracle, key=lambda x: x[1]['oracle_f1']):
            print(f'    Page {idx}: F1={r["oracle_f1"]:.4f} P={r["oracle_prec"]:.4f} R={r["oracle_recall"]:.4f} '
                  f'blocks={r["n_blocks"]} main={r["n_oracle_main"]} '
                  f'ref={r["ref_len"]} orc={r["oracle_len"]} [{r["level"]}]')
            if r['divergences']:
                for d in r['divergences'][:2]:
                    ref_show = d['ref_span'][:60] if d['ref_span'] else '(empty)'
                    pred_show = d['pred_span'][:60] if d['pred_span'] else '(empty)'
                    print(f'          {d["type"]:>8}: ref="{ref_show}"')
                    print(f'                   pred="{pred_show}"')

    print(f'\n  KEY INSIGHT: If oracle avg is ~1.0, the pipeline architecture is NOT the')
    print(f'  bottleneck. The gap between 0.687 (model) and {avg_oracle:.3f} (oracle) is')
    print(f'  entirely due to classification errors.')
    gap = avg_oracle - 0.687
    print(f'  Classification-recoverable gap: {gap:.3f} ROUGE-5 F1 points')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 3: Simplify coverage
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'SECTION 2: SIMPLIFY COVERAGE (is simplify_html losing content?)')
    print(f'{"=" * 90}')

    coverages = [r['simplify_coverage'] for r in results]
    avg_cov = sum(coverages) / n

    print(f'\n  Average character coverage: {avg_cov * 100:.1f}%')
    print(f'  Min coverage: {min(coverages) * 100:.1f}%')
    print(f'  Max coverage: {max(coverages) * 100:.1f}%')

    low_cov = [(i, r) for i, r in enumerate(results) if r['simplify_coverage'] < 0.90]
    if low_cov:
        print(f'\n  Pages with coverage < 90% ({len(low_cov)} pages):')
        for idx, r in sorted(low_cov, key=lambda x: x[1]['simplify_coverage']):
            print(f'    Page {idx}: coverage={r["simplify_coverage"] * 100:.1f}% '
                  f'ref={r["ref_len"]} map_text={r.get("map_html_len", "?")} [{r["level"]}]')
    else:
        print(f'\n  All pages have >= 90% coverage. simplify_html is not losing content.')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 4: html2text alignment
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'SECTION 3: HTML2TEXT ALIGNMENT (oracle output vs reference)')
    print(f'{"=" * 90}')

    # Aggregate divergence types
    div_type_counts = Counter()
    div_total = 0
    for r in results:
        for d in r['divergences']:
            div_type_counts[d['type']] += 1
            div_total += 1

    print(f'\n  Total divergence operations across all pages: {div_total}')
    print(f'  By type:')
    for dtype, count in div_type_counts.most_common():
        print(f'    {dtype:>10}: {count:>4}')

    # Categorize divergences
    formatting_divs = 0
    content_divs = 0
    for r in results:
        for d in r['divergences']:
            ref_stripped = d['ref_span'].strip()
            pred_stripped = d['pred_span'].strip()
            # If after stripping whitespace/newlines the content is similar, it's formatting
            ref_words = set(ref_stripped.split())
            pred_words = set(pred_stripped.split())
            if ref_words and pred_words:
                jaccard = len(ref_words & pred_words) / len(ref_words | pred_words)
                if jaccard > 0.5:
                    formatting_divs += 1
                else:
                    content_divs += 1
            else:
                content_divs += 1

    print(f'\n  Estimated formatting-only divergences: {formatting_divs}')
    print(f'  Estimated content divergences:         {content_divs}')

    # Show sample divergences from worst oracle pages
    worst_oracle = sorted(results, key=lambda r: r['oracle_f1'])[:5]
    print(f'\n  Sample divergences from worst oracle pages:')
    for r in worst_oracle:
        if r['oracle_f1'] >= 0.999:
            continue
        print(f'\n    URL: {r["url"][:70]}  oracle_F1={r["oracle_f1"]:.4f}')
        for d in r['divergences'][:3]:
            ref_show = d['ref_span'][:70] if d['ref_span'] else '(empty)'
            pred_show = d['pred_span'][:70] if d['pred_span'] else '(empty)'
            ctx = d['context_before'][-30:] if d['context_before'] else ''
            print(f'      [{d["type"]}] after: "...{ctx}"')
            print(f'        ref:  "{ref_show}"')
            print(f'        pred: "{pred_show}"')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 5: All-main baseline
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'SECTION 4: ALL-MAIN BASELINE (every block labeled "main")')
    print(f'{"=" * 90}')

    all_main_f1s = [r['all_main_f1'] for r in results]
    all_main_precs = [r['all_main_prec'] for r in results]
    all_main_recalls = [r['all_main_recall'] for r in results]
    avg_all_main = sum(all_main_f1s) / n
    avg_all_main_p = sum(all_main_precs) / n
    avg_all_main_r = sum(all_main_recalls) / n

    print(f'\n  Average All-Main ROUGE-5 F1:        {avg_all_main:.4f}')
    print(f'  Average All-Main ROUGE-5 Precision: {avg_all_main_p:.4f}')
    print(f'  Average All-Main ROUGE-5 Recall:    {avg_all_main_r:.4f}')

    print(f'\n  All-main by difficulty:')
    for level in ['simple', 'mid', 'hard']:
        level_results = [r for r in results if r['level'] == level]
        if level_results:
            avg_f1 = sum(r['all_main_f1'] for r in level_results) / len(level_results)
            avg_p = sum(r['all_main_prec'] for r in level_results) / len(level_results)
            avg_r = sum(r['all_main_recall'] for r in level_results) / len(level_results)
            print(f'    {level:>8}: F1={avg_f1:.4f}  P={avg_p:.4f}  R={avg_r:.4f}  (n={len(level_results)})')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 6: Summary comparison
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'SUMMARY COMPARISON')
    print(f'{"=" * 90}')

    print(f'\n  {"Method":<35} {"R5-F1":>8} {"Prec":>8} {"Recall":>8}')
    print(f'  {"-" * 59}')

    comparisons = [
        ('Dripper 0.6B (reported)', 0.878, None, None),
        ('Oracle (perfect labels)', avg_oracle, avg_oracle_p, avg_oracle_r),
        ('Our model (reported)', 0.687, None, None),
        ('All-main baseline', avg_all_main, avg_all_main_p, avg_all_main_r),
    ]
    for name, f1, p, r in comparisons:
        p_str = f'{p:.4f}' if p is not None else '   -   '
        r_str = f'{r:.4f}' if r is not None else '   -   '
        print(f'  {name:<35} {f1:>8.4f} {p_str:>8} {r_str:>8}')

    # ════════════════════════════════════════════════════════════════════
    #  SECTION 7: Diagnosis
    # ════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 90}')
    print(f'DIAGNOSIS')
    print(f'{"=" * 90}')

    # Block statistics
    total_blocks = sum(r['n_blocks'] for r in results)
    total_main = sum(r['n_oracle_main'] for r in results)
    main_frac = total_main / max(total_blocks, 1)
    print(f'\n  Block statistics:')
    print(f'    Total blocks:          {total_blocks}')
    print(f'    Total oracle-main:     {total_main} ({main_frac * 100:.1f}%)')
    print(f'    Avg blocks/page:       {total_blocks / n:.1f}')
    print(f'    Avg main blocks/page:  {total_main / n:.1f}')

    # Key findings
    print(f'\n  Key findings:')

    if avg_oracle >= 0.95:
        print(f'    [1] Pipeline architecture is SOUND. Oracle ceiling = {avg_oracle:.4f}')
        print(f'        The gap to Dripper ({0.878 - avg_oracle:+.3f}) is achievable with better classification.')
    else:
        pipeline_gap = 0.878 - avg_oracle
        print(f'    [1] Pipeline has a CEILING PROBLEM. Oracle = {avg_oracle:.4f}')
        print(f'        Even perfect classification cannot reach Dripper ({pipeline_gap:.3f} gap).')

    if avg_cov >= 0.95:
        print(f'    [2] simplify_html preserves content well. Coverage = {avg_cov * 100:.1f}%')
    else:
        print(f'    [2] simplify_html is LOSING CONTENT. Coverage = {avg_cov * 100:.1f}%')

    model_vs_baseline = 0.687 - avg_all_main
    print(f'    [3] All-main baseline = {avg_all_main:.4f}.')
    if model_vs_baseline < 0:
        print(f'        Model at 0.687 is {model_vs_baseline:+.3f} BELOW all-main baseline!')
        print(f'        The model is ACTIVELY HURTING by misclassifying main blocks as other.')
        print(f'        Check: is the model dropping too many true-main blocks?')
    elif model_vs_baseline < 0.05:
        print(f'        Model at 0.687 only {model_vs_baseline:+.3f} above all-main.')
        print(f'        The model is barely filtering anything useful.')
    else:
        print(f'        Model at 0.687 is {model_vs_baseline:+.3f} above all-main. Meaningful gain.')

    classification_gap = avg_oracle - 0.687
    print(f'    [4] Recoverable gap with better classification: {classification_gap:.3f} F1 points')
    print(f'        (from 0.687 model -> {avg_oracle:.3f} oracle)')

    # Precision vs recall of all-main
    if avg_all_main_p < avg_all_main_r:
        print(f'    [5] All-main has low precision ({avg_all_main_p:.3f}) but high recall ({avg_all_main_r:.3f}).')
        print(f'        The model needs to REMOVE boilerplate (improve precision) without')
        print(f'        losing main content (maintain recall).')
    else:
        print(f'    [5] All-main precision ({avg_all_main_p:.3f}) vs recall ({avg_all_main_r:.3f}).')

    if errors:
        print(f'    [6] {len(errors)} pages had simplify_html errors.')

    print(f'\n{"=" * 90}')
    print(f'Finished. Processed {n} pages.', flush=True)


if __name__ == '__main__':
    main()
