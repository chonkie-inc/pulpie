"""Inspect eval pages: compare pulpie output vs ground truth.

For each of the 50 eval pages:
- Extract ground truth text from cc-select annotations
- Extract pulpie output
- Compare to find: missing content, leaked boilerplate
"""

import json
import os
import random
import re
import subprocess
import tempfile

from bs4 import BeautifulSoup, Tag

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")
HBIRD_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "pulpie")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_sample50_results.json")

SEED = 42
N_SAMPLES = 50

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th", "figcaption", "dt", "dd",
}


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def get_ground_truth_blocks(html):
    """Extract blocks that are marked cc-select='true' (ground truth content)."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup

    gt_blocks = []
    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        if el.get("cc-select") == "true":
            text = el.get_text().strip()
            if len(text) >= 5:
                gt_blocks.append(text)

    return gt_blocks


def get_all_text_blocks(html):
    """Get all text blocks from HTML (both content and boilerplate)."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup

    blocks = []
    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        if el.name in BLOCK_TAGS:
            text = el.get_text().strip()
            if len(text) >= 5:
                is_content = False
                # Check if this element or any ancestor has cc-select
                node = el
                while node:
                    if isinstance(node, Tag) and node.get("cc-select") == "true":
                        is_content = True
                        break
                    node = node.parent
                blocks.append({"text": text, "is_content": is_content})

    return blocks


def extract_with_pulpie(html_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout
    except Exception:
        return ""
    finally:
        os.unlink(tmp_path)


def find_missing_content(gt_blocks, pulpie_md):
    """Find ground truth content blocks not present in pulpie output."""
    md_norm = normalize(pulpie_md)
    missing = []
    found = []
    for block_text in gt_blocks:
        # Check if a significant chunk of the block text appears in the markdown
        norm_block = normalize(block_text)
        # Use first 80 chars as match key (markdown formatting may alter text)
        key = norm_block[:80] if len(norm_block) > 80 else norm_block
        if key in md_norm:
            found.append(block_text)
        else:
            # Try shorter match
            key_short = norm_block[:40] if len(norm_block) > 40 else norm_block
            if key_short in md_norm:
                found.append(block_text)
            else:
                missing.append(block_text)
    return missing, found


def find_boilerplate_in_output(all_blocks, pulpie_md):
    """Find boilerplate blocks that ended up in pulpie output."""
    md_norm = normalize(pulpie_md)
    leaked = []
    for block in all_blocks:
        if block["is_content"]:
            continue
        norm = normalize(block["text"])
        key = norm[:60] if len(norm) > 60 else norm
        if len(key) >= 10 and key in md_norm:
            leaked.append(block["text"])
    return leaked


def main():
    # Load eval results
    with open(RESULTS_PATH) as f:
        eval_results = json.load(f)

    # Load same pages
    random.seed(SEED)
    with open(BENCH_PATH) as f:
        lines = f.readlines()
    indices = random.sample(range(len(lines)), N_SAMPLES)
    pages = [json.loads(lines[i]) for i in indices]

    # Separate by label
    dirty_pages = []
    clean_pages = []
    empty_pages = []
    for i, (page, result) in enumerate(zip(pages, eval_results)):
        entry = {"idx": i, "page": page, "result": result}
        if result["label"] == "dirty":
            dirty_pages.append(entry)
        elif result["label"] == "empty":
            empty_pages.append(entry)
        else:
            clean_pages.append(entry)

    # =============================================
    # DIRTY PAGES - Where did the model go wrong?
    # =============================================
    print("=" * 100)
    print(f"DIRTY PAGES ({len(dirty_pages)} pages) - Analyzing failures")
    print("=" * 100)

    for entry in dirty_pages:
        page = entry["page"]
        result = entry["result"]
        html = page["html"]
        url = result["url"]

        print(f"\n{'─' * 100}")
        print(f"Page #{entry['idx']+1}: {url}")
        print(f"  P(clean)={result['clean_prob']:.3f}  md_chars={result['md_chars']}  level={result['level']}")

        # Get ground truth
        gt_blocks = get_ground_truth_blocks(html)
        all_blocks = get_all_text_blocks(html)
        n_boilerplate = sum(1 for b in all_blocks if not b["is_content"])

        # Get pulpie output
        md = extract_with_pulpie(html)

        print(f"  Ground truth: {len(gt_blocks)} content blocks, {n_boilerplate} boilerplate blocks")
        print(f"  Pulpie output: {len(md.strip())} chars")

        # Check for missing content
        if gt_blocks:
            missing, found = find_missing_content(gt_blocks, md)
            print(f"  Content found: {len(found)}/{len(gt_blocks)} blocks")
            if missing:
                print(f"  MISSING CONTENT ({len(missing)} blocks):")
                for m in missing[:5]:
                    print(f"    - {m[:120]}...")
        else:
            print("  (No cc-select ground truth blocks found)")

        # Check for leaked boilerplate
        leaked = find_boilerplate_in_output(all_blocks, md)
        if leaked:
            print(f"  LEAKED BOILERPLATE ({len(leaked)} blocks):")
            for l in leaked[:5]:
                print(f"    - {l[:120]}...")

        # Show first/last of hummingbird output
        md_stripped = md.strip()
        if md_stripped:
            print(f"\n  --- First 300 chars of output ---")
            print(f"  {md_stripped[:300]}")
            print(f"\n  --- Last 300 chars of output ---")
            print(f"  {md_stripped[-300:]}")

    # =============================================
    # CLEAN PAGES - Was main content preserved?
    # =============================================
    print("\n\n" + "=" * 100)
    print(f"CLEAN PAGES ({len(clean_pages)} pages) - Checking content preservation")
    print("=" * 100)

    pages_with_missing = []
    pages_with_boilerplate = []

    for entry in clean_pages:
        page = entry["page"]
        result = entry["result"]
        html = page["html"]
        url = result["url"]

        gt_blocks = get_ground_truth_blocks(html)
        all_blocks = get_all_text_blocks(html)

        if not gt_blocks:
            continue

        md = extract_with_pulpie(html)

        missing, found = find_missing_content(gt_blocks, md)
        leaked = find_boilerplate_in_output(all_blocks, md)

        coverage = len(found) / max(len(gt_blocks), 1)
        n_boilerplate = sum(1 for b in all_blocks if not b["is_content"])
        leak_rate = len(leaked) / max(n_boilerplate, 1)

        if len(missing) > 0:
            pages_with_missing.append({
                "url": url, "idx": entry["idx"],
                "coverage": coverage,
                "missing": missing,
                "found": len(found),
                "total_gt": len(gt_blocks),
                "clean_prob": result["clean_prob"],
            })

        if len(leaked) > 2:
            pages_with_boilerplate.append({
                "url": url, "idx": entry["idx"],
                "leaked": leaked,
                "leak_count": len(leaked),
                "total_boilerplate": n_boilerplate,
                "clean_prob": result["clean_prob"],
            })

    # Report pages with missing content
    print(f"\n{'─' * 100}")
    print(f"CLEAN PAGES WITH MISSING CONTENT: {len(pages_with_missing)}/{len(clean_pages)}")
    print(f"{'─' * 100}")

    # Sort by most missing
    pages_with_missing.sort(key=lambda x: len(x["missing"]), reverse=True)
    for p in pages_with_missing[:15]:
        print(f"\n  Page #{p['idx']+1}: {p['url'][:80]}")
        print(f"    P(clean)={p['clean_prob']:.3f}  Coverage: {p['found']}/{p['total_gt']} ({p['coverage']*100:.0f}%)")
        print(f"    Missing {len(p['missing'])} blocks:")
        for m in p["missing"][:3]:
            print(f"      - {m[:150]}")

    # Report pages with leaked boilerplate
    print(f"\n{'─' * 100}")
    print(f"CLEAN PAGES WITH LEAKED BOILERPLATE (>2): {len(pages_with_boilerplate)}/{len(clean_pages)}")
    print(f"{'─' * 100}")

    pages_with_boilerplate.sort(key=lambda x: x["leak_count"], reverse=True)
    for p in pages_with_boilerplate[:10]:
        print(f"\n  Page #{p['idx']+1}: {p['url'][:80]}")
        print(f"    P(clean)={p['clean_prob']:.3f}  Leaked: {p['leak_count']}/{p['total_boilerplate']} boilerplate blocks")
        for l in p["leaked"][:3]:
            print(f"      - {l[:150]}")

    # =============================================
    # EMPTY PAGES
    # =============================================
    print(f"\n\n{'=' * 100}")
    print(f"EMPTY PAGES ({len(empty_pages)} pages) - No output at all")
    print(f"{'=' * 100}")
    for entry in empty_pages:
        page = entry["page"]
        result = entry["result"]
        html = page["html"]
        gt_blocks = get_ground_truth_blocks(html)
        all_blocks = get_all_text_blocks(html)
        print(f"\n  Page #{entry['idx']+1}: {result['url'][:80]}")
        print(f"    GT content blocks: {len(gt_blocks)}, Total blocks: {len(all_blocks)}, HTML size: {len(html)}")
        if gt_blocks:
            print(f"    First GT block: {gt_blocks[0][:150]}")

    # =============================================
    # SUMMARY
    # =============================================
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    total_clean_with_gt = sum(1 for e in clean_pages
                              if len(get_ground_truth_blocks(e["page"]["html"])) > 0)
    print(f"Clean pages: {len(clean_pages)} total")
    print(f"  With missing content: {len(pages_with_missing)}")
    print(f"  With >2 leaked boilerplate: {len(pages_with_boilerplate)}")
    print(f"Dirty pages: {len(dirty_pages)}")
    print(f"Empty pages: {len(empty_pages)}")


if __name__ == "__main__":
    main()
