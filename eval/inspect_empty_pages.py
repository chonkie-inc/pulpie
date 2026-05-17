"""Investigate why 7 eval pages produce empty output from pulpie."""

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
EXPORT_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "export_features")

SEED = 42
N_SAMPLES = 50

EMPTY_INDICES = [12, 15, 19, 24, 27, 42, 47]  # 0-indexed positions in the 50-sample list


def run_pulpie(html, verbose=False):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [HBIRD_BIN, tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        if verbose:
            print(f"    stdout: {len(result.stdout)} chars")
            print(f"    stderr: {result.stderr[:500] if result.stderr else '(none)'}")
            print(f"    returncode: {result.returncode}")
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)
    finally:
        os.unlink(tmp_path)


def run_export_features(html):
    """Run the export_features binary to see what blocks/features are extracted."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [EXPORT_BIN], input=html,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None, result.stderr
        return json.loads(result.stdout), None
    except Exception as e:
        return None, str(e)
    finally:
        os.unlink(tmp_path)


def analyze_html_structure(html):
    """Get basic HTML structure stats."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")

    stats = {
        "html_len": len(html),
        "has_body": body is not None,
        "tag_counts": {},
    }

    root = body or soup
    for tag in root.find_all(True):
        name = tag.name
        stats["tag_counts"][name] = stats["tag_counts"].get(name, 0) + 1

    # Check for scripts/styles that might dominate
    scripts = sum(len(s.get_text()) for s in soup.find_all("script"))
    styles = sum(len(s.get_text()) for s in soup.find_all("style"))
    total_text = len(root.get_text())

    stats["script_chars"] = scripts
    stats["style_chars"] = styles
    stats["total_text_chars"] = total_text

    # Check for frames/iframes
    stats["iframes"] = len(soup.find_all("iframe"))
    stats["frames"] = len(soup.find_all("frame"))

    return stats


def get_ground_truth_blocks(html):
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup
    gt_blocks = []
    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        if el.get("cc-select") == "true":
            text = el.get_text().strip()
            if len(text) >= 5:
                gt_blocks.append({"tag": el.name, "text": text[:200]})
    return gt_blocks


def main():
    random.seed(SEED)
    with open(BENCH_PATH) as f:
        lines = f.readlines()
    indices = random.sample(range(len(lines)), N_SAMPLES)
    pages = [json.loads(lines[i]) for i in indices]

    for empty_idx in EMPTY_INDICES:
        page = pages[empty_idx]
        url = page["url"]
        html = page["html"]

        print("=" * 100)
        print(f"Page #{empty_idx+1}: {url}")
        print("=" * 100)

        # 1. HTML structure
        stats = analyze_html_structure(html)
        print(f"\n  HTML structure:")
        print(f"    HTML size: {stats['html_len']:,} chars")
        print(f"    Has <body>: {stats['has_body']}")
        print(f"    Total text: {stats['total_text_chars']:,} chars")
        print(f"    Script chars: {stats['script_chars']:,}")
        print(f"    Style chars: {stats['style_chars']:,}")
        print(f"    iframes: {stats['iframes']}, frames: {stats['frames']}")

        # Top tags
        top_tags = sorted(stats["tag_counts"].items(), key=lambda x: -x[1])[:15]
        print(f"    Top tags: {', '.join(f'{t}({c})' for t, c in top_tags)}")

        # 2. Ground truth
        gt = get_ground_truth_blocks(html)
        print(f"\n  Ground truth: {len(gt)} content blocks")
        if gt:
            print(f"    First 3:")
            for b in gt[:3]:
                print(f"      <{b['tag']}> {b['text'][:120]}")
            print(f"    Last 3:")
            for b in gt[-3:]:
                print(f"      <{b['tag']}> {b['text'][:120]}")

        # 3. Run export_features to see what blocks get segmented
        blocks, err = run_export_features(html)
        if blocks is None:
            print(f"\n  export_features failed: {err[:300]}")
        else:
            print(f"\n  export_features: {len(blocks)} blocks extracted")
            if blocks:
                # Show feature distributions
                text_lens = [b["features"]["text_len"] for b in blocks]
                link_ratios = [b["features"]["link_ratio"] for b in blocks]
                positions = [b["features"]["position"] for b in blocks]
                print(f"    text_len: min={min(text_lens):.0f} max={max(text_lens):.0f} median={sorted(text_lens)[len(text_lens)//2]:.0f}")
                print(f"    link_ratio: min={min(link_ratios):.2f} max={max(link_ratios):.2f} mean={sum(link_ratios)/len(link_ratios):.2f}")

                # Count how many blocks have high link ratio
                high_lr = sum(1 for lr in link_ratios if lr > 0.5)
                short = sum(1 for tl in text_lens if tl < 20)
                print(f"    High link_ratio (>0.5): {high_lr}/{len(blocks)}")
                print(f"    Short text (<20 chars): {short}/{len(blocks)}")

                # Show first/last few blocks
                print(f"\n    First 3 blocks:")
                for b in blocks[:3]:
                    f = b["features"]
                    print(f"      text_len={f['text_len']} lr={f['link_ratio']} tag={f['tag_type']} depth={f['dom_depth']} | {b['text'][:100]}")
                print(f"    Last 3 blocks:")
                for b in blocks[-3:]:
                    f = b["features"]
                    print(f"      text_len={f['text_len']} lr={f['link_ratio']} tag={f['tag_type']} depth={f['dom_depth']} | {b['text'][:100]}")

        # 4. Run pulpie with verbose
        print(f"\n  Pulpie output:")
        stdout, stderr = run_pulpie(html, verbose=True)
        if stdout.strip():
            print(f"    Output preview: {stdout.strip()[:200]}")
        else:
            print(f"    (empty output)")

        # 5. Check if the HTML is mostly tables (common for empty extraction)
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if tables:
            total_table_text = sum(len(t.get_text()) for t in tables)
            body_text = len((soup.find("body") or soup).get_text())
            print(f"\n  Table analysis: {len(tables)} tables, {total_table_text:,} chars in tables vs {body_text:,} total body text")
            if body_text > 0:
                print(f"    Table text ratio: {total_table_text/body_text*100:.0f}%")

        print()


if __name__ == "__main__":
    main()
