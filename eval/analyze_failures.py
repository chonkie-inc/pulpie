"""Analyze the dirty pages from the 50-sample eval to find failure patterns."""

import json
import os
import subprocess
import tempfile

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_sample50_results.json")
HBIRD_BIN = os.path.join(os.path.dirname(DATA_DIR), "target", "release", "pulpie")

import random
random.seed(42)

with open(BENCH_PATH) as f:
    lines = f.readlines()
indices = random.sample(range(len(lines)), 50)
pages = {json.loads(lines[i])["url"]: json.loads(lines[i]) for i in indices}

with open(RESULTS_PATH) as f:
    results = json.load(f)

dirty = [r for r in results if r["label"] == "dirty"]

for r in dirty:
    url = r["url"]
    page = pages[url]
    html = page["html"]

    # Extract
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        tmp = f.name
    result = subprocess.run([HBIRD_BIN, tmp], capture_output=True, text=True, timeout=30)
    md = result.stdout
    os.unlink(tmp)

    # Count HTML features
    html_lower = html.lower()
    nav_count = html_lower.count("<nav")
    footer_count = html_lower.count("<footer")
    sidebar_count = html_lower.count("sidebar")
    comment_count = html_lower.count("comment")

    print(f"=== {url[:80]} ===")
    print(f"  P(clean)={r['clean_prob']:.3f}  chars={r['md_chars']}  level={r['level']}")
    print(f"  HTML: nav={nav_count} footer={footer_count} sidebar={sidebar_count} comment={comment_count}")
    print(f"  --- First 300 chars ---")
    print(f"  {md[:300]}")
    print(f"  --- Last 300 chars ---")
    print(f"  {md[-300:]}")
    print()
