"""Evaluate Qwen3.5-27B as a Dripper-style block classifier on WebMainBench.

Pipeline (matches Dripper paper):
  1. Raw HTML → Simplified HTML (strip attrs, add _item_id to blocks)
  2. LLM classifies each _item_id as "main" or "other"
  3. Reconstruct content HTML from "main" blocks
  4. html2text canonicalization → ROUGE-5 F1 scoring

Uses async batched requests to saturate the GPU.

Usage:
  python eval/bench_qwen_classify.py [--pages N] [--sanity] [--concurrency N]
"""

import asyncio
import json
import os
import re
import time
from collections import Counter

import html2text
from lxml import etree
from lxml.html import fromstring, tostring
from openai import AsyncOpenAI

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_PATH = os.path.join(os.path.dirname(DATA_DIR), "data", "webmainbench.jsonl")

VLLM_BASE = "http://localhost:8234/v1"
MODEL = "Qwen/Qwen3.5-27B"

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "td", "th",
    "figcaption", "dt", "dd",
}

STRIP_TAGS = {
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "footer", "aside", "header", "template",
    "select", "textarea", "video", "audio", "embed", "object",
}

KEEP_ATTRS = {"class", "id"}

CLASSIFY_PROMPT = """As a front-end engineering expert in HTML, classify elements with `_item_id` as "keep" (primary content) or "discard" (supplementary/boilerplate).

"keep": article body text, forum posts, Q&A questions/answers, discussion replies.
"discard": navigation, menus, sidebars, footers, breadcrumbs, metadata, ads, social buttons, related content links.

Return a JSON object mapping each _item_id to "keep" or "discard".

HTML:

"""


# ── HTML Simplification ──────────────────────────────────────────────


def simplify_html(raw_html):
    """Build Simplified HTML with _item_id on block elements.
    Returns (simplified_html, id_to_xpath, num_items).
    """
    try:
        doc = fromstring(raw_html)
    except Exception:
        return None, {}, 0

    # Remove STRIP_TAGS
    for tag in STRIP_TAGS:
        for el in list(doc.iter(tag)):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    item_id = 0
    id_to_xpath = {}

    def get_xpath(element):
        parts = []
        while element is not None:
            parent = element.getparent()
            if parent is None:
                parts.append(element.tag)
                break
            siblings = [c for c in parent if c.tag == element.tag]
            if len(siblings) == 1:
                parts.append(element.tag)
            else:
                idx = siblings.index(element) + 1
                parts.append(f"{element.tag}[{idx}]")
            element = parent
        return "/" + "/".join(reversed(parts))

    def is_leaf_container(el):
        for child in el.iter():
            if child is el:
                continue
            if child.tag in BLOCK_TAGS:
                return False
        return True

    def walk_and_tag(el):
        nonlocal item_id
        if not isinstance(el.tag, str):
            return
        tag = el.tag.lower()
        if tag in BLOCK_TAGS:
            item_id += 1
            el.set("_item_id", str(item_id))
            id_to_xpath[item_id] = get_xpath(el)
            text_content = el.text_content() or ""
            if len(text_content) > 500:
                _truncate_text(el, 500)
        elif tag in ("div", "section", "article", "main", "form"):
            if is_leaf_container(el):
                text = (el.text_content() or "").strip()
                if len(text) >= 5:
                    item_id += 1
                    el.set("_item_id", str(item_id))
                    id_to_xpath[item_id] = get_xpath(el)
                    if len(text) > 500:
                        _truncate_text(el, 500)
            else:
                for child in el:
                    walk_and_tag(child)
        else:
            for child in el:
                walk_and_tag(child)

    def _truncate_text(el, max_chars):
        total = [0]
        def _trunc(node):
            if total[0] >= max_chars:
                if node.text:
                    node.text = ""
                if node.tail:
                    node.tail = ""
                for child in list(node):
                    node.remove(child)
                return
            if node.text:
                remaining = max_chars - total[0]
                if len(node.text) > remaining:
                    node.text = node.text[:remaining] + "..."
                    total[0] = max_chars
                    for child in list(node):
                        node.remove(child)
                    return
                total[0] += len(node.text)
            for child in node:
                _trunc(child)
                if total[0] >= max_chars:
                    break
            if node.tail:
                remaining = max_chars - total[0]
                if len(node.tail) > remaining:
                    node.tail = node.tail[:remaining] + "..."
                    total[0] = max_chars
                else:
                    total[0] += len(node.tail)
        _trunc(el)

    # Strip attributes
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        attrs_to_remove = [a for a in el.attrib if a not in KEEP_ATTRS and a != "_item_id"]
        for a in attrs_to_remove:
            del el.attrib[a]

    # Remove comments
    for node in list(doc.iter()):
        if isinstance(node, etree._Comment):
            parent = node.getparent()
            if parent is not None:
                if node.tail:
                    prev = node.getprevious()
                    if prev is not None:
                        prev.tail = (prev.tail or "") + node.tail
                    else:
                        parent.text = (parent.text or "") + node.tail
                parent.remove(node)

    # Walk and tag blocks
    body = doc.find(".//body")
    root = body if body is not None else doc
    walk_and_tag(root)

    # Remove empty containers after tagging
    def remove_empty(el):
        for child in list(el):
            if not isinstance(child.tag, str):
                continue
            remove_empty(child)
        if not isinstance(el.tag, str):
            return
        if el.tag in BLOCK_TAGS or "_item_id" in el.attrib:
            return
        text = (el.text_content() or "").strip()
        if not text and el.getparent() is not None:
            if el.tail:
                prev = el.getprevious()
                if prev is not None:
                    prev.tail = (prev.tail or "") + el.tail
                else:
                    el.getparent().text = (el.getparent().text or "") + el.tail
            el.getparent().remove(el)

    remove_empty(root)

    if item_id == 0:
        return None, {}, 0

    try:
        simplified = tostring(root, encoding="unicode", method="html")
        simplified = re.sub(r'\s+', ' ', simplified)
        simplified = re.sub(r'> <', '><', simplified)
    except Exception:
        return None, {}, 0

    return simplified, id_to_xpath, item_id


# ── Classification parsing ───────────────────────────────────────────


def parse_classification(text):
    """Parse LLM classification JSON response."""
    if not text:
        return {}
    try:
        data = json.loads(text)
        return {int(k): v for k, v in data.items()}
    except (ValueError, json.JSONDecodeError):
        pass
    # Fallback: find JSON in response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return {int(k): v for k, v in data.items()}
    except (ValueError, json.JSONDecodeError):
        pass
    return {}


# ── Reconstruction & scoring ─────────────────────────────────────────


def reconstruct_html(raw_html, id_to_xpath, labels):
    """Reconstruct HTML from blocks classified as 'main'."""
    main_ids = {iid for iid, label in labels.items() if label == "keep"}
    if not main_ids:
        return ""
    try:
        doc = fromstring(raw_html)
    except Exception:
        return ""
    parts = []
    for item_id in sorted(main_ids):
        xpath = id_to_xpath.get(item_id)
        if not xpath:
            continue
        try:
            matches = doc.xpath(xpath)
            if matches:
                parts.append(tostring(matches[0], encoding="unicode", method="html"))
        except Exception:
            continue
    return "\n".join(parts)


def html_to_text(html_str):
    h = html2text.HTML2Text(bodywidth=0)
    h.ignore_links = True
    h.ignore_images = True
    return h.handle(html_str)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n_f1(reference, prediction, n=5):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ref_ngrams = Counter(ngrams(ref_tokens, n))
    pred_ngrams = Counter(ngrams(pred_tokens, n))
    if not ref_ngrams or not pred_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & pred_ngrams).values())
    precision = overlap / max(sum(pred_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Async batch processing ───────────────────────────────────────────


async def classify_one(client, semaphore, simplified_html):
    """Classify a single page's blocks via the LLM, respecting concurrency."""
    prompt = CLASSIFY_PROMPT + simplified_html
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_json": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "enum": ["keep", "discard"]},
                    },
                },
            )
            text = response.choices[0].message.content.strip()
            return parse_classification(text)
        except Exception:
            return {}


async def process_batch(client, semaphore, batch, progress):
    """Process a batch of pre-simplified pages concurrently."""
    async def noop():
        return {}

    tasks = []
    for item in batch:
        if item["simplified"] is None:
            tasks.append(noop())
        else:
            tasks.append(classify_one(client, semaphore, item["simplified"]))

    labels_list = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for item, labels in zip(batch, labels_list):
        if isinstance(labels, Exception) or not labels:
            results.append({"score": 0.0, "level": item["level"], "status": "parse_fail" if item["simplified"] else item.get("status", "empty")})
            continue

        content_html = reconstruct_html(item["raw_html"], item["id_to_xpath"], labels)
        if not content_html:
            results.append({"score": 0.0, "level": item["level"], "status": "empty_recon"})
            continue

        pred_text = html_to_text(content_html).strip()
        if not pred_text:
            results.append({"score": 0.0, "level": item["level"], "status": "empty_h2t"})
            continue

        r5 = rouge_n_f1(item["reference"], pred_text, n=5)
        results.append({"score": r5, "level": item["level"], "status": "ok"})

    progress["done"] += len(results)
    return results


async def run(args):
    client = AsyncOpenAI(base_url=VLLM_BASE, api_key="unused")
    semaphore = asyncio.Semaphore(args.concurrency)

    print("Loading WebMainBench (English-only)...", flush=True)
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    records = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("meta", {}).get("language") == "en":
            records.append(rec)

    limit = min(args.pages, len(records))
    print(f"  {len(records)} English pages, evaluating {limit} with concurrency={args.concurrency}", flush=True)

    # Pre-process all pages (simplification is CPU-bound, do it upfront)
    print("  Simplifying HTML...", flush=True)
    items = []
    overflow_count = 0
    no_blocks_count = 0
    prompt_overhead = len(CLASSIFY_PROMPT) // 4

    for rec in records[:limit]:
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        level = rec.get("meta", {}).get("level", "unknown")

        if not html or not reference:
            items.append({"simplified": None, "raw_html": html, "reference": reference,
                         "level": level, "id_to_xpath": {}, "status": "empty"})
            continue

        simplified, id_to_xpath, num_items = simplify_html(html)
        if simplified is None or num_items == 0:
            no_blocks_count += 1
            items.append({"simplified": None, "raw_html": html, "reference": reference,
                         "level": level, "id_to_xpath": {}, "status": "no_blocks"})
            continue

        est_tokens = len(simplified) // 4 + prompt_overhead
        if est_tokens > 31000:
            overflow_count += 1
            items.append({"simplified": None, "raw_html": html, "reference": reference,
                         "level": level, "id_to_xpath": id_to_xpath, "status": "overflow"})
            continue

        items.append({"simplified": simplified, "raw_html": html, "reference": reference,
                     "level": level, "id_to_xpath": id_to_xpath, "status": "ready"})

    ready_count = sum(1 for it in items if it["status"] == "ready")
    print(f"  Ready: {ready_count}, Overflow: {overflow_count}, No blocks: {no_blocks_count}", flush=True)

    # Dispatch all LLM calls concurrently (semaphore limits in-flight)
    print("  Classifying blocks...", flush=True)
    t0 = time.time()
    progress = {"done": 0}

    # Process in batches for progress reporting
    batch_size = 50
    all_results = []
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        batch_results = await process_batch(client, semaphore, batch, progress)
        all_results.extend(batch_results)

        elapsed = time.time() - t0
        n_done = len(all_results)
        avg_score = sum(r["score"] for r in all_results) / max(n_done, 1)
        rate = n_done / max(elapsed, 0.1)
        eta = (limit - n_done) / max(rate, 0.01)
        ok_count = sum(1 for r in all_results if r["status"] == "ok")
        print(f"  {n_done}/{limit}: avg={avg_score:.4f} ok={ok_count} ({rate:.1f} pg/s, ETA {eta/60:.0f}m)", flush=True)

    elapsed = time.time() - t0

    # Aggregate results
    scores_all = [r["score"] for r in all_results]
    scores_by_level = {}
    status_counts = Counter(r["status"] for r in all_results)
    for r in all_results:
        scores_by_level.setdefault(r["level"], []).append(r["score"])

    n = len(scores_all)
    avg = sum(scores_all) / max(n, 1)

    print(f"\n{'='*70}", flush=True)
    print(f"QWEN3.5-27B CLASSIFY — WebMainBench ROUGE-5 F1 ({n} English pages)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Time: {elapsed/60:.1f} min ({n/max(elapsed,0.1):.1f} pg/s)", flush=True)
    print(f"  Status: {dict(status_counts)}", flush=True)

    print(f"\n  {'':>30} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*62}", flush=True)

    avgs = {}
    for lev in ["simple", "mid", "hard"]:
        vals = scores_by_level.get(lev, [])
        avgs[lev] = sum(vals) / max(len(vals), 1)

    print(f"  {'Qwen3.5-27B (classify)':>30} {avg:>8.4f} {avgs['simple']:>8.4f} {avgs['mid']:>8.4f} {avgs['hard']:>8.4f}", flush=True)

    # Comparison
    print(f"\n  {'='*62}", flush=True)
    print(f"  COMPARISON", flush=True)
    print(f"  {'='*62}", flush=True)
    print(f"  {'Tool':<30} {'All':>8} {'Simple':>8} {'Mid':>8} {'Hard':>8}", flush=True)
    print(f"  {'-'*62}", flush=True)

    comparisons = [
        ("DeepSeek-V3.2 (paper)", 0.9098, 0.9415, 0.9104, 0.8771),
        ("GPT-5 (paper)", 0.9024, 0.9382, 0.9042, 0.8638),
        ("Dripper 0.6B (paper)", 0.8779, 0.9205, 0.8804, 0.8313),
        ("Pulpie (h2t, en)", 0.8059, 0.8843, 0.8060, 0.7330),
        ("magic-html (paper)", 0.7138, 0.7857, 0.7121, 0.6434),
    ]

    qwen = ("** Qwen3.5-27B **", avg, avgs["simple"], avgs["mid"], avgs["hard"])
    all_entries = comparisons + [qwen]
    all_entries.sort(key=lambda x: -x[1])

    for name, r_all, r_s, r_m, r_h in all_entries:
        marker = " <--" if "Qwen" in name else ""
        print(f"  {name:<30} {r_all:>8.4f} {r_s:>8.4f} {r_m:>8.4f} {r_h:>8.4f}{marker}", flush=True)

    print(f"\n  NOTE: Paper numbers are full dataset (incl. non-English).", flush=True)

    out_path = os.path.join(DATA_DIR, "bench_qwen_classify_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "pages": n,
            "concurrency": args.concurrency,
            "rouge5_all": avg,
            "rouge5_simple": avgs["simple"],
            "rouge5_mid": avgs["mid"],
            "rouge5_hard": avgs["hard"],
            "status_counts": dict(status_counts),
            "elapsed_sec": elapsed,
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    if args.sanity:
        # Run synchronous sanity mode for debugging
        import openai
        client = openai.OpenAI(base_url=VLLM_BASE, api_key="unused")
        run_sanity(client, args.pages)
    else:
        asyncio.run(run(args))


def run_sanity(client, max_pages):
    """Synchronous verbose sanity check on first few pages."""
    with open(BENCH_PATH) as f:
        lines = f.readlines()

    records = [json.loads(l) for l in lines if json.loads(l).get("meta", {}).get("language") == "en"]

    for i, rec in enumerate(records[:max_pages]):
        html = rec.get("html", "")
        reference = rec.get("convert_main_content", "")
        print(f"\n{'='*60}\nPage {i}: {rec.get('url', '?')[:80]}\n{'='*60}")

        simplified, id_to_xpath, num_items = simplify_html(html)
        if not simplified:
            print("  SKIP: no blocks")
            continue

        est_tokens = len(simplified) // 4 + len(CLASSIFY_PROMPT) // 4
        if est_tokens > 31000:
            print(f"  SKIP: overflow ({est_tokens} est tokens)")
            continue

        print(f"  {len(simplified)} chars, {num_items} blocks")

        prompt = CLASSIFY_PROMPT + simplified
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1024,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_json": {
                        "type": "object",
                        "additionalProperties": {"type": "string", "enum": ["keep", "discard"]},
                    },
                },
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM error: {e}")
            continue

        labels = parse_classification(text)
        main_count = sum(1 for v in labels.values() if v == "keep")
        print(f"  Labels: {len(labels)} parsed, {main_count} main")

        content_html = reconstruct_html(html, id_to_xpath, labels)
        pred_text = html_to_text(content_html).strip() if content_html else ""
        r5 = rouge_n_f1(reference, pred_text) if pred_text else 0.0
        print(f"  ROUGE-5: {r5:.4f}")
        print(f"  Ref[:150]: {reference[:150]}")
        print(f"  Pred[:150]: {pred_text[:150]}")


if __name__ == "__main__":
    main()
