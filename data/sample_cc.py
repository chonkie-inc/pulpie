"""Sample pages from Common Crawl WARC files for training data generation.

Extracts HTML pages, filters by language (English) and size,
samples one page per domain for diversity, and saves as JSONL.
"""

import json
import os
import random
import sys
from urllib.parse import urlparse

import fasttext
from warcio.archiveiterator import ArchiveIterator

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WARC_DIR = os.path.join(DATA_DIR, "cc_raw")
LID_MODEL = os.path.join(DATA_DIR, "lid.176.bin")
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "cc_sampled.jsonl")

# Filtering thresholds
MIN_HTML_BYTES = 2_000       # skip tiny pages
MAX_HTML_BYTES = 500_000     # skip huge pages (slow to process)
MIN_TEXT_LEN = 200           # must have some visible text
LANG_CONFIDENCE = 0.5        # fasttext confidence threshold
TARGET_LANG = "en"
DEFAULT_MAX_PAGES = 200_000  # cap total output


def extract_visible_text(html_bytes):
    """Quick visible text extraction for language ID (no full parse)."""
    import re
    try:
        html = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
    # Strip script/style
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:5000]  # first 5K chars is enough for lang ID


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument('--exclude', type=str, default=None,
                        help='JSONL file of already-sampled pages to exclude (by domain)')
    args = parser.parse_args()

    output_path = args.output
    max_pages = args.max_pages

    lid = fasttext.load_model(LID_MODEL)
    print(f"FastText LID model loaded", flush=True)

    warc_files = sorted(f for f in os.listdir(WARC_DIR) if f.endswith(".warc.gz"))
    print(f"Found {len(warc_files)} WARC files", flush=True)

    # Load excluded domains (already sampled in previous runs)
    excluded_domains = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude) as f:
            for line in f:
                r = json.loads(line)
                if 'domain' in r:
                    excluded_domains.add(r['domain'])
        print(f"  Excluding {len(excluded_domains)} already-sampled domains", flush=True)

    # First pass: collect all candidate pages (one per domain)
    domain_pages = {}  # domain -> (url, html_bytes)
    total_scanned = 0
    total_html = 0
    total_lang_ok = 0

    for wf in warc_files:
        path = os.path.join(WARC_DIR, wf)
        print(f"\nProcessing {wf}...", flush=True)

        with open(path, "rb") as f:
            for record in ArchiveIterator(f):
                if record.rec_type != "response":
                    continue
                total_scanned += 1

                url = record.rec_headers.get_header("WARC-Target-URI") or ""
                ct = record.http_headers.get_header("Content-Type") if record.http_headers else ""
                status = record.http_headers.get_statuscode() if record.http_headers else ""

                if str(status) != "200" or not ct or "text/html" not in ct:
                    continue

                content = record.content_stream().read()
                if len(content) < MIN_HTML_BYTES or len(content) > MAX_HTML_BYTES:
                    continue
                total_html += 1

                # Extract domain, keep one page per domain
                try:
                    domain = urlparse(url).netloc.lower()
                except Exception:
                    continue

                if domain in domain_pages or domain in excluded_domains:
                    continue  # already have this domain

                # Language filter
                text = extract_visible_text(content)
                if len(text) < MIN_TEXT_LEN:
                    continue

                # FastText language ID on first line (no newlines)
                pred = lid.predict(text.replace("\n", " ")[:2000])
                lang = pred[0][0].replace("__label__", "")
                conf = pred[1][0]

                if lang != TARGET_LANG or conf < LANG_CONFIDENCE:
                    continue
                total_lang_ok += 1

                domain_pages[domain] = {
                    "url": url,
                    "html": content.decode("utf-8", errors="replace"),
                    "domain": domain,
                    "html_bytes": len(content),
                    "lang_conf": round(float(conf), 3),
                }

                if total_scanned % 5000 == 0:
                    print(f"  scanned={total_scanned} html={total_html} en={total_lang_ok} domains={len(domain_pages)}", flush=True)

                if len(domain_pages) >= max_pages:
                    break

        if len(domain_pages) >= max_pages:
            break

    print(f"\n{'='*60}")
    print(f"Scan complete:")
    print(f"  Total records scanned: {total_scanned:,}")
    print(f"  HTML pages (size OK):  {total_html:,}")
    print(f"  English pages:         {total_lang_ok:,}")
    print(f"  Unique domains kept:   {len(domain_pages):,}")

    # Shuffle and write
    pages = list(domain_pages.values())
    random.seed(42)
    random.shuffle(pages)

    with open(output_path, "w") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  Saved {len(pages)} pages to {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
