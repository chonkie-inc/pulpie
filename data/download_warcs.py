"""Download WARC files from CC-MAIN-2026-12 for training data expansion.

Selects WARC indices spread across the full crawl for domain diversity.
Skips already-downloaded files.
"""

import os
import subprocess
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WARC_DIR = os.path.join(DATA_DIR, "cc_raw")
CC_BASE = "https://data.commoncrawl.org/"

# Already have: 0, 500, 800
# Pick 20 more indices spread across 0-99999 for diversity
# Avoid clustering — spread evenly with some jitter
NEW_INDICES = [
    100, 200, 300, 400, 600, 700, 900,
    1500, 2500, 3500, 5000, 7500,
    10000, 15000, 20000, 30000, 40000,
    50000, 60000, 80000,
]

# Fetch the WARC paths list
print("Fetching WARC paths list...", flush=True)
result = subprocess.run(
    ["bash", "-c", f'curl -s "{CC_BASE}crawl-data/CC-MAIN-2026-12/warc.paths.gz" | zcat'],
    capture_output=True, text=True, timeout=60,
)
all_paths = result.stdout.strip().split("\n")
print(f"  {len(all_paths)} WARC files available", flush=True)

# Check what we already have
existing = set()
for f in os.listdir(WARC_DIR):
    if f.endswith(".warc.gz"):
        # Extract index from filename like warc_00000.warc.gz
        idx_str = f.replace("warc_", "").replace(".warc.gz", "")
        try:
            existing.add(int(idx_str))
        except ValueError:
            pass
print(f"  Already have: {sorted(existing)}", flush=True)

# Filter to new indices
to_download = [i for i in NEW_INDICES if i not in existing]
print(f"  Will download: {len(to_download)} WARCs at indices {to_download}", flush=True)

# Download
for idx in to_download:
    if idx >= len(all_paths):
        print(f"  Index {idx} out of range, skipping", flush=True)
        continue

    warc_path = all_paths[idx]
    url = CC_BASE + warc_path
    out_file = os.path.join(WARC_DIR, f"warc_{idx:05d}.warc.gz")

    print(f"\n  Downloading index {idx}: {warc_path.split('/')[-1]}", flush=True)
    print(f"    → {out_file}", flush=True)

    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", out_file, url],
        timeout=600,
    )
    if result.returncode != 0:
        print(f"    FAILED (exit code {result.returncode})", flush=True)
        if os.path.exists(out_file):
            os.remove(out_file)
    else:
        size_mb = os.path.getsize(out_file) / 1024 / 1024
        print(f"    OK ({size_mb:.0f} MB)", flush=True)

# Summary
final_files = [f for f in os.listdir(WARC_DIR) if f.endswith(".warc.gz")]
total_gb = sum(os.path.getsize(os.path.join(WARC_DIR, f)) for f in final_files) / 1e9
print(f"\n{'='*60}")
print(f"Done: {len(final_files)} WARC files, {total_gb:.1f} GB total")
