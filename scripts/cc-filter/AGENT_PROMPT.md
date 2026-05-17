# Agent prompt: bring up a CC filter shard on a fresh box

Paste this as the initial prompt to an agent (or use it as a human runbook).

---

You're setting up a CPU-only pipeline on a fresh GCP VM (target: `c2-standard-16` spot in `us-central1` or `us-east1`, Ubuntu 22.04+). The pipeline streams Common Crawl WARC files, filters to English HTML, and uploads zstd-compressed JSONL to a private HF bucket. The bucket and source script already exist — you're bringing up a new worker to contribute to the shared run.

**Inputs you need from the user before starting:**

1. `HF_TOKEN` with write access to the `chonkie-ai` org
2. A `--start` / `--end` WARC index range to process (full range is 0–100000; ask which shard this box should own so it doesn't overlap with other boxes)
3. A way to get `stream_filter_upload.py` onto the box. It lives in this repo at `scripts/cc-filter/stream_filter_upload.py`. Easiest: `git clone https://github.com/chonkie-inc/hummingbird` and copy the file out, or `gh repo clone chonkie-inc/hummingbird` if `gh` is installed. (The repo is still named "hummingbird" on GitHub.)

**Authoritative setup guide:** `RECREATE_SETUP.md` in this same directory. Follow it exactly. Condensed version:

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3 python3-dev curl

# uv + venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 "$HOME/.venv"

# Deps (pinned — DO NOT install `fasttext`, it's broken on numpy 2; use fasttext-numpy2)
uv pip install --python "$HOME/.venv/bin/python" \
    "huggingface_hub==1.12.2" \
    "warcio==1.8.1" \
    "zstandard==0.25.0" \
    "fasttext-numpy2==0.10.4" \
    "numpy==2.2.6" \
    "requests==2.33.1"

# Workspace + fasttext LID model + WARC path listing
mkdir -p "$HOME/workspace/cc_data/staging"
cd "$HOME/workspace/cc_data"
curl -sL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
    -o lid.176.bin
curl -sL https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/warc.paths.gz \
    -o warc.paths.gz && gunzip -k warc.paths.gz

# Place stream_filter_upload.py at $HOME/workspace/cc_data/stream_filter_upload.py
# (copy from this repo: scripts/cc-filter/stream_filter_upload.py)
```

**Before launching the full run, smoke-test on 3 WARCs** in the user's assigned range:

```bash
export HF_TOKEN=<from user>
"$HOME/.venv/bin/python" stream_filter_upload.py \
    --start <assigned_start> --end <assigned_start+3> --workers 16
```

Expect: ~95 s wall time, each WARC produces ~48 MB zst, ~2900 English pages each, all 3 appear in the bucket at `warc/warc_{idx:05d}.jsonl.zst`. If it works, launch the real shard:

```bash
nohup env HF_TOKEN="$HF_TOKEN" \
    "$HOME/.venv/bin/python" stream_filter_upload.py \
    --start <assigned_start> --end <assigned_end> --workers 16 \
    > run.log 2>&1 &
echo $! > run.pid
disown
```

**Monitor**: `grep "^\[" run.log | tail -1` shows the latest progress line with `pps` and `eta=`. Expect ~500 pages/sec sustained.

**Known gotchas** (flag these if you hit them; don't blindly work around):

- `fasttext==0.9.3` breaks on numpy 2 (`np.array(..., copy=False)` ValueError). Install `fasttext-numpy2` instead.
- Do NOT install `transformers` here (unrelated to filter, and its version requirements conflict with `huggingface_hub>=1.0`).
- `list_bucket_tree` takes `prefix=`, not `path=` (a past bug we already fixed in the checked-in script).
- Spot preemption is safe — the script lists the bucket on startup and skips already-uploaded WARCs.

**When done**: report back the shard's page count, output GB, wall time, and any WARC indices that errored (`ok=False` lines in `run.log`).
