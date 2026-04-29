"""Sample a small stratified WMB eval set for training-time ROUGE-5 monitoring.

Saves 200 English pages (balanced by difficulty level) to wmb_eval_sample.jsonl.
Each record has: track_id, url, html, convert_main_content, meta.
"""

import json
import os
import random

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WMB_PATH = os.path.join(SCRIPT_DIR, 'webmainbench.jsonl')
OUT_PATH = os.path.join(SCRIPT_DIR, 'wmb_eval_sample.jsonl')

SAMPLE_PER_LEVEL = 67  # ~200 total (67 * 3 = 201)

by_level = {'simple': [], 'mid': [], 'hard': []}

with open(WMB_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('meta', {}).get('language') != 'en':
            continue
        level = rec.get('meta', {}).get('level', 'unknown')
        if level in by_level:
            by_level[level].append(rec)

print(f'English pages: {sum(len(v) for v in by_level.values())}')
for lev, pages in by_level.items():
    print(f'  {lev}: {len(pages)}')

sampled = []
for lev in ['simple', 'mid', 'hard']:
    pages = by_level[lev]
    random.shuffle(pages)
    sampled.extend(pages[:SAMPLE_PER_LEVEL])

random.shuffle(sampled)

with open(OUT_PATH, 'w') as f:
    for rec in sampled:
        # Only keep fields needed for eval
        out = {
            'track_id': rec['track_id'],
            'url': rec['url'],
            'html': rec['html'],
            'convert_main_content': rec['convert_main_content'],
            'meta': rec['meta'],
        }
        f.write(json.dumps(out, ensure_ascii=False) + '\n')

print(f'\nSaved {len(sampled)} pages to {OUT_PATH}')
