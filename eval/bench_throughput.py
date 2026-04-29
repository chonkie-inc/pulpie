"""Benchmark [BLOCK] classifier throughput with batching."""

import json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

# MinerU-HTML loading (same boilerplate)
import importlib.util
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

def _make_module(name):
    mod = type(sys)(name)
    sys.modules[name] = mod
    return mod

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
for cn in ['MinerUHTMLPreprocessError','MinerUHTMLPromptError',
           'MinerUHTMLResponseParseError','MinerUHTMLMapToMainError',
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

simplify_html = sys.modules['mineru_html.process.simplify_html'].simplify_html

BLOCK_TOKEN = "[BLOCK]"
MAX_LENGTH = 32768
MODEL_PATH = os.path.join(DATA_DIR, 'block_classifier_0.6B', 'final')
WMB_PATH = os.path.join(DATA_DIR, 'webmainbench.jsonl')


def insert_block_markers(simplified_html):
    pattern = re.compile(r'(_item_id="(\d+)")')
    item_ids = []
    parts = []
    last_end = 0
    for m in pattern.finditer(simplified_html):
        parts.append(simplified_html[last_end:m.start()])
        parts.append(BLOCK_TOKEN + ' ')
        parts.append(m.group(0))
        last_end = m.end()
        item_ids.append(m.group(2))
    if not item_ids:
        return None, []
    parts.append(simplified_html[last_end:])
    return ''.join(parts), item_ids


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')

    print(f'Loading model...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to(device).eval()

    for m in model.modules():
        if hasattr(m, 'is_causal'):
            m.is_causal = False

    block_token_id = tokenizer.convert_tokens_to_ids(BLOCK_TOKEN)

    # Load and simplify pages
    print(f'Loading and simplifying pages...', flush=True)
    records = []
    with open(WMB_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('meta', {}).get('language') == 'en':
                records.append(rec)
    if args.limit > 0:
        records = records[:args.limit]

    # Pre-simplify and tokenize
    prepared = []
    for rec in records:
        try:
            simplified, _ = simplify_html(rec['html'])
            marked, item_ids = insert_block_markers(simplified)
            if marked:
                toks = tokenizer.encode(marked, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
                prepared.append({'tokens': toks, 'n_blocks': len(item_ids)})
        except:
            pass

    print(f'  {len(prepared)} pages prepared', flush=True)
    lengths = [len(p['tokens']) for p in prepared]
    print(f'  Token lengths: median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}', flush=True)

    # Sort by length for efficient batching
    prepared.sort(key=lambda x: len(x['tokens']))

    # Warmup
    with torch.no_grad():
        dummy = torch.randint(0, 1000, (1, 100), device=device)
        model(input_ids=dummy, attention_mask={'full_attention': None})

    # Benchmark batched inference
    BS = args.batch_size
    total_pages = 0
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for batch_start in range(0, len(prepared), BS):
            batch = prepared[batch_start:batch_start + BS]
            max_len = max(len(p['tokens']) for p in batch)

            # Pad to max length in batch
            input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            for i, p in enumerate(batch):
                input_ids[i, :len(p['tokens'])] = torch.tensor(p['tokens'], dtype=torch.long)

            outputs = model(input_ids=input_ids, attention_mask={'full_attention': None})
            total_pages += len(batch)

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f'\n{"="*60}')
    print(f'THROUGHPUT BENCHMARK (BS={BS}, {total_pages} pages)')
    print(f'{"="*60}')
    print(f'  Total time:  {elapsed:.1f}s')
    print(f'  Throughput:   {total_pages/elapsed:.1f} pages/sec')
    print(f'  Avg latency:  {elapsed/total_pages*1000:.0f} ms/page')
    print(f'  GPU memory:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak')

    # Also benchmark BS=1 for comparison
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        for p in prepared:
            ids = torch.tensor([p['tokens']], dtype=torch.long, device=device)
            model(input_ids=ids, attention_mask={'full_attention': None})
    torch.cuda.synchronize()
    elapsed_bs1 = time.time() - t1
    print(f'\n  BS=1 baseline: {total_pages/elapsed_bs1:.1f} pages/sec ({elapsed_bs1:.1f}s)')
    print(f'  Speedup:       {elapsed_bs1/elapsed:.1f}x')


if __name__ == '__main__':
    main()
