"""Train EuroBERT-2.1B block classifier with 64-token block truncation.

Same as train_eurobert_classifier.py but truncates each block to 64 tokens
instead of relying on 500-char cutoff in simplify_html. This matches the
production inference configuration where token-level truncation is used.

simplify_html is called with cutoff_length=100000 (effectively no char truncation)
and tokenize_blocks truncates each block to 64 tokens.

Usage:
  accelerate launch --num_processes 4 data/train_eurobert_classifier_64tok.py --epochs 1
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import html2text
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# ── MinerU-HTML module loading ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MINERU_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'MinerU-HTML')

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
_simplify = _load_file('mineru_html.process.simplify_html', 'simplify_html.py')
_map_main = _load_file('mineru_html.process.map_to_main', 'map_to_main.py')
simplify_html = _simplify.simplify_html
extract_main_html = _map_main.extract_main_html

from block_chunker import extract_blocks, tokenize_blocks, pack_chunks, SEP_TOKEN

# ── Config ──
DEFAULT_MODEL = "EuroBERT/EuroBERT-2.1B"
MAX_TOKENS = 8192
SEED = 42

SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled.jsonl')
LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_filtered.jsonl')
DOUBLECHECK_PATH = os.path.join(SCRIPT_DIR, 'cc_doublecheck_results.jsonl')
WMB_EVAL_PATH = os.path.join(SCRIPT_DIR, 'wmb_eval_sample.jsonl')
DRIPPER_LABELED_PATH = os.path.join(SCRIPT_DIR, 'cc_labeled_dripper_83k.jsonl')
DRIPPER_SAMPLED_PATH = os.path.join(SCRIPT_DIR, 'cc_sampled_100k.jsonl')
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'block_classifier_eurobert_2.1B_64tok')

BLOCK_MAX_TOKENS = 64
SIMPLIFY_CUTOFF = 100000

LABEL_OTHER = 0
LABEL_MAIN = 1
IGNORE_LABEL = -100
NUM_LABELS = 2


def build_chunk_examples(simplified_html, labels_dict, tokenizer, sep_token_id):
    """Build training examples from a single page using block chunking.

    Each page gets split into blocks, packed into 8K chunks with <|sep|> between them.
    Labels are assigned at <|sep|> positions only.

    Returns list of dicts with 'input_ids' and 'labels'.
    """
    blocks = extract_blocks(simplified_html)
    if not blocks:
        return []

    # Extract item_ids from each block to map to labels
    item_id_pattern = re.compile(r'_item_id="(\d+)"')
    block_item_ids = []
    for block in blocks:
        m = item_id_pattern.search(block)
        if m:
            block_item_ids.append(m.group(1))
        else:
            block_item_ids.append(None)

    # Get label for each block
    block_labels = []
    valid_blocks = []
    valid_indices = []
    for i, (block, item_id) in enumerate(zip(blocks, block_item_ids)):
        if item_id is None:
            continue
        label_str = labels_dict.get(item_id)
        if label_str is None:
            continue
        valid_blocks.append(block)
        valid_indices.append(i)
        block_labels.append(LABEL_MAIN if label_str == 'main' else LABEL_OTHER)

    if not valid_blocks:
        return []

    # Tokenize valid blocks with truncation
    block_token_ids = [tokenizer.encode(b, add_special_tokens=False)[:BLOCK_MAX_TOKENS] for b in valid_blocks]

    # Pack into chunks
    chunks = pack_chunks(
        block_token_ids,
        max_tokens=MAX_TOKENS,
        sep_token_id=sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Build training examples from chunks
    examples = []
    global_block_idx = 0

    for chunk_ids, chunk_block_indices in chunks:
        token_labels = [IGNORE_LABEL] * len(chunk_ids)

        # Find <|sep|> positions and assign labels
        sep_count = 0
        for tok_idx, tid in enumerate(chunk_ids):
            if tid == sep_token_id:
                # This sep corresponds to chunk_block_indices[sep_count]
                if sep_count < len(chunk_block_indices):
                    label_idx = chunk_block_indices[sep_count]
                    token_labels[tok_idx] = block_labels[label_idx]
                sep_count += 1

        n_labeled = sum(1 for l in token_labels if l != IGNORE_LABEL)
        if n_labeled == 0:
            continue

        examples.append({
            'input_ids': chunk_ids,
            'labels': token_labels,
            'n_blocks': n_labeled,
            'n_main': sum(1 for l in token_labels if l == LABEL_MAIN),
        })

    return examples


class ChunkDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        return {'input_ids': e['input_ids'], 'labels': e['labels']}


class ChunkCollator:
    """Pads chunks to same length within a batch. Standard attention mask."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(f['input_ids']) + [0] * pad_len)
            labels.append(f['labels'] + [IGNORE_LABEL] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class WeightedCETrainer(Trainer):
    """Trainer with class-weighted CE loss at <|sep|> positions."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=weight,
            ignore_index=IGNORE_LABEL,
        )
        return (loss, outputs) if return_outputs else loss


def load_cc_pages(min_agreement=0.70, limit=0, original_only=False):
    """Load CC pages from original (DeepSeek+Dripper) and optionally new Dripper-only data."""
    labels_by_url = {}

    if os.path.exists(DOUBLECHECK_PATH) and os.path.exists(LABELED_PATH):
        agreement_by_url = {}
        with open(DOUBLECHECK_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'ok':
                    agreement_by_url[r['url']] = r['agreement_rate']

        with open(LABELED_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') != 'ok':
                    continue
                url = r['url']
                agr = agreement_by_url.get(url)
                if agr is None or agr < min_agreement:
                    continue
                labels_by_url[url] = r.get('labels', {})

        print(f'  Original labeled (agreement >= {min_agreement}): {len(labels_by_url)}', flush=True)

    dripper_count = 0
    if not original_only and os.path.exists(DRIPPER_LABELED_PATH):
        with open(DRIPPER_LABELED_PATH) as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') != 'ok':
                    continue
                url = r['url']
                if url not in labels_by_url:
                    labels_by_url[url] = r.get('labels', {})
                    dripper_count += 1
        print(f'  Dripper-only labeled (new): {dripper_count}', flush=True)
    elif original_only:
        print(f'  Skipping Dripper-only data (--original-only)', flush=True)

    print(f'  Total unique labeled URLs: {len(labels_by_url)}', flush=True)

    sampled_paths = [SAMPLED_PATH]
    if not original_only:
        sampled_paths.append(DRIPPER_SAMPLED_PATH)
    pages = []
    for sampled_path in sampled_paths:
        if not os.path.exists(sampled_path):
            continue
        with open(sampled_path) as f:
            for line in f:
                r = json.loads(line)
                if r['url'] in labels_by_url:
                    pages.append({
                        'url': r['url'],
                        'html': r['html'],
                        'labels': labels_by_url.pop(r['url']),
                    })

    print(f'  Pages with HTML + labels: {len(pages)}', flush=True)

    if limit > 0:
        pages = pages[:limit]
    return pages


def prepare_examples(tokenizer, sep_token_id, limit=0, original_only=False):
    """Prepare chunked training examples from CC data."""
    print('Loading CC pages...', flush=True)
    pages = load_cc_pages(limit=limit, original_only=original_only)
    print(f'  {len(pages)} pages loaded', flush=True)

    examples = []
    fail = 0
    pages_with_examples = 0
    total_chunks = 0

    for i, page in enumerate(pages):
        try:
            simplified, _ = simplify_html(page['html'], cutoff_length=SIMPLIFY_CUTOFF)
        except Exception:
            fail += 1
            continue

        chunk_examples = build_chunk_examples(simplified, page['labels'], tokenizer, sep_token_id)
        if not chunk_examples:
            fail += 1
            continue

        examples.extend(chunk_examples)
        pages_with_examples += 1
        total_chunks += len(chunk_examples)

        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{len(pages)}: {len(examples)} chunks from {pages_with_examples} pages, '
                  f'{fail} failed', flush=True)

    print(f'  Done: {len(examples)} chunks from {pages_with_examples} pages, {fail} failed', flush=True)
    print(f'  Avg chunks/page: {total_chunks/max(pages_with_examples,1):.2f}', flush=True)

    total_blocks = sum(e['n_blocks'] for e in examples)
    total_main = sum(e['n_main'] for e in examples)
    total_other = total_blocks - total_main
    print(f'  Blocks: {total_blocks:,} (main={total_main:,} [{total_main/total_blocks*100:.1f}%], '
          f'other={total_other:,} [{total_other/total_blocks*100:.1f}%])', flush=True)
    print(f'  Avg blocks/chunk: {total_blocks/len(examples):.1f}', flush=True)
    print(f'  Avg tokens/chunk: {sum(len(e["input_ids"]) for e in examples)/len(examples):.0f}', flush=True)

    return examples


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    mask = labels != IGNORE_LABEL
    flat_preds = preds[mask]
    flat_labels = labels[mask]

    acc = (flat_preds == flat_labels).mean()
    f1 = f1_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)
    prec = precision_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)
    rec = recall_score(flat_labels, flat_preds, pos_label=LABEL_MAIN, zero_division=0)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
    }


# ── Inference / Eval utilities ──

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


@torch.no_grad()
def classify_page_chunked(model, tokenizer, sep_token_id, simplified_html, device):
    """Classify blocks on a page using chunked inference. Returns {item_id: 'main'/'other'}."""
    blocks = extract_blocks(simplified_html)
    if not blocks:
        return {}

    item_id_pattern = re.compile(r'_item_id="(\d+)"')
    block_item_ids = []
    for block in blocks:
        m = item_id_pattern.search(block)
        block_item_ids.append(m.group(1) if m else None)

    block_token_ids = [tokenizer.encode(b, add_special_tokens=False)[:BLOCK_MAX_TOKENS] for b in blocks]

    chunks = pack_chunks(
        block_token_ids,
        max_tokens=MAX_TOKENS,
        sep_token_id=sep_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    labels = {}
    for chunk_ids, chunk_block_indices in chunks:
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits[0]

        sep_positions = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
        preds = logits[sep_positions].argmax(dim=-1).cpu().tolist()

        for sep_idx, block_idx in enumerate(chunk_block_indices):
            if sep_idx < len(preds) and block_idx < len(block_item_ids):
                item_id = block_item_ids[block_idx]
                if item_id is not None:
                    labels[item_id] = 'main' if preds[sep_idx] == LABEL_MAIN else 'other'

    return labels


def eval_rouge5_on_wmb(model, tokenizer, sep_token_id, device, wmb_records):
    """Run end-to-end ROUGE-5 eval on WMB sample."""
    model.eval()
    scores = []
    scores_by_level = {}
    empty = 0

    for rec in wmb_records:
        html_content = rec.get('html', '')
        reference = rec.get('convert_main_content', '')
        level = rec.get('meta', {}).get('level', 'unknown')

        if not html_content or not reference:
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        try:
            simplified, map_html = simplify_html(html_content, cutoff_length=SIMPLIFY_CUTOFF)
        except Exception:
            scores.append(0.0)
            scores_by_level.setdefault(level, []).append(0.0)
            continue

        labels = classify_page_chunked(model, tokenizer, sep_token_id, simplified, device)

        n_main = sum(1 for v in labels.values() if v == 'main')
        if n_main == 0:
            pred_text = ''
        else:
            try:
                main_html = extract_main_html(map_html, labels)
                pred_text = html_to_text(main_html).strip()
            except Exception:
                pred_text = ''

        if not pred_text:
            empty += 1
            r5 = 0.0
        else:
            r5 = rouge_n_f1(reference, pred_text, n=5)

        scores.append(r5)
        scores_by_level.setdefault(level, []).append(r5)

    avg = sum(scores) / max(len(scores), 1)
    level_avgs = {}
    for lev in ['simple', 'mid', 'hard']:
        vals = scores_by_level.get(lev, [])
        level_avgs[lev] = sum(vals) / max(len(vals), 1)

    return avg, level_avgs, empty


class RougeEvalCallback(TrainerCallback):
    """Runs end-to-end ROUGE-5 on WMB sample after each eval."""

    def __init__(self, tokenizer, sep_token_id, wmb_records):
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        self.wmb_records = wmb_records

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.local_rank > 0:
            return
        if not self.wmb_records:
            return

        device = next(model.parameters()).device
        avg, level_avgs, empty = eval_rouge5_on_wmb(
            model, self.tokenizer, self.sep_token_id, device, self.wmb_records,
        )
        print(f'\n  WMB ROUGE-5 (step {state.global_step}, {len(self.wmb_records)} pages):  '
              f'All={avg:.4f}  Simple={level_avgs["simple"]:.4f}  '
              f'Mid={level_avgs["mid"]:.4f}  Hard={level_avgs["hard"]:.4f}  '
              f'Empty={empty}', flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--val-pages', type=int, default=1000)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True)
    parser.add_argument('--original-only', action='store_true')
    args = parser.parse_args()

    model_name = args.model
    output_dir = BASE_OUTPUT_DIR

    set_seed(SEED)

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_main = local_rank <= 0

    if is_main:
        print(f'Loading tokenizer: {model_name}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add <|sep|> special token
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': [SEP_TOKEN]})
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
    if is_main:
        print(f'  Added {num_added} special tokens. <|sep|> id = {sep_token_id}', flush=True)
        print(f'  Vocab size: {len(tokenizer)}', flush=True)

    # Load WMB eval sample
    wmb_records = []
    if os.path.exists(WMB_EVAL_PATH):
        with open(WMB_EVAL_PATH) as f:
            for line in f:
                wmb_records.append(json.loads(line))
        if is_main:
            print(f'  Loaded {len(wmb_records)} WMB eval pages for ROUGE-5 monitoring', flush=True)

    # Prepare data
    if is_main:
        print('\nPreparing training examples (chunked to 8K)...', flush=True)
    examples = prepare_examples(tokenizer, sep_token_id, limit=args.limit,
                                original_only=args.original_only)

    # Split train/val
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(examples))
    val_size = min(args.val_pages, len(examples) // 10)
    val_idx = set(indices[:val_size])

    train_examples = [e for i, e in enumerate(examples) if i not in val_idx]
    val_examples = [e for i, e in enumerate(examples) if i in val_idx]
    if is_main:
        print(f'\n  Train: {len(train_examples)} chunks', flush=True)
        print(f'  Val:   {len(val_examples)} chunks', flush=True)

    # Compute class weights
    train_main = sum(e['n_main'] for e in train_examples)
    train_other = sum(e['n_blocks'] - e['n_main'] for e in train_examples)
    train_total = train_main + train_other
    w_other = train_total / (2 * train_other) if train_other > 0 else 1.0
    w_main = train_total / (2 * train_main) if train_main > 0 else 1.0
    class_weights = [w_other, w_main]
    if is_main:
        print(f'  Class weights: other={w_other:.3f}, main={w_main:.3f}', flush=True)

    # Load model
    if is_main:
        print(f'\nLoading model: {model_name} (bidirectional encoder)', flush=True)
        print(f'  Output dir: {output_dir}', flush=True)
        print(f'  Gradient checkpointing: {args.gradient_checkpointing}', flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Parameters: {n_params/1e9:.2f}B total, {n_trainable/1e9:.2f}B trainable', flush=True)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy='steps',
        eval_steps=0.25,
        save_strategy='steps',
        save_steps=0.25,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to='none',
        ddp_find_unused_parameters=False,
    )

    callbacks = []
    if wmb_records:
        callbacks.append(RougeEvalCallback(tokenizer, sep_token_id, wmb_records))

    trainer = WeightedCETrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ChunkDataset(train_examples),
        eval_dataset=ChunkDataset(val_examples),
        data_collator=ChunkCollator(pad_token_id),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    if is_main:
        print('\nTraining...', flush=True)
    t0 = time.time()
    trainer.train()
    if is_main:
        print(f'\nTraining took {time.time() - t0:.0f}s', flush=True)

    # Final eval
    if is_main:
        print('\nFinal evaluation:', flush=True)
    metrics = trainer.evaluate()
    if is_main:
        for k, v in sorted(metrics.items()):
            print(f'  {k}: {v}', flush=True)

    # Save
    if is_main:
        trainer.save_model(os.path.join(output_dir, 'final'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'final'))
        print(f'\nModel saved to {output_dir}/final', flush=True)


if __name__ == '__main__':
    main()
