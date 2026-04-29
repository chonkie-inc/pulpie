"""Analyze GBM cascade potential: how much can high-confidence pruning strip
from pages before sending to a transformer?

Uses the CC-trained model scored on WMB data (honest out-of-sample).
"""

import numpy as np
import lightgbm as lgb
import pandas as pd

MODEL_PATH = "data/model_dom_v3_cc.txt"
WMB_DATA = "data/training_data_dom.csv"  # WMB blocks with labels

TOP_FEATURES = [
    "link_ratio", "page_total_blocks", "parent_tag_type", "page_total_link_ratio",
    "tag_type", "section_block_count", "page_total_text_len", "dom_depth",
    "section_link_density", "position", "tag_type_score", "section_heading_text_len",
    "blocks_since_heading", "word_count", "next_block_link_ratio", "prev_block_link_ratio",
    "text_to_tag_ratio", "semantic_ancestor", "blocks_until_heading", "page_heading_count",
    "parent_class_id_score", "next_block_text_len", "link_count", "text_len",
    "tag_count", "avg_word_length", "in_nav", "in_footer", "in_aside", "in_header",
]

def main():
    print("Loading CC-only model (out-of-sample for WMB)...")
    model = lgb.Booster(model_file=MODEL_PATH)

    print(f"Loading WMB block data from {WMB_DATA}...")
    df = pd.read_csv(WMB_DATA)
    print(f"  {len(df)} blocks, {df['label'].sum()} content, {(~df['label'].astype(bool)).sum()} boilerplate\n")

    X = df[TOP_FEATURES].values
    y = df["label"].values
    text_lens = df["text_len"].values

    print("Predicting probabilities...")
    probs = model.predict(X)

    print(f"\nBlock distribution:")
    print(f"  Total blocks: {len(y)}")
    print(f"  Content (label=1): {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Boilerplate (label=0): {(1-y).sum():.0f} ({100*(1-y.mean()):.1f}%)")

    total_text = text_lens.sum()
    content_text = text_lens[y == 1].sum()

    print(f"\n  Total text chars: {total_text:,.0f}")
    print(f"  Content text chars: {content_text:,.0f} ({100*content_text/total_text:.1f}%)")

    # Cascade analysis: discard blocks where P(content) < threshold
    print(f"\n{'='*80}")
    print(f"CASCADE ANALYSIS: GBM pre-filter at various discard thresholds")
    print(f"  'Discard if P(content) < threshold' — high threshold = aggressive pruning")
    print(f"{'='*80}")
    print(f"\n{'Thresh':>8}  {'Discarded':>10}  {'% Blocks':>9}  {'% Text':>8}  {'Content Lost':>13}  {'% Content Lost':>15}  {'Remaining':>10}")
    print(f"{'-'*8}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*13}  {'-'*15}  {'-'*10}")

    for threshold in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        discard_mask = probs < threshold
        keep_mask = ~discard_mask

        n_discarded = discard_mask.sum()
        n_remaining = keep_mask.sum()
        pct_blocks_discarded = 100 * n_discarded / len(y)

        text_discarded = text_lens[discard_mask].sum()
        pct_text_discarded = 100 * text_discarded / total_text

        # How much actual content do we lose?
        content_lost = (y[discard_mask] == 1).sum()
        content_text_lost = text_lens[discard_mask & (y == 1)].sum()
        pct_content_lost = 100 * content_text_lost / content_text if content_text > 0 else 0

        print(f"{threshold:>8.2f}  {n_discarded:>10,}  {pct_blocks_discarded:>8.1f}%  {pct_text_discarded:>7.1f}%  {content_lost:>13,}  {pct_content_lost:>14.2f}%  {n_remaining:>10,}")

    # Token-level analysis: estimate how much sequence length shrinks
    print(f"\n{'='*80}")
    print(f"SEQUENCE LENGTH REDUCTION (chars as proxy for tokens)")
    print(f"{'='*80}")
    print(f"\n  Avg text chars per WMB page (all blocks): {total_text / df['page_total_blocks'].nunique() if 'page_total_blocks' in df.columns else 'N/A'}")

    # Per-page analysis would be better, but we don't have page IDs in this CSV
    # Use position=0.0 as proxy for "first block of a page"
    # Instead, just show aggregate reduction
    for threshold in [0.05, 0.10, 0.20]:
        keep_mask = probs >= threshold
        remaining_text = text_lens[keep_mask].sum()
        reduction = 1 - remaining_text / total_text
        print(f"\n  Threshold {threshold}: {100*reduction:.0f}% text removed → seq lengths ~{1/(1-reduction):.1f}x shorter")

        # Content recall at this threshold
        content_kept = text_lens[keep_mask & (y == 1)].sum()
        recall = content_kept / content_text
        print(f"    Content recall: {100*recall:.2f}%")

    # Show the "sweet spot"
    print(f"\n{'='*80}")
    print(f"SWEET SPOT ANALYSIS")
    print(f"{'='*80}")
    print(f"\n  Goal: >99.5% content recall (lose <0.5% of actual content text)")

    for threshold in np.arange(0.01, 0.50, 0.01):
        keep_mask = probs >= threshold
        content_kept = text_lens[keep_mask & (y == 1)].sum()
        recall = content_kept / content_text
        if recall < 0.995:
            prev_t = threshold - 0.01
            keep_prev = probs >= prev_t
            text_remaining_pct = 100 * text_lens[keep_prev].sum() / total_text
            blocks_remaining_pct = 100 * keep_prev.sum() / len(y)
            print(f"  Best threshold: {prev_t:.2f}")
            print(f"  Blocks remaining: {blocks_remaining_pct:.1f}%")
            print(f"  Text remaining: {text_remaining_pct:.1f}%")
            print(f"  Seq length reduction: ~{100-text_remaining_pct:.0f}%")
            print(f"  Content recall: >99.5%")
            break


if __name__ == "__main__":
    main()
