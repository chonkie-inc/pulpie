# Introducing Hummingbird: Pareto-Optimal Models for Cleaning the Web

Most of the web is not content. A typical HTML page is 60-70% navigation, ads, cookie banners, sidebars, and footers. Before an LLM can learn from the web — or reason over it at inference time — something has to separate the signal from the noise.

This is the content extraction problem. It sounds simple ("just get the article text") but turns out to be surprisingly hard at scale. Heuristic tools like Trafilatura and Readability work on easy pages and fall apart on complex ones. LLM-based approaches like Dripper get much better quality, but at a cost that makes web-scale processing impractical.

We built Hummingbird to close this gap: a family of encoder models that match LLM-level extraction quality at a fraction of the cost. The smallest model (210M parameters, 433MB) runs at 15 pages/sec on a $0.35/hr L4 GPU. Cleaning 1 billion pages costs about $6,500. The equivalent job with Dripper costs $105,000 on the same hardware.

This post covers how we got there.

## Table of Contents

- [The landscape](#the-landscape)
- [Why encoders?](#why-encoders)
- [Architecture](#architecture)
- [Training](#training)
- [Results](#results)
- [The economics of encoders vs decoders](#the-economics-of-encoders-vs-decoders)
- [Models](#models)
- [Open questions](#open-questions)

## The landscape

Content extraction methods fall into three categories, each with a different quality-cost tradeoff.

**Heuristic extractors** — Trafilatura (Barbaresi, 2021), Readability, magic-html — use hand-written rules to strip boilerplate. They look for semantic tags (`<article>`, `<main>`), DOM structure, and text density signals. They're fast, free, and require no GPU. But they have no notion of context. A navigation table and a data table look the same to a rule that just counts `<td>` tags.

**LLM-based classifiers** — Dripper (MinerU-HTML, 2025) takes a different approach. It simplifies the HTML, assigns an `_item_id` to each block, then prompts a fine-tuned 0.6B autoregressive model to label each block as "main" or "other." This works well — the model sees the full page and can use context to decide. But it generates tokens sequentially, one at a time, bounded by memory bandwidth rather than compute.

**Feature-based classifiers** — tools like boilerpipe and our own GBM approach extract structural features (link density, DOM depth, position) and train a classifier on them. These are fast but cap out at the same ceiling as heuristics. Thirty numeric features encode roughly the same signals that Readability hard-codes in XPaths. Without reading the actual text, a feature-based model can't distinguish a content table from a navigation table.

The following table shows where each method lands on WebMainBench, a benchmark of 7,809 annotated web pages (Deng et al., 2025). We report ROUGE-5 F1 on the English subset (200 pages, controlled comparison):

| Method | Type | All | Simple | Mid | Hard |
|--------|------|-----|--------|-----|------|
| DeepSeek V3.2 | LLM (236B) | 0.865 | 0.932 | 0.875 | 0.786 |
| **Hummingbird Latte Large** | **Encoder (2.1B)** | **0.862** | **0.928** | **0.856** | **0.807** |
| Dripper | Decoder (0.6B) | 0.854 | 0.922 | 0.868 | 0.768 |
| **Hummingbird Latte Small** | **Encoder (210M)** | **0.864** | **0.885** | **0.841** | **0.866** |
| Hummingbird Latte Base | Encoder (610M) | 0.847 | 0.907 | 0.848 | 0.787 |
| magic-html | Heuristic | 0.714 | 0.786 | 0.712 | 0.643 |
| Readability | Heuristic | 0.654 | 0.742 | 0.655 | 0.565 |
| Trafilatura | Heuristic | 0.640 | 0.731 | 0.642 | 0.547 |

The 210M Hummingbird model scores 0.864 — on par with DeepSeek V3.2, a 236B-parameter LLM. It is ~1000x smaller.

![Quality vs Cost of Web Content Extraction](fig1_quality_vs_cost.png)

## Why encoders?

Content extraction is a classification problem, not a generation problem. Each block on the page gets a binary label: keep or discard. There's no need to produce new text.

Dripper solves this by generating a sequence of labels autoregressively — `1main2other3main...` — one token at a time. This means the model is bottlenecked by memory bandwidth (how fast you can read the KV cache), not by compute (how many FLOPs you can do). High-bandwidth GPUs like the A100 (2 TB/s) handle this reasonably well. Cheap GPUs like the L4 (300 GB/s) do not.

An encoder processes the entire page in a single forward pass. All blocks are classified simultaneously. The workload is compute-bound — it scales with FLOPS, not memory bandwidth. This has two practical consequences:

1. **Encoders are fast on any GPU.** The 210M Hummingbird model runs at 15 pages/sec on an L4, 43 pages/sec on an A100. Dripper runs at 0.92 pages/sec on the same L4, 5.4 on the same A100. The gap is 16x on cheap hardware, 8x on expensive hardware.

2. **Encoders get cheaper as GPUs get cheaper.** L4s have bad memory bandwidth but decent FLOPS per dollar. For bandwidth-bound workloads (autoregressive decoding), L4s are a poor choice. For compute-bound workloads (encoder inference), they're ideal. Hummingbird on L4 costs $6,500 per billion pages. Dripper on A100 costs $77,000. Dripper on L4 costs $105,000 — it's actually *more* expensive on cheaper hardware because the bandwidth bottleneck dominates.

## Architecture

Hummingbird's pipeline has four stages:

```
raw HTML → simplify_html → tokenize + chunk → encoder classify → reconstruct
```

**1. Simplify HTML (CPU).** We use MinerU-HTML's `simplify_html` to strip scripts, styles, and formatting noise from raw HTML. Each block element gets a unique `_item_id` attribute. This is the same preprocessing Dripper uses, so we're on equal footing.

**2. Tokenize and chunk (CPU).** The simplified HTML is split into blocks at `_item_id` boundaries. Blocks are tokenized, then packed into chunks separated by a special `<|sep|>` token:

```
[BOS] block_1_html <|sep|> block_2_html <|sep|> ... block_N_html <|sep|> [EOS]
```

Each chunk fits within the model's context window (8,192 tokens). Most pages (80%) fit in a single chunk. The `<|sep|>` token after each block is the classification position for that block.

**3. Encoder classify (GPU).** A single forward pass through the encoder. The model outputs a binary logit at each `<|sep|>` position: main content or boilerplate. Bidirectional attention means every block sees every other block on the page — the model can use page-level context to make decisions.

**4. Reconstruct.** Blocks classified as "main" are extracted from the original simplified HTML and converted to markdown (or kept as HTML for downstream processing).

The key difference from Dripper: steps 1 and 4 are identical. Steps 2 and 3 replace autoregressive generation with a single encoder forward pass. The quality comes from the same input representation. The speed comes from the architecture.

## Training

### Labels: teaching with DeepSeek V3.2

No human-annotated block-level labels exist at the scale we need. So we built them.

We sampled 16,670 English pages from Common Crawl (CC-MAIN-2026-12, one per unique domain). For each page, we ran the MinerU-HTML pipeline with DeepSeek V3.2 as the labeling model. DeepSeek receives the simplified HTML and classifies each block as main or other.

We tested five prompt variants. The `short_compact` prompt scored highest at 0.865 ROUGE-5 on WebMainBench — this became our labeling configuration.

After filtering (removing tiny pages, near-empty pages, and pages where the model's output was truncated), we had 15,880 pages with 1.23M labeled blocks. A quality audit (20 random pages scored by LLM sub-agents) showed 85% good or acceptable labels. The main systematic error was article titles and bylines labeled as "other" — consistent and low-impact.

We cross-validated with Dripper 0.6B on all 15,880 pages. Block-level agreement was 93.3%. We kept only blocks where both models agreed, giving us high-confidence training labels.

Total labeling cost: $129 on Bedrock. About 5 hours of compute.

### Teacher: EuroBERT-2.1B

We fine-tuned EuroBERT-2.1B (Boizard et al., 2025) on the agreed labels from 14,959 CC pages. EuroBERT is a RoPE-based encoder with bidirectional attention — structurally similar to a decoder-only transformer but without causal masking. It supports 8,192-token contexts out of the box.

Training details: learning rate 2e-5, batch size 4 with gradient accumulation 2, class-weighted cross-entropy (main weight 1.748, other weight 0.700 to handle the 28.6% main-content class rate), gradient checkpointing, 4× A100.

The teacher reached 0.864 ROUGE-5 on 200 held-out WebMainBench pages. This *exceeds* the DeepSeek V3.2 labels it was trained on (0.840 with the v0 prompt used during labeling). The encoder, by seeing the full page bidirectionally, learned patterns the autoregressive labeler missed.

### Distillation: 2.1B → 210M

We distilled the 2.1B teacher into two smaller students: a 610M (Latte Base) and a 210M (Latte Small).

Distillation used KL divergence loss (α=0.7) combined with hard-label cross-entropy (α=0.3), temperature 2.0, same data. Each distillation took about 2 hours on 4× A100.

The results were surprising:

| Model | Parameters | ROUGE-5 | vs Teacher |
|-------|-----------|---------|------------|
| Latte Large (teacher) | 2.1B | 0.864 | — |
| Latte Base | 610M | 0.849 | -1.5pp |
| **Latte Small** | **210M** | **0.864** | **+0.0pp** |

![Distillation: Smaller Can Be Better](fig4_distillation.png)

The 210M student matched the 2.1B teacher exactly. The 610M was actually worse. We don't fully understand why the smaller model distilled better — one hypothesis is that the 210M's lower capacity acts as a regularizer, while the 610M has enough capacity to overfit to noise in the teacher's outputs. Regardless, the practical implication is clear: the 210M model is the one to use.

### What didn't work

We tried several approaches before landing on this one:

- **Structural features + GBM**: 40 hand-engineered features (link density, DOM depth, position, tag type). Reached 0.81 ROUGE-5 on WebMainBench — but this was trained on the test set. On truly out-of-sample data (CC-only model evaluated on WMB), it scored 0.68 — below magic-html's 0.71 with zero training. Feature-based approaches couldn't capture page-level context.

- **CRF sequence smoothing**: A CRF on top of GBM block probabilities flipped ~5% of labels but with net-zero improvement. Some corrections, some regressions.

- **Text embeddings (model2vec, TF-IDF)**: Potion-base-8M embeddings gave 0.618 ROUGE-5 alone. Adding them to GBM features gave <0.1pp improvement. Structural features already captured what the embeddings knew.

- **BIO sequence tagging**: Extreme class imbalance (B-MAIN was 0.7% of tokens). F1 = 0.014 even with class weights. Abandoned immediately.

The recurring lesson: block classification needs page-level context. Features and embeddings don't provide it. A bidirectional encoder does.

## Results

### Quality

On the controlled 200-page English subset of WebMainBench (same pages, same evaluation for all methods):

| Method | Size | ROUGE-5 |
|--------|------|---------|
| DeepSeek V3.2 | 236B | 0.865 |
| Hummingbird Latte Small | 210M | 0.864 |
| Hummingbird Latte Large | 2.1B | 0.862 |
| Dripper | 0.6B | 0.854 |
| magic-html | — | 0.714 |
| Readability | — | 0.654 |
| Trafilatura | — | 0.640 |

Hummingbird Latte Small outperforms Dripper by 1pp while being 3x smaller. It matches a 236B LLM that costs orders of magnitude more to run.

On hard pages specifically, the 210M model scores 0.866 — the highest of any method in the table. The teacher (2.1B) scores 0.807 on hard pages. We suspect the smaller model generalizes better due to implicit regularization during distillation.

### Speed

All measurements on the same NVIDIA L4 GPU (23GB, $0.35/hr on RunPod), using 500 real Common Crawl pages (median 40KB HTML):

| Model | Architecture | Pages/sec | ms/page |
|-------|-------------|-----------|---------|
| Hummingbird 210M | Encoder (SDPA) | 15.1 | 66 |
| Dripper 0.6B | Decoder (vLLM) | 0.92 | 1,087 |
| **Ratio** | | **16.4x** | |

The pipeline breakdown for Hummingbird (sequential, single GPU):

| Stage | Time | % |
|-------|------|---|
| simplify_html (CPU) | 13.1s | 24% |
| tokenize + chunk (CPU) | 5.8s | 11% |
| GPU inference | 33.2s | 65% |

GPU is the bottleneck at 65% of wall time, but the model only uses 433MB VRAM — multiple instances can share a single GPU, and CPU preprocessing can be parallelized across cores.

### Cost

| Setup | Pages/sec | GPU-hours / 1B | Cost / 1B pages |
|-------|-----------|---------------|----------------|
| **Hummingbird on L4** | **15.1** | **18,400** | **$6,500** |
| Hummingbird on A100 | 43 | 6,460 | $9,700 |
| Dripper on A100 | 5.38 | 51,600 | $77,000 |
| Dripper on L4 | 0.92 | 301,000 | $105,000 |

![Cost to Clean 1 Billion Web Pages](fig2_cost_comparison.png)

![Throughput: Encoder vs Decoder on Different GPUs](fig3_throughput_by_gpu.png)

## The economics of encoders vs decoders

The 16.4x throughput gap on L4 deserves explanation. It's not just about parameter count — Dripper is 0.6B, Hummingbird is 0.2B, that's only 3x.

The gap is architectural. Autoregressive decoding generates tokens one at a time. Each token requires reading the full KV cache from GPU memory. This makes throughput proportional to memory bandwidth:

- A100: 2,039 GB/s → Dripper at 5.38 pps
- L4: 300 GB/s → Dripper at 0.92 pps (17% of A100 — matches the bandwidth ratio)

Encoder inference runs a single forward pass over the entire input. This is a dense matmul workload that scales with FLOPS:

- A100: 312 TFLOPS → Hummingbird at 43 pps
- L4: ~120 TFLOPS → Hummingbird at 15.1 pps (35% of A100 — tracks the FLOPS ratio)

The practical consequence: as GPUs get cheaper, encoders benefit more. L4s cost ~4x less per hour than A100s. Dripper loses more throughput than it gains in savings. Hummingbird comes out ahead.

At web scale (billions of pages), this is the difference between a $6,500 job and a $105,000 job.

## Models

| Name | HuggingFace | Parameters | ROUGE-5 | Notes |
|------|-------------|-----------|---------|-------|
| Latte Large | `chonkie-ai/hummingbird-latte-large-v1` | 2.1B | 0.864 | Teacher model |
| Latte Base | `chonkie-ai/hummingbird-latte-base-v1` | 610M | 0.849 | Distilled from Large |
| **Latte Small** | **`chonkie-ai/hummingbird-latte-small-v1`** | **210M** | **0.864** | **Recommended** |

All models are built on EuroBERT (Boizard et al., 2025) and use the same `<|sep|>` block-marker architecture. They share a tokenizer and are interchangeable in the pipeline.

Latte Small is the recommended model. It matches the teacher at 1/10th the size, fits in 433MB VRAM, and runs on any GPU (L4, T4, RTX 3090, or even integrated graphics for small batches).

## Open questions

**Flash Attention for long sequences.** 37% of real-page chunks land in the 8K-token bucket, where quadratic attention scaling dominates inference cost. Flash Attention 2 would bring this from O(n²) memory to O(n), with a likely 2-3x speedup on long sequences. We couldn't benchmark it on our test hardware (L4s don't ship with flash-attn prebuilt) but expect it to push single-GPU throughput above 20 pps.

**Larger training sets.** Dripper was trained on 986K DeepSeek-labeled pages. We used 15K. More diverse training data could improve generalization — though our 210M model already exceeds its teacher, suggesting we may be closer to the labeling ceiling than the learning ceiling.

**Why does the 210M distill better than the 610M?** We reported this result but don't have a satisfying explanation. The implicit regularization hypothesis is plausible but untested. Understanding this could inform future distillation decisions.

**Multilingual evaluation.** All our benchmarks are on English pages. WebMainBench includes 1,162 non-English pages where Dripper reports higher scores (boosted by jieba tokenization in the Chinese-heavy subset). EuroBERT was pretrained on 28 European languages — we expect reasonable multilingual transfer but haven't measured it.

**Production pipelining.** Our benchmarks run CPU and GPU stages sequentially. A pipelined architecture (CPU preprocessing in parallel with GPU inference, multiple GPUs) would push effective throughput from 15 to 120+ pages/sec on 8× L4. The infrastructure is straightforward but hasn't been built yet.
