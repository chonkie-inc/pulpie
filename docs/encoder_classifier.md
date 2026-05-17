# Encoder-Based Block Classification

## Motivation

Pulpie's LightGBM classifier is saturated at ~0.81 ROUGE-5 / 66% qrater clean rate. Dripper 0.6B achieves 0.878 ROUGE-5 / 80% qrater clean by seeing the full page context, but uses autoregressive decoding — fundamentally slow (~20 pg/s batched, ~100K A100-hours for AICC-scale 7.3B pages).

**Key insight**: Dripper is doing classification (every output token is one of exactly two choices: "main" or "other"), but paying the cost of generation. An encoder can classify all blocks in a single forward pass — 10-50x faster.

## Target

Close the 14pp qrater gap (66% → 80%) while staying fast enough for large-scale processing:
- **Quality**: ≥0.85 ROUGE-5, ≥75% qrater clean rate
- **Speed**: ≥100 pg/s per GPU (under 20K A100-hours for AICC-scale)
- **Model size**: ≤150M params (CPU-viable with ONNX)

## Architecture Approaches

### Approach 1: PL-Marker Style — `[BLOCK]` Token Classification (recommended)

Insert a special `[BLOCK]` token at each block boundary in the simplified HTML. Encoder self-attention lets each marker attend to the full page. Classify at marker positions only.

**Input format:**
```
[BLOCK] Nav Home About Contact [BLOCK] Article Title [BLOCK] This is the
main content paragraph about pulpie... [BLOCK] Related articles sidebar
```

**Architecture:**
```
ModernBERT-base (150M, 8K context)
+ 1 special [BLOCK] token added to vocabulary (random init, fine-tuned)
+ Linear(hidden_dim, 1) classification head
+ BCE loss at [BLOCK] positions only
```

**Inference:**
```python
hidden_states = encoder(input_ids)            # [batch, seq_len, 768]
block_mask = (input_ids == BLOCK_TOKEN_ID)
block_repr = hidden_states[block_mask]        # [n_blocks, 768]
logits = classifier_head(block_repr)          # [n_blocks, 1]
```

**Why this approach:**
- No pooling over variable-length spans — each marker's hidden state is a learned summary
- No BIO aggregation — one prediction per block, no inconsistency risk
- Self-attention naturally captures "I'm after a heading, page has nav at top, content blocks surround me"
- Minimal token overhead (~50 extra tokens for markers, well within 8K context)
- Dead simple implementation — standard HuggingFace fine-tuning

**Prior work:** PL-Marker (Ye et al., ACL 2022) uses this pattern for NER span classification and achieves SOTA on 6 benchmarks. The marker positions learn to aggregate span + context information through attention.

### Approach 2: BIO Token Tagging

Label each token as B-MAIN, I-MAIN, or O. Standard token classification — no custom architecture needed.

**Input:** Simplified HTML, tokenized normally
**Output:** Per-token BIO labels, aggregated to blocks by majority vote or first-token label

**Pros:**
- Simplest implementation — off-the-shelf HuggingFace `ForTokenClassification`
- Full page context from self-attention
- No special tokens needed

**Cons:**
- Aggregation from token to block level can be noisy
- Long blocks have many I-MAIN tokens dominating the loss
- Risk of inconsistent predictions within a block (B without I, etc.)

### Approach 3: Boundary Token Span Classification (Lee et al. 2017)

For each block, take the start and end token hidden states plus a width embedding:
```
span_repr = [h_start; h_end; width_embedding]
```

**Pros:**
- Proven in coreference and NER (consistently outperforms mean pooling)
- SpanBERT showed boundary tokens can encode full span content

**Cons:**
- Requires tracking token→block span mappings
- Two positions per block (start + end) vs one for PL-Marker
- More complex than Approach 1 for similar expected performance

### Approach 4: Two-Stage Block Sequence Model (Web2Text / BoilerNet style)

1. Encode each block independently with a small encoder → local representation
2. Sequence model (Transformer/LSTM/CRF) over block representations → contextualized
3. Classify each contextualized block

**Pros:**
- Handles arbitrarily long pages (no context window limit)
- Proven in neural boilerplate detection (Web2Text: CNN+HMM, BoilerNet: LSTM)
- Modeling inter-block dependencies shown to improve over independent classification

**Cons:**
- Two-stage adds complexity and latency
- Block-level encoder misses cross-block token attention
- More parameters overall

### Approach 5: MarkupLM Token Classification

Use Microsoft's MarkupLM (BERT + XPath embeddings). Already pre-trained on HTML structure. Fine-tune with per-node labels.

**Pros:**
- Pre-trained on HTML — XPath embeddings encode DOM position natively
- HuggingFace implementation ready to use

**Cons:**
- 512 token context limit — too short for most pages (avg ~1-2K tokens simplified)
- Pre-trained model is old (BERT-base era, 2021)
- Would need to upgrade to a long-context variant

## Recommended Plan

1. **Start with Approach 1 (PL-Marker `[BLOCK]` tokens + ModernBERT-base)**
   - Fastest to prototype, cleanest architecture
   - 8K context covers ~95% of pages
   - Training: ~30-60 min on 1 A100

2. **If context window is a bottleneck**, fall back to Approach 4 (two-stage)

3. **If quality is close but not enough**, try adding XPath-style positional features (DOM depth, tag type) as extra embeddings — borrowing from MarkupLM

## Training Data

- **WebMainBench**: 7,809 pages with `cc-select="true"` annotations → ~450K blocks
- **Common Crawl**: 14,959 pages with DeepSeek + Dripper agreed labels → ~350K blocks
- **Total**: ~23K pages, ~800K labeled blocks

Training format per page:
1. Simplify HTML (MinerU-HTML pipeline)
2. Insert `[BLOCK]` tokens at each `_item_id` position
3. Tokenize full page
4. Labels: 1 (main) or 0 (other) at each `[BLOCK]` position

## Speed Estimates

| Model | Params | Est. pg/s (GPU) | AICC-scale hours | Est. cost |
|-------|--------|-----------------|-----------------|-----------|
| Dripper 0.5B (AR) | 500M | ~20 | 100K | $200K |
| ModernBERT-base | 150M | ~300 | ~7K | $14K |
| DeBERTa-v3-small | 44M | ~500+ | ~4K | $8K |
| ONNX-quantized 44M | 44M | ~50 pg/s CPU | N/A (CPU) | — |

## References

- PL-Marker: Ye et al., "Packed Levitated Marker for Entity and Relation Extraction", ACL 2022 (arXiv 2109.06067)
- SpanBERT: Joshi et al., "SpanBERT: Improving Pre-Training by Representing and Predicting Spans", 2019 (arXiv 1907.10529)
- Lee et al., "End-to-end Neural Coreference Resolution", EMNLP 2017
- Web2Text: Vogels et al., "Web2Text: Deep Structured Boilerplate Removal", ECIR 2018 (arXiv 1801.02607)
- BoilerNet: Leonhardt et al., LSTM-based boilerplate detection
- MarkupLM: Li et al., "MarkupLM: Pre-Training of Text and Markup Language for Visually-rich Document Understanding", ACL 2022 (arXiv 2110.08518)
- Dripper: Liu et al., "Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM", 2025 (arXiv 2511.23119)
