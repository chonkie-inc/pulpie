<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/chonkie-inc/pulpie/main/assets/banner-dark.png">
  <img alt="pulpie" src="https://raw.githubusercontent.com/chonkie-inc/pulpie/main/assets/banner-light.png" width="460">
</picture>

[![PyPI version](https://img.shields.io/pypi/v/pulpie.svg)](https://pypi.org/project/pulpie/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://pypi.org/project/pulpie/)
[![License](https://img.shields.io/github/license/chonkie-inc/pulpie.svg)](https://github.com/chonkie-inc/pulpie/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/pulpie)](https://pepy.tech/project/pulpie)
[![Blog](https://img.shields.io/badge/blog-read%20the%20writeup-E34C26.svg)](https://usefeyn.com/blog/pulpie-pareto-optimal-models-for-cleaning-the-web/)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/pulpie.svg)](https://github.com/chonkie-inc/pulpie/stargazers)

_Pareto-optimal models for cleaning the web. Extract main content from HTML at one twentieth the cost._

[Install](#installation) •
[Usage](#usage) •
[Models](#models) •
[How it works](#how-it-works) •
[Benchmarks](#benchmarks) •
[Blog](https://usefeyn.com/blog/pulpie-pareto-optimal-models-for-cleaning-the-web/)

</div>

Pulpie extracts the main content from raw HTML, stripping navigation, ads, sidebars, and footers. It uses small encoder models that label every block in a single forward pass, approaching state-of-the-art extraction quality while running up to 20x faster and 20x cheaper than autoregressive extractors on an L4 GPU.

- **Fast.** An encoder labels every block in one forward pass (13.7 pages/sec on an L4).
- **Accurate.** Matches state-of-the-art quality: 0.862 to 0.873 ROUGE-5 F1 on WebMainBench.
- **Small.** The recommended model is 210M parameters and fits on any GPU.
- **Cheap.** Clean 1 billion pages for ~$7,900, versus ~$159,000 for the leading decoder.
- **Simple.** Run `pip install pulpie`, then `Extractor().extract(html)`.
- **Batched.** An overlapped CPU and GPU pipeline scales across multiple GPUs.

## Installation

```bash
pip install pulpie
```

For Markdown output, install the `markdown` extra:

```bash
pip install "pulpie[markdown]"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "pulpie[markdown]"
```

## Usage

### Basic

```python
from pulpie import Extractor

extractor = Extractor()                # defaults to pulpie-orange-small (210M)
result = extractor.extract(html)

print(result.markdown)                 # clean Markdown
print(result.html)                     # clean HTML
print(result.n_main, result.n_other)   # blocks kept vs dropped
```

The model downloads from Hugging Face on first use.

### Choosing a model

```python
extractor = Extractor(model="orange-large")   # "orange-small" (default), "orange-base", "orange-large"
extractor = Extractor(model="path/to/model")  # or a custom checkpoint
extractor = Extractor(device="cpu")           # force CPU
```

### Batch processing

For bulk extraction, `Pipeline` overlaps CPU preprocessing with GPU inference and self-balances across one or more GPUs:

```python
from pulpie import Pipeline, PageInput

pipeline = Pipeline(model="orange-small")
results = pipeline.extract_batch(
    [PageInput(html=h, page_id=i) for i, h in enumerate(pages)]
)
```

## Models

All three models are built on [EuroBERT](https://arxiv.org/abs/2503.05500), share a tokenizer, and use the same `<|sep|>` block-marker architecture. Large is the teacher; Base and Small are distilled from it.

| Model | Hugging Face | Params | ROUGE-5 F1 | Notes |
|-------|--------------|--------|------------|-------|
| **Orange Small** | [`chonkie-ai/pulpie-orange-small`](https://huggingface.co/chonkie-ai/pulpie-orange-small) | 210M | 0.862 | **Recommended**, best size-to-quality ratio |
| Orange Base | [`chonkie-ai/pulpie-orange-base`](https://huggingface.co/chonkie-ai/pulpie-orange-base) | 610M | 0.863 | Distilled from Large |
| Orange Large | [`chonkie-ai/pulpie-orange-large`](https://huggingface.co/chonkie-ai/pulpie-orange-large) | 2.1B | 0.873 | Teacher (highest quality) |

`orange-small` is the default. Despite being a third the size of Dripper (the leading extractor), it matches its quality (0.862 vs 0.864) while running 20x faster.

## How it works

Pulpie keeps the "read the page" approach of model-based extractors but moves the bottleneck from memory bandwidth to compute by using an encoder instead of a decoder. The pipeline runs in four stages:

1. **Simplify.** Remove scripts, styles, and formatting noise; tag each content block with a unique ID.
2. **Chunk.** Split, tokenize, and pack blocks into chunks of up to 8,192 tokens (≈80% of pages fit in one chunk).
3. **Classify.** A single encoder forward pass labels every block as content or boilerplate.
4. **Reconstruct.** Return the kept blocks as HTML, or convert them to Markdown.

A decoder emits labels one token at a time, re-reading the full model from GPU memory each step. An encoder runs one dense forward pass over the whole input, so the gap widens on bandwidth-limited GPUs (7x faster than Dripper on A100, 20x on L4).

## Benchmarks

Quality on the English subset of [WebMainBench](https://github.com/opendatalab/WebMainBench) (6,647 pages), ROUGE-5 F1:

| Method | Params | ROUGE-5 F1 | Empty pages |
|--------|--------|------------|-------------|
| **Pulpie Orange Large** | 2.1B | **0.873** | 21 |
| Dripper | 0.6B | 0.864 | 135 |
| **Pulpie Orange Base** | 610M | 0.863 | 36 |
| **Pulpie Orange Small** | 210M | 0.862 | 45 |
| magic-html | - | 0.700 | 384 |
| Trafilatura | - | 0.619 | 16 |

Speed and cost (Pulpie Orange Small vs Dripper, 1 billion pages):

| | Pulpie Orange Small | Dripper |
|--|--------------------|---------|
| Throughput (L4) | **13.7 pages/sec** | 0.68 pages/sec |
| Cost / 1B pages (L4) | **~$7,900** | ~$159,000 |

Pulpie Orange Small matches Dripper's quality at **20x the throughput** and **20x lower cost** on an L4. See [BENCHMARKS.md](BENCHMARKS.md) for the full comparison, per-difficulty breakdown, and reproduction command.

## Acknowledgements

Pulpie builds directly on the work of the MinerU-HTML and Dripper team (Ma et al., 2025). Their `simplify_html` preprocessing, block-level annotation scheme, and the WebMainBench benchmark are foundational to this work. We also use their Dripper 0.6B model to cross-validate our training labels. We're grateful they released their tools and data.

## Citation

If you use Pulpie in your research, please cite:

```bibtex
@note{pulpie2026,
  title  = {Pulpie: Pareto-Optimal Models for Cleaning the Web},
  author = {Minhas, Bhavnick and Nigam, Shreyash and Feyn Research},
  year   = {2026},
  venue  = {Feyn Field Notes}
}
```

---

<div align="center">
Built by <a href="https://usefeyn.com">Feyn</a>.
</div>
