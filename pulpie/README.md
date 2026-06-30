<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/chonkie-inc/pulpie/main/pulpie/assets/banner-dark.png">
    <img alt="pulpie" src="https://raw.githubusercontent.com/chonkie-inc/pulpie/main/pulpie/assets/banner-light.png" width="440">
  </picture>
</p>

Fast content extraction from HTML using encoder models. 16x faster than autoregressive approaches at the same quality.

## Install

```bash
pip install pulpie
```

For markdown output:
```bash
pip install pulpie[markdown]
```

## Usage

```python
from pulpie import Extractor

extractor = Extractor()  # downloads pulpie-orange-small (210M) on first use

result = extractor.extract(html)
print(result.markdown)   # clean markdown
print(result.html)       # clean HTML
print(result.n_main)     # number of content blocks
print(result.n_other)    # number of boilerplate blocks
```

## Models

| Model | Size | ROUGE-5 | Speed (L4) |
|-------|------|---------|------------|
| `orange-small` | 210M | 0.864 | 15 pps |
| `orange-base` | 610M | 0.849 | ~6 pps |
| `orange-large` | 2.1B | 0.862 | ~2 pps |

`orange-small` is the default and recommended model — it matches the 2.1B teacher at 1/10th the size.

```python
# Use a specific model
extractor = Extractor(model="orange-large")

# Use a custom model path
extractor = Extractor(model="path/to/your/model")

# Force CPU
extractor = Extractor(device="cpu")
```

## How it works

Pulpie classifies each HTML block as "main content" or "boilerplate" using a bidirectional encoder. The pipeline:

1. **Simplify** — Strip scripts, styles, normalize HTML (via MinerU-HTML)
2. **Chunk** — Pack blocks into sequences separated by `<|sep|>` tokens
3. **Classify** — Single encoder forward pass classifies all blocks simultaneously
4. **Reconstruct** — Extract content blocks, convert to markdown

## Performance

On 500 real Common Crawl pages (NVIDIA L4 GPU):

- **15.1 pages/sec** (single GPU, 210M model)
- **$6,500** to clean 1 billion pages
- **16.4x faster** than Dripper (autoregressive) on the same hardware
- **433 MB** VRAM — fits on any GPU

## License

Apache 2.0
