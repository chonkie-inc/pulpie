# Benchmarks

## WebMainBench (English-only, 6,647 pages)

Evaluation on the full English subset of [WebMainBench](https://github.com/nickochar/WebMainBench) using ROUGE-5 F1 (5-gram overlap, whitespace tokenized).

Ground truth: `convert_main_content` field (html2text with bodywidth=0, ignore_links=True, ignore_images=True).

| Method | Params | All | Simple | Mid | Hard | Empty | P | R | pg/s |
|--------|--------|-----|--------|-----|------|-------|---|---|------|
| **Pulpie Large** | 2.1B | **0.873** | 0.914 | **0.878** | **0.827** | 21 | 0.865 | 0.917 | 1.2 |
| Dripper | 0.6B | 0.864 | **0.914** | 0.865 | 0.818 | 135 | 0.860 | 0.901 | 3.5 |
| **Pulpie Base** | 610M | 0.863 | 0.906 | 0.868 | 0.818 | 36 | 0.858 | 0.906 | 2.1 |
| **Pulpie Small** | 210M | 0.862 | 0.906 | 0.868 | 0.813 | 45 | 0.854 | 0.910 | 4.0 |
| magic-html | — | 0.700 | 0.773 | 0.697 | 0.637 | 384 | 0.778 | 0.704 | 14.7 |
| Raw html2text | — | 0.620 | 0.779 | 0.605 | 0.491 | 0 | 0.515 | 0.943 | — |
| Trafilatura | — | 0.619 | 0.721 | 0.619 | 0.526 | 16 | 0.688 | 0.610 | 17.9 |

**Difficulty distribution**: simple=1,888 | mid=2,720 | hard=2,039

### Notes

- **pg/s**: pages/second on a single A100 80GB GPU.
- **Pulpie models** use a naive sequential inference loop (no batching). Throughput can be improved with batched inference.
- **Dripper** ([MinerU-HTML v1.1](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact)) uses vLLM with guided regex decoding for batched inference.
- **Empty**: pages where the method produced no output (context overflow for Dripper, no blocks detected for Pulpie).
- **magic-html**: [opendatalab/magic-html](https://github.com/opendatalab/magic-html) rule-based extractor.
- **Trafilatura**: [trafilatura](https://github.com/adbar/trafilatura) with `include_tables=True`.
- **Raw html2text**: full HTML passed directly through html2text (no extraction).

### Cost comparison (1B pages, projected)

| Setup | pg/s | GPU-hours | Cost (RunPod) |
|-------|------|-----------|---------------|
| Pulpie Small on L4 | 15.1 | 18,400 | $6,500 |
| Pulpie Small on A100 | 4.0 (unbatched) | 69,400 | $9,700 |
| Dripper on A100 (vLLM) | 3.5 | 79,400 | $77,000 |

Pulpie Small matches Dripper quality (0.862 vs 0.864) at ~12x lower cost.

### Hardware

- GPU: NVIDIA A100 80GB
- Framework: PyTorch 2.6 + transformers 4.57
- vLLM 0.11.1 (for Dripper)

### Reproducibility

```bash
python eval/bench_all_methods.py \
    --methods pulpie-small,pulpie-base,pulpie-large,dripper,trafilatura,magic-html,raw-h2t \
    --limit 0 \
    --device cuda:0
```
