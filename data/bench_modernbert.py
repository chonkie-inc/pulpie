"""Benchmark ModernBERT-base (149M) vs EuroBERT-210M with torch.compile + SDPA."""

import time
import torch
import numpy as np
from transformers import AutoModelForTokenClassification

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 4, 8, 16, 32]
N_WARMUP = 10
N_RUNS = 50

MODELS = [
    ("ModernBERT-base (149M)", "answerdotai/ModernBERT-base"),
    ("EuroBERT-210M", "EuroBERT/EuroBERT-210m"),
]


def bench_model(name, model_id, device):
    print(f"\n{'='*70}")
    print(f"{name}: {model_id}")
    print(f"{'='*70}")

    kwargs = dict(num_labels=2, trust_remote_code=True, torch_dtype=torch.float16)
    if "ModernBERT" not in model_id:
        kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForTokenClassification.from_pretrained(model_id, **kwargs).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"Params: {n_params/1e6:.0f}M, VRAM: {mem:.0f}MB")

    print("Compiling...")
    model = torch.compile(model, mode="reduce-overhead")

    with torch.no_grad():
        dummy = torch.randint(100, 30000, (1, 512), device=device)
        mask = torch.ones(1, 512, dtype=torch.long, device=device)
        for _ in range(3):
            model(input_ids=dummy, attention_mask=mask)
    torch.cuda.synchronize()

    results = {}

    print(f"\n{'SeqLen':>8}  {'Batch':>6}  {'Latency':>10}  {'Pages/sec':>10}  {'ms/page':>10}  {'Peak MB':>8}")
    print(f"{'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for seq_len in SEQ_LENGTHS:
        for batch_size in BATCH_SIZES:
            try:
                torch.cuda.reset_peak_memory_stats()
                input_ids = torch.randint(100, 30000, (batch_size, seq_len), device=device)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

                with torch.no_grad():
                    for _ in range(N_WARMUP):
                        model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()

                latencies = []
                with torch.no_grad():
                    for _ in range(N_RUNS):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        model(input_ids=input_ids, attention_mask=attention_mask)
                        torch.cuda.synchronize()
                        latencies.append((time.perf_counter() - t0) * 1000)

                avg = np.mean(latencies)
                pps = batch_size * 1000.0 / avg
                ms_per_page = avg / batch_size
                peak_mb = torch.cuda.max_memory_allocated() / 1e6

                print(f"{seq_len:>8}  {batch_size:>6}  {avg:>8.1f}ms  {pps:>10.1f}  {ms_per_page:>8.2f}ms  {peak_mb:>7.0f}")
                results[(seq_len, batch_size)] = {"pps": pps, "ms": avg}

            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8}  {batch_size:>6}  OOM")
                torch.cuda.empty_cache()
        print()

    del model
    torch.cuda.empty_cache()
    return results


def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {}
    for name, model_id in MODELS:
        all_results[name] = bench_model(name, model_id, device)

    # Comparison
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD (best batch size per seq length)")
    print(f"{'='*70}")
    print(f"\n{'SeqLen':>8}  {'ModernBERT pps':>15}  {'EuroBERT pps':>15}  {'Speedup':>10}")
    print(f"{'-'*8}  {'-'*15}  {'-'*15}  {'-'*10}")

    for seq_len in SEQ_LENGTHS:
        best_modern = max((v["pps"] for (s, b), v in all_results[MODELS[0][0]].items() if s == seq_len), default=0)
        best_euro = max((v["pps"] for (s, b), v in all_results[MODELS[1][0]].items() if s == seq_len), default=0)
        speedup = best_modern / best_euro if best_euro > 0 else 0
        print(f"{seq_len:>8}  {best_modern:>14.0f}  {best_euro:>14.0f}  {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
