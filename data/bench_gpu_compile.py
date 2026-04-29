"""GPU benchmark for EuroBERT-210m with torch.compile + SDPA.

Tests seq lengths up to 8192 with compile optimizations to get
realistic throughput numbers for cost estimation.
"""

import time
import torch
import numpy as np
from transformers import AutoModelForTokenClassification

MODEL = "EuroBERT/EuroBERT-210m"
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 4, 8, 16, 32]
N_WARMUP = 10
N_RUNS = 50


def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"torch.compile + SDPA enabled\n")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.0f}M")
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"GPU memory (model): {mem:.0f}MB")

    print("\nCompiling model...")
    model = torch.compile(model, mode="reduce-overhead")

    # Warmup compile with a small input
    with torch.no_grad():
        dummy = torch.randint(100, 30000, (1, 512), device=device)
        mask = torch.ones(1, 512, dtype=torch.long, device=device)
        for _ in range(3):
            model(input_ids=dummy, attention_mask=mask)
    torch.cuda.synchronize()
    print("Compile warmup done.\n")

    print(f"{'SeqLen':>8}  {'Batch':>6}  {'Latency':>10}  {'Pages/sec':>10}  {'ms/page':>10}  {'Peak MB':>8}")
    print(f"{'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    results = {}

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

                results[(seq_len, batch_size)] = {
                    "latency_ms": avg, "pps": pps, "ms_per_page": ms_per_page
                }

            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8}  {batch_size:>6}  OOM")
                torch.cuda.empty_cache()

        print()

    # Cost table
    print("\n" + "="*80)
    print("COST ESTIMATES (best batch size per seq length)")
    print("="*80)

    gpus = [
        ("A100 80GB",  1.00, 1.64),
        ("RTX 4090",   0.55, 0.39),
        ("RTX 3090",   0.45, 0.22),
        ("L4 24GB",    0.40, 0.35),
        ("L40S 48GB",  0.60, 0.80),
        ("T4 16GB",    0.21, 0.20),
        ("A10 24GB",   0.40, 0.50),
    ]

    for seq_len in SEQ_LENGTHS:
        best_pps = 0
        for (sl, bs), r in results.items():
            if sl == seq_len and r["pps"] > best_pps:
                best_pps = r["pps"]

        if best_pps == 0:
            continue

        print(f"\n  seq={seq_len} (A100 baseline: {best_pps:.0f} pages/sec)")
        print(f"  {'GPU':<15} {'Pages/sec':>10} {'$/hr':>8} {'$/1M pages':>12} {'$/1B pages':>12}")
        print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
        for name, ratio, cost_hr in gpus:
            pps = best_pps * ratio
            cost_1m = cost_hr / (pps * 3600) * 1e6
            cost_1b = cost_1m * 1000
            print(f"  {name:<15} {pps:>10.0f} {cost_hr:>7.2f} {cost_1m:>11.1f} {cost_1b:>11.0f}")


if __name__ == "__main__":
    main()
