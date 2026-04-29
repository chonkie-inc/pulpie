"""Quick GPU inference benchmark for EuroBERT-210m token classification."""

import time
import torch
import numpy as np
from transformers import AutoModelForTokenClassification

MODEL = "EuroBERT/EuroBERT-210m"
SEQ_LENGTHS = [512, 1024, 2048]
BATCH_SIZES = [1, 4, 8, 16, 32]
N_WARMUP = 10
N_RUNS = 50

def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}\n")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=2, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.0f}M")
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"GPU memory: {mem:.0f}MB\n")

    print(f"{'SeqLen':>8}  {'Batch':>6}  {'Latency':>10}  {'Pages/sec':>10}  {'ms/page':>10}")
    print(f"{'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

    for seq_len in SEQ_LENGTHS:
        for batch_size in BATCH_SIZES:
            try:
                input_ids = torch.randint(100, 30000, (batch_size, seq_len), device=device)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

                # Warmup
                with torch.no_grad():
                    for _ in range(N_WARMUP):
                        model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize()

                # Timed
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

                print(f"{seq_len:>8}  {batch_size:>6}  {avg:>8.1f}ms  {pps:>10.1f}  {ms_per_page:>8.2f}ms")

            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8}  {batch_size:>6}  OOM")
                torch.cuda.empty_cache()

        print()

    # Rough GPU tier estimates
    print("\nEstimated throughput on different GPUs (seq=1024, batch=16):")
    print("(A100 = 1.0x baseline, others estimated from FP16 TFLOPS ratios)\n")

    # Get our A100 number at seq=1024 batch=16
    input_ids = torch.randint(100, 30000, (16, 1024), device=device)
    attention_mask = torch.ones(16, 1024, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    a100_ms = np.mean(latencies)
    a100_pps = 16 * 1000.0 / a100_ms

    # FP16 TFLOPS ratios relative to A100 (312 TFLOPS)
    gpus = [
        ("A100 80GB", 1.0, 1.80),      # $1.80/hr runpod
        ("A10 24GB",  0.40, 0.50),      # $0.50/hr
        ("L4 24GB",   0.40, 0.35),      # $0.35/hr
        ("T4 16GB",   0.21, 0.20),      # $0.20/hr
        ("A4000 16GB",0.32, 0.30),      # $0.30/hr
        ("L40S 48GB", 0.60, 0.80),      # $0.80/hr
    ]

    print(f"  {'GPU':<15} {'Pages/sec':>10} {'$/hr':>8} {'$/1M pages':>12} {'$/1B pages':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
    for name, ratio, cost_hr in gpus:
        pps = a100_pps * ratio
        cost_1m = cost_hr / (pps * 3600) * 1e6
        cost_1b = cost_1m * 1000
        print(f"  {name:<15} {pps:>10.0f} {cost_hr:>7.2f} {cost_1m:>11.1f} {cost_1b:>11.0f}")


if __name__ == "__main__":
    main()
