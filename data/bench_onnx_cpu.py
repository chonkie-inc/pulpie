"""Benchmark ONNX INT8 inference latency on CPU.

Exports real models via torch.onnx, quantizes with onnxruntime,
and measures actual latency at realistic sequence lengths.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def benchmark_model(model_name, seq_lengths, num_labels=2, n_warmup=5, n_runs=30, num_threads=4):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}", flush=True)

    tmpdir = tempfile.mkdtemp()
    onnx_fp32 = os.path.join(tmpdir, "model.onnx")
    onnx_int8 = os.path.join(tmpdir, "model_int8.onnx")

    try:
        # Load model
        print(f"  Loading model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels, trust_remote_code=True
        )
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params/1e6:.0f}M", flush=True)

        # Export to ONNX
        print(f"  Exporting to ONNX...", flush=True)
        dummy_len = 512
        dummy_ids = torch.randint(100, 30000, (1, dummy_len))
        dummy_mask = torch.ones(1, dummy_len, dtype=torch.long)

        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask),
            onnx_fp32,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=14,
        )

        fp32_size = os.path.getsize(onnx_fp32)
        print(f"  FP32 ONNX: {fp32_size/1e6:.0f}MB", flush=True)

        # Quantize to INT8
        print(f"  Quantizing to INT8...", flush=True)
        quantize_dynamic(onnx_fp32, onnx_int8, weight_type=QuantType.QInt8)
        int8_size = os.path.getsize(onnx_int8)
        print(f"  INT8 ONNX: {int8_size/1e6:.0f}MB", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Benchmark both FP32 and INT8
        for label, path in [("FP32", onnx_fp32), ("INT8", onnx_int8)]:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(path, sess_options, providers=["CPUExecutionProvider"])

            print(f"\n  {label} (threads={num_threads}):")
            print(f"  {'SeqLen':>8}  {'Latency':>10}  {'Std':>8}  {'Pages/sec':>10}  {'ms/block':>10}")
            print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

            for seq_len in seq_lengths:
                input_ids = np.random.randint(100, 30000, size=(1, seq_len)).astype(np.int64)
                attention_mask = np.ones((1, seq_len), dtype=np.int64)
                inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

                for _ in range(n_warmup):
                    session.run(None, inputs)

                latencies = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    session.run(None, inputs)
                    latencies.append((time.perf_counter() - t0) * 1000)

                avg = np.mean(latencies)
                std = np.std(latencies)
                pps = 1000.0 / avg
                # Estimate ~20 tokens per block, so blocks = seq_len / 20
                blocks = seq_len / 20
                ms_per_block = avg / blocks

                print(f"  {seq_len:>8}  {avg:>8.1f}ms  {std:>6.1f}ms  {pps:>10.1f}  {ms_per_block:>8.2f}ms")

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    seq_lengths = [512, 1024, 2048]
    num_threads = 4

    print("=" * 60)
    print("ONNX CPU Inference Latency Benchmark")
    print("=" * 60)
    print(f"Threads: {num_threads}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Goal: <150ms per page")

    models = [
        "distilbert/distilbert-base-uncased",   # 66M
        "EuroBERT/EuroBERT-210m",               # 210M
    ]

    for m in models:
        benchmark_model(m, seq_lengths, num_threads=num_threads)

    # Also test with more threads
    print(f"\n\n{'='*60}")
    print(f"Scaling test: EuroBERT-210m INT8 at different thread counts")
    print(f"{'='*60}")
    # Quick test just at 1024 tokens
    benchmark_model("EuroBERT/EuroBERT-210m", [1024], num_threads=8)
    benchmark_model("EuroBERT/EuroBERT-210m", [1024], num_threads=16)


if __name__ == "__main__":
    main()
