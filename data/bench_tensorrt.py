"""Benchmark EuroBERT-210M: torch.compile+SDPA vs TensorRT (via ONNX).

Exports to ONNX, converts to TRT with FP16, then compares latency.
"""

import os
import time
import tempfile
import numpy as np
import torch
from transformers import AutoModelForTokenClassification

MODEL = "EuroBERT/EuroBERT-210m"
SEQ_LENGTHS = [512, 1024, 2048, 4096]
BATCH_SIZE = 16
N_WARMUP = 10
N_RUNS = 50


def bench_torch_compile(device):
    """Benchmark torch.compile + SDPA."""
    print("\n=== torch.compile + SDPA ===")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=2, trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="sdpa",
    ).to(device).eval()

    model = torch.compile(model, mode="reduce-overhead")

    # Warmup compile
    with torch.no_grad():
        dummy = torch.randint(100, 30000, (1, 512), device=device)
        mask = torch.ones(1, 512, dtype=torch.long, device=device)
        for _ in range(3):
            model(input_ids=dummy, attention_mask=mask)
    torch.cuda.synchronize()

    results = {}
    for seq_len in SEQ_LENGTHS:
        input_ids = torch.randint(100, 30000, (BATCH_SIZE, seq_len), device=device)
        attention_mask = torch.ones(BATCH_SIZE, seq_len, dtype=torch.long, device=device)

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
        pps = BATCH_SIZE * 1000.0 / avg
        results[seq_len] = {"ms": avg, "pps": pps}
        print(f"  seq={seq_len}: {avg:.1f}ms, {pps:.0f} pps")

    del model
    torch.cuda.empty_cache()
    return results


def bench_tensorrt(device):
    """Export ONNX → TensorRT FP16, benchmark."""
    import tensorrt as trt

    print("\n=== TensorRT FP16 (via ONNX) ===")

    tmpdir = tempfile.mkdtemp()
    onnx_path = os.path.join(tmpdir, "eurobert.onnx")
    trt_path = os.path.join(tmpdir, "eurobert.trt")

    # Export ONNX
    print("  Exporting ONNX...", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=2, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(device).eval()

    dummy_ids = torch.randint(100, 30000, (1, 512), device=device)
    dummy_mask = torch.ones(1, 512, dtype=torch.long, device=device)

    torch.onnx.export(
        model, (dummy_ids, dummy_mask), onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print(f"  ONNX: {os.path.getsize(onnx_path)/1e6:.0f}MB", flush=True)

    del model
    torch.cuda.empty_cache()

    # Build TRT engine
    print("  Building TensorRT engine (FP16)...", flush=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            return {}

    # Set dynamic shapes
    max_seq = max(SEQ_LENGTHS)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, 64), (BATCH_SIZE, 1024), (BATCH_SIZE, max_seq))
    profile.set_shape("attention_mask", (1, 64), (BATCH_SIZE, 1024), (BATCH_SIZE, max_seq))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("  TRT build failed!")
        return {}

    with open(trt_path, "wb") as f:
        f.write(engine_bytes)
    print(f"  TRT engine: {os.path.getsize(trt_path)/1e6:.0f}MB", flush=True)

    # Load and benchmark
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    results = {}
    stream = torch.cuda.Stream()

    for seq_len in SEQ_LENGTHS:
        input_ids = torch.randint(100, 30000, (BATCH_SIZE, seq_len), device=device, dtype=torch.int64)
        attention_mask = torch.ones(BATCH_SIZE, seq_len, device=device, dtype=torch.int64)
        output = torch.zeros(BATCH_SIZE, seq_len, 2, device=device, dtype=torch.float16)

        context.set_input_shape("input_ids", (BATCH_SIZE, seq_len))
        context.set_input_shape("attention_mask", (BATCH_SIZE, seq_len))

        context.set_tensor_address("input_ids", input_ids.data_ptr())
        context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        context.set_tensor_address("logits", output.data_ptr())

        # Warmup
        for _ in range(N_WARMUP):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        latencies = []
        for _ in range(N_RUNS):
            stream.synchronize()
            t0 = time.perf_counter()
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(latencies)
        pps = BATCH_SIZE * 1000.0 / avg
        results[seq_len] = {"ms": avg, "pps": pps}
        print(f"  seq={seq_len}: {avg:.1f}ms, {pps:.0f} pps")

    return results


def main():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}, batch={BATCH_SIZE}")

    compile_results = bench_torch_compile(device)
    trt_results = bench_tensorrt(device)

    if trt_results:
        print(f"\n{'='*60}")
        print(f"COMPARISON (batch={BATCH_SIZE})")
        print(f"{'='*60}")
        print(f"{'SeqLen':>8}  {'Compile pps':>12}  {'TRT pps':>12}  {'Speedup':>10}")
        print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
        for seq_len in SEQ_LENGTHS:
            c = compile_results.get(seq_len, {})
            t = trt_results.get(seq_len, {})
            c_pps = c.get("pps", 0)
            t_pps = t.get("pps", 0)
            speedup = t_pps / c_pps if c_pps > 0 else 0
            print(f"{seq_len:>8}  {c_pps:>11.0f}  {t_pps:>11.0f}  {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
