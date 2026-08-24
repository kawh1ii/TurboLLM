import csv
import os
import platform
import statistics
import time
from pathlib import Path

import torch


CSV_PATH = Path(os.environ["P1_M1_CSV"])

results = []


def add_result(category, test, dtype, size, value, unit, notes=""):
    results.append(
        {
            "category": category,
            "test": test,
            "dtype": dtype,
            "size": size,
            "value": value,
            "unit": unit,
            "notes": notes,
        }
    )


def median_cpu_matmul(n, repeats=5):
    a = torch.randn((n, n), dtype=torch.float32)
    b = torch.randn((n, n), dtype=torch.float32)

    # warmup
    _ = a @ b

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        c = a @ b
        elapsed = time.perf_counter() - start

        # force a real use of the result
        _ = c[0, 0].item()

        times.append(elapsed * 1000)

    return statistics.median(times)


def median_gpu_matmul(n, dtype, repeats=20, warmup=5):
    a = torch.randn((n, n), device="cuda", dtype=dtype)
    b = torch.randn((n, n), device="cuda", dtype=dtype)

    for _ in range(warmup):
        _ = a @ b

    torch.cuda.synchronize()

    times = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        c = a @ b
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    _ = c[0, 0].item()

    del a, b, c
    torch.cuda.empty_cache()

    return statistics.median(times)


def copy_h2d(size_mb, repeats=10):
    numel = size_mb * 1024 * 1024 // 4

    host = torch.randn(numel, dtype=torch.float32)
    device = torch.empty(numel, device="cuda", dtype=torch.float32)

    # warmup
    device.copy_(host)
    torch.cuda.synchronize()

    times = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        device.copy_(host)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_ms = statistics.median(times)
    gb = size_mb / 1024
    gbps = gb / (median_ms / 1000)

    del host, device
    torch.cuda.empty_cache()

    return median_ms, gbps


def copy_d2h(size_mb, repeats=10):
    numel = size_mb * 1024 * 1024 // 4

    device = torch.randn(numel, device="cuda", dtype=torch.float32)
    host = torch.empty(numel, dtype=torch.float32)

    host.copy_(device)
    torch.cuda.synchronize()

    times = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        host.copy_(device)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    median_ms = statistics.median(times)
    gb = size_mb / 1024
    gbps = gb / (median_ms / 1000)

    del host, device
    torch.cuda.empty_cache()

    return median_ms, gbps


LINE = "=" * 120

print(LINE)
print("AI INFRA PROJECT 1")
print("P1-M1 PYTORCH GPU BASICS")
print(LINE)

print()
print("SYSTEM")
print("-" * 120)
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)

print("GPU:", device_name)
print("Compute capability:", f"{props.major}.{props.minor}")
print("VRAM GiB:", round(props.total_memory / 1024**3, 3))
print("SM count:", props.multi_processor_count)

print()
print(LINE)
print("EXPERIMENT 1 — CPU VS GPU MATRIX MULTIPLICATION")
print(LINE)

sizes = [512, 1024, 2048, 4096]

for n in sizes:
    cpu_ms = median_cpu_matmul(n)

    gpu_fp32_ms = median_gpu_matmul(
        n=n,
        dtype=torch.float32,
    )

    gpu_fp16_ms = median_gpu_matmul(
        n=n,
        dtype=torch.float16,
    )

    print()
    print(f"Matrix: {n} x {n}")
    print(f"CPU FP32 median: {cpu_ms:.6f} ms")
    print(f"GPU FP32 median: {gpu_fp32_ms:.6f} ms")
    print(f"GPU FP16 median: {gpu_fp16_ms:.6f} ms")

    add_result(
        "matmul",
        "cpu",
        "fp32",
        n,
        cpu_ms,
        "ms",
    )

    add_result(
        "matmul",
        "gpu",
        "fp32",
        n,
        gpu_fp32_ms,
        "ms",
    )

    add_result(
        "matmul",
        "gpu",
        "fp16",
        n,
        gpu_fp16_ms,
        "ms",
    )

print()
print(LINE)
print("EXPERIMENT 2 — HOST ↔ GPU MEMORY COPY")
print(LINE)

for size_mb in [16, 64, 256]:
    h2d_ms, h2d_gbps = copy_h2d(size_mb)
    d2h_ms, d2h_gbps = copy_d2h(size_mb)

    print()
    print(f"Transfer size: {size_mb} MiB")
    print(f"H2D median: {h2d_ms:.6f} ms")
    print(f"H2D effective bandwidth: {h2d_gbps:.3f} GiB/s")
    print(f"D2H median: {d2h_ms:.6f} ms")
    print(f"D2H effective bandwidth: {d2h_gbps:.3f} GiB/s")

    add_result(
        "memory_copy",
        "host_to_device",
        "fp32",
        size_mb,
        h2d_ms,
        "ms",
    )

    add_result(
        "memory_copy",
        "host_to_device_bandwidth",
        "fp32",
        size_mb,
        h2d_gbps,
        "GiB/s",
    )

    add_result(
        "memory_copy",
        "device_to_host",
        "fp32",
        size_mb,
        d2h_ms,
        "ms",
    )

    add_result(
        "memory_copy",
        "device_to_host_bandwidth",
        "fp32",
        size_mb,
        d2h_gbps,
        "GiB/s",
    )

print()
print(LINE)
print("EXPERIMENT 3 — CUDA ASYNCHRONOUS EXECUTION")
print(LINE)

n = 4096

a = torch.randn((n, n), device="cuda", dtype=torch.float16)
b = torch.randn((n, n), device="cuda", dtype=torch.float16)

for _ in range(5):
    _ = a @ b

torch.cuda.synchronize()

host_start = time.perf_counter()

for _ in range(20):
    c = a @ b

host_enqueue_end = time.perf_counter()

torch.cuda.synchronize()

sync_end = time.perf_counter()

enqueue_ms = (host_enqueue_end - host_start) * 1000
total_ms = (sync_end - host_start) * 1000

print(f"20 GPU matmuls host enqueue time: {enqueue_ms:.6f} ms")
print(f"20 GPU matmuls synchronized total time: {total_ms:.6f} ms")
print()
print(
    "Interpretation: CUDA kernel launches are asynchronous from the CPU's "
    "point of view. Measuring only host enqueue time does NOT measure full "
    "GPU execution time."
)

add_result(
    "async_execution",
    "host_enqueue_20_matmuls",
    "fp16",
    n,
    enqueue_ms,
    "ms",
)

add_result(
    "async_execution",
    "synchronized_20_matmuls",
    "fp16",
    n,
    total_ms,
    "ms",
)

del a, b, c
torch.cuda.empty_cache()

print()
print(LINE)
print("EXPERIMENT 4 — GPU MEMORY")
print(LINE)

torch.cuda.reset_peak_memory_stats()

before_alloc = torch.cuda.memory_allocated()
before_reserved = torch.cuda.memory_reserved()

x = torch.empty(
    (256 * 1024 * 1024 // 2,),
    dtype=torch.float16,
    device="cuda",
)

after_alloc = torch.cuda.memory_allocated()
after_reserved = torch.cuda.memory_reserved()
peak = torch.cuda.max_memory_allocated()

print(
    "Allocated before:",
    round(before_alloc / 1024**2, 3),
    "MiB",
)

print(
    "Reserved before:",
    round(before_reserved / 1024**2, 3),
    "MiB",
)

print(
    "Allocated after 256 MiB tensor:",
    round(after_alloc / 1024**2, 3),
    "MiB",
)

print(
    "Reserved after:",
    round(after_reserved / 1024**2, 3),
    "MiB",
)

print(
    "Peak allocated:",
    round(peak / 1024**2, 3),
    "MiB",
)

add_result(
    "gpu_memory",
    "allocated_after_tensor",
    "fp16",
    256,
    after_alloc / 1024**2,
    "MiB",
)

add_result(
    "gpu_memory",
    "reserved_after_tensor",
    "fp16",
    256,
    after_reserved / 1024**2,
    "MiB",
)

del x
torch.cuda.empty_cache()

print()
print(LINE)
print("WRITE STRUCTURED RESULTS")
print(LINE)

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "test",
            "dtype",
            "size",
            "value",
            "unit",
            "notes",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("CSV:", CSV_PATH)

print()
print(LINE)
print("P1-M1 CONCEPT CHECKPOINT")
print(LINE)

print(
    """
1. CPU Tensor 和 CUDA Tensor 位于不同的 memory space。
2. Host→Device / Device→Host 数据移动本身有成本。
3. CUDA kernel launch 默认具有异步特征。
4. 因此 GPU benchmark 必须明确 synchronization。
5. GPU memory allocated 与 reserved 不是同一个概念。
6. 小 workload 不一定能充分利用 GPU。
7. FP16/BF16 等低精度计算是后续 LLM 性能分析的重要基础。
""".strip()
)

print()
print(LINE)
print("P1-M1 VERDICT")
print(LINE)

assert len(results) > 0
assert CSV_PATH.exists()

print("CUDA_EXECUTION = SUCCESS")
print("CPU_GPU_MATMUL = SUCCESS")
print("HOST_DEVICE_COPY = SUCCESS")
print("CUDA_ASYNC_DEMONSTRATION = SUCCESS")
print("GPU_MEMORY_OBSERVATION = SUCCESS")
print("STRUCTURED_RESULTS = SUCCESS")
print("VERDICT = READY_FOR_P1_M2_CUDA_EXECUTION_MODEL")
