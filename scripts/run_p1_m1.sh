#!/usr/bin/env bash
set -euo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"
RESULT="${PROJECT}/results/p1_m1"
ARTIFACT="${BASE}/artifacts"
TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p "${RESULT}" "${ARTIFACT}" "${PROJECT}/src/p1_m1"

TXT="${ARTIFACT}/P1_M1_PYTORCH_GPU_BASICS_${TS}.txt"
CSV="${RESULT}/P1_M1_RESULTS_${TS}.csv"
PY="${PROJECT}/src/p1_m1/p1_m1_gpu_basics.py"

cat > "${PY}" <<'PY'
import csv
import os
import statistics
import time

import torch

LINE = "=" * 120
CSV_PATH = os.environ["RESULT_CSV"]

records = []

def record(section, test, size, dtype, value, unit):
    records.append({
        "section": section,
        "test": test,
        "size": size,
        "dtype": dtype,
        "value": value,
        "unit": unit,
    })

def cpu_matmul(n, repeats=5):
    a = torch.randn(n, n)
    b = torch.randn(n, n)

    _ = a @ b

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        c = a @ b
        _ = c[0, 0].item()
        times.append((time.perf_counter() - t0) * 1000)

    return statistics.median(times)

def gpu_matmul(n, dtype, repeats=20):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    for _ in range(5):
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

    del a, b, c
    torch.cuda.empty_cache()

    return statistics.median(times)

def copy_test(size_mb, pinned, direction, repeats=10):
    numel = size_mb * 1024 * 1024 // 4

    if direction == "H2D":
        host = torch.randn(
            numel,
            dtype=torch.float32,
            pin_memory=pinned,
        )
        device = torch.empty(
            numel,
            device="cuda",
            dtype=torch.float32,
        )

        device.copy_(host)
        torch.cuda.synchronize()

        times = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            device.copy_(host, non_blocking=pinned)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    else:
        device = torch.randn(
            numel,
            device="cuda",
            dtype=torch.float32,
        )

        host = torch.empty(
            numel,
            dtype=torch.float32,
            pin_memory=pinned,
        )

        host.copy_(device)
        torch.cuda.synchronize()

        times = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            host.copy_(device, non_blocking=pinned)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    ms = statistics.median(times)
    gib = size_mb / 1024
    bandwidth = gib / (ms / 1000)

    del host, device
    torch.cuda.empty_cache()

    return ms, bandwidth


print(LINE)
print("AI INFRA PROJECT 1")
print("P1-M1 — PYTORCH GPU EXECUTION BASICS")
print(LINE)

print()
print("ENVIRONMENT")
print("-" * 120)

print("PyTorch =", torch.__version__)
print("PyTorch CUDA =", torch.version.cuda)
print("CUDA available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

props = torch.cuda.get_device_properties(0)

print("GPU =", torch.cuda.get_device_name(0))
print("VRAM_GiB =", round(props.total_memory / 1024**3, 3))
print("Compute Capability =", f"{props.major}.{props.minor}")
print("SM Count =", props.multi_processor_count)

print()
print(LINE)
print("EXPERIMENT 1 — CPU VS GPU MATRIX MULTIPLICATION")
print(LINE)

for n in [512, 1024, 2048, 4096]:
    cpu = cpu_matmul(n)

    gpu32 = gpu_matmul(
        n,
        torch.float32,
    )

    gpu16 = gpu_matmul(
        n,
        torch.float16,
    )

    print()
    print(f"{n} x {n}")
    print(f"CPU FP32       = {cpu:.6f} ms")
    print(f"GPU FP32       = {gpu32:.6f} ms")
    print(f"GPU FP16       = {gpu16:.6f} ms")
    print(f"CPU/GPU FP32   = {cpu/gpu32:.3f}x")
    print(f"GPU FP32/FP16  = {gpu32/gpu16:.3f}x")

    record("matmul", "CPU", n, "FP32", cpu, "ms")
    record("matmul", "GPU", n, "FP32", gpu32, "ms")
    record("matmul", "GPU", n, "FP16", gpu16, "ms")


print()
print(LINE)
print("EXPERIMENT 2 — HOST ↔ DEVICE MEMORY TRANSFER")
print(LINE)

for size in [16, 64, 256]:

    for pinned in [False, True]:

        h2d_ms, h2d_bw = copy_test(
            size,
            pinned,
            "H2D",
        )

        d2h_ms, d2h_bw = copy_test(
            size,
            pinned,
            "D2H",
        )

        label = "PINNED" if pinned else "PAGEABLE"

        print()
        print(f"{size} MiB / {label}")
        print(f"H2D = {h2d_ms:.6f} ms / {h2d_bw:.3f} GiB/s")
        print(f"D2H = {d2h_ms:.6f} ms / {d2h_bw:.3f} GiB/s")

        record(
            "transfer",
            f"H2D_{label}",
            size,
            "FP32",
            h2d_bw,
            "GiB/s",
        )

        record(
            "transfer",
            f"D2H_{label}",
            size,
            "FP32",
            d2h_bw,
            "GiB/s",
        )


print()
print(LINE)
print("EXPERIMENT 3 — CUDA ASYNCHRONOUS EXECUTION")
print(LINE)

n = 4096

a = torch.randn(
    n,
    n,
    device="cuda",
    dtype=torch.float16,
)

b = torch.randn(
    n,
    n,
    device="cuda",
    dtype=torch.float16,
)

for _ in range(5):
    _ = a @ b

torch.cuda.synchronize()

start = time.perf_counter()

for _ in range(30):
    c = a @ b

enqueue_end = time.perf_counter()

torch.cuda.synchronize()

finish = time.perf_counter()

enqueue_ms = (enqueue_end - start) * 1000
full_ms = (finish - start) * 1000

print("30 kernel launches enqueue time =", f"{enqueue_ms:.6f} ms")
print("30 operations synchronized time =", f"{full_ms:.6f} ms")

print()
print(
    "IMPORTANT: CPU enqueue time != actual GPU execution time."
)

record(
    "async",
    "CPU_enqueue_30_matmul",
    n,
    "FP16",
    enqueue_ms,
    "ms",
)

record(
    "async",
    "GPU_synchronized_30_matmul",
    n,
    "FP16",
    full_ms,
    "ms",
)

del a, b, c
torch.cuda.empty_cache()


print()
print(LINE)
print("EXPERIMENT 4 — PYTORCH CUDA MEMORY ALLOCATOR")
print(LINE)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

before_alloc = torch.cuda.memory_allocated()
before_reserved = torch.cuda.memory_reserved()

# Exactly ~512 MiB FP16 tensor
numel = 512 * 1024 * 1024 // 2

x = torch.empty(
    numel,
    device="cuda",
    dtype=torch.float16,
)

after_alloc = torch.cuda.memory_allocated()
after_reserved = torch.cuda.memory_reserved()

print(
    "allocated_before_MiB =",
    round(before_alloc / 1024**2, 3),
)

print(
    "reserved_before_MiB =",
    round(before_reserved / 1024**2, 3),
)

print(
    "allocated_after_MiB =",
    round(after_alloc / 1024**2, 3),
)

print(
    "reserved_after_MiB =",
    round(after_reserved / 1024**2, 3),
)

print()
print(
    "allocated = memory occupied by live tensors"
)

print(
    "reserved  = memory held by PyTorch CUDA caching allocator"
)

record(
    "memory",
    "allocated",
    512,
    "FP16",
    after_alloc / 1024**2,
    "MiB",
)

record(
    "memory",
    "reserved",
    512,
    "FP16",
    after_reserved / 1024**2,
    "MiB",
)

del x
torch.cuda.empty_cache()


print()
print(LINE)
print("SAVE RESULTS")
print(LINE)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "section",
            "test",
            "size",
            "dtype",
            "value",
            "unit",
        ],
    )

    writer.writeheader()
    writer.writerows(records)

print("CSV_RESULT =", CSV_PATH)


print()
print(LINE)
print("P1-M1 KNOWLEDGE CHECKPOINT")
print(LINE)

print("""
1. CPU 与 GPU 使用不同的 memory space。
2. CPU→GPU 与 GPU→CPU 的数据搬运存在真实成本。
3. Pinned memory 可以支持更高效的异步传输。
4. CUDA kernel launch 默认具有异步特征。
5. 因此 GPU benchmark 不能只使用普通 CPU timer。
6. 正确 GPU benchmark 通常需要 CUDA Event 或 synchronize。
7. FP16 在 RTX 4090 上可以利用 Tensor Core。
8. 小矩阵不一定能充分发挥 GPU。
9. torch.cuda.memory_allocated() 和 memory_reserved() 含义不同。
10. 后续 LLM Serving 的性能问题，本质上会不断回到：
    compute / memory / transfer / synchronization。
""".strip())


print()
print(LINE)
print("FINAL VERDICT")
print(LINE)

print("CUDA_EXECUTION = SUCCESS")
print("CPU_GPU_MATMUL = SUCCESS")
print("HOST_DEVICE_TRANSFER = SUCCESS")
print("PINNED_MEMORY_TEST = SUCCESS")
print("CUDA_ASYNC_TEST = SUCCESS")
print("CUDA_MEMORY_ALLOCATOR_TEST = SUCCESS")
print("STRUCTURED_CSV_RESULT = SUCCESS")
print("VERDICT = READY_FOR_P1_M2_GPU_CUDA_EXECUTION_MODEL")
PY


exec > >(tee "${TXT}") 2>&1

echo "========================================================================================================================"
echo "AI INFRA PROJECT 1 - P1-M1"
echo "PYTORCH GPU EXECUTION BASICS"
echo "========================================================================================================================"
echo

echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

echo "PYTHON:"
python --version
echo

echo "PYTORCH:"
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY

echo
echo "========================================================================================================================"
echo "RUN EXPERIMENT"
echo "========================================================================================================================"
echo

RESULT_CSV="${CSV}" \
python "${PY}"

echo
echo "========================================================================================================================"
echo "FILES"
echo "========================================================================================================================"
echo "${CSV}"
echo "${PY}"

echo
echo "========================================================================================================================"
echo "END"
echo "========================================================================================================================"

exec 1>&-
exec 2>&-

echo "${TXT}"
