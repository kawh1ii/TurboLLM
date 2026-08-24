import collections
import gzip
import json
import os
import sys
from pathlib import Path


profile_dir = Path(
    sys.argv[1]
)


files = sorted(
    list(
        profile_dir.rglob("*.json")
    )
    +
    list(
        profile_dir.rglob("*.json.gz")
    )
)


print("=" * 120)
print("P1-M11B PYTORCH PROFILER TRACE ANALYSIS")
print("=" * 120)

print(
    "PROFILE_DIR =",
    profile_dir,
)

print(
    "TRACE_FILE_COUNT =",
    len(files),
)


if not files:
    raise RuntimeError(
        "No PyTorch profiler trace files found."
    )


kernel_time = collections.defaultdict(
    float
)

kernel_count = collections.Counter()

cpu_time = collections.defaultdict(
    float
)

cpu_count = collections.Counter()

memcpy_time = collections.defaultdict(
    float
)

all_kernel_events = 0
all_kernel_duration_us = 0.0


for path in files:

    print(
        "TRACE_FILE =",
        path,
    )

    try:
        if path.suffix == ".gz":
            with gzip.open(
                path,
                "rt",
                encoding="utf-8",
            ) as f:
                obj = json.load(f)
        else:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                obj = json.load(f)

    except Exception as e:
        print(
            "TRACE_PARSE_WARNING =",
            repr(e),
        )
        continue


    events = obj.get(
        "traceEvents",
        [],
    )


    for event in events:

        if event.get("ph") != "X":
            continue

        name = str(
            event.get(
                "name",
                "UNKNOWN",
            )
        )

        cat = str(
            event.get(
                "cat",
                "",
            )
        ).lower()

        try:
            dur = float(
                event.get(
                    "dur",
                    0.0,
                )
            )
        except Exception:
            dur = 0.0


        # Kineto CUDA kernel events normally use cat=kernel.
        if (
            "kernel" in cat
            or
            cat == "cuda_kernel"
        ):
            kernel_time[name] += dur
            kernel_count[name] += 1

            all_kernel_events += 1
            all_kernel_duration_us += dur

            continue


        if (
            "gpu_memcpy" in cat
            or
            "memcpy" in cat
        ):
            memcpy_time[name] += dur


        if (
            "cpu_op" in cat
            or
            "operator" in cat
        ):
            cpu_time[name] += dur
            cpu_count[name] += 1


print()
print("=" * 120)
print("CUDA KERNEL SUMMARY")
print("=" * 120)

print(
    "CUDA_KERNEL_EVENT_COUNT =",
    all_kernel_events,
)

print(
    "CUDA_KERNEL_TOTAL_TIME_MS =",
    f"{all_kernel_duration_us / 1000.0:.6f}",
)


print()
print("TOP CUDA KERNELS BY TOTAL CUDA TIME:")

top = sorted(
    kernel_time.items(),
    key=lambda x: x[1],
    reverse=True,
)[:30]


for rank, (
    name,
    total_us,
) in enumerate(
    top,
    1,
):
    count = kernel_count[name]

    avg_us = (
        total_us / count
        if count
        else 0.0
    )

    pct = (
        100.0
        * total_us
        / all_kernel_duration_us
        if all_kernel_duration_us > 0
        else 0.0
    )

    print(
        f"{rank:02d}. "
        f"TOTAL_MS={total_us/1000.0:.6f} "
        f"PCT={pct:.3f}% "
        f"COUNT={count} "
        f"AVG_US={avg_us:.3f} "
        f"NAME={name}"
    )


print()
print("=" * 120)
print("KERNEL FAMILY ATTRIBUTION")
print("=" * 120)


families = {
    "ATTENTION": [
        "attention",
        "flash",
        "paged",
    ],

    "GEMM_MATMUL": [
        "gemm",
        "matmul",
        "cutlass",
        "cublas",
    ],

    "RMSNORM_NORM": [
        "rms",
        "norm",
    ],

    "SOFTMAX": [
        "softmax",
    ],

    "SAMPLING": [
        "sample",
        "sampling",
        "gumbel",
        "multinomial",
    ],

    "KV_CACHE": [
        "cache",
        "kv",
        "reshape_and_cache",
    ],

    "ELEMENTWISE_REDUCTION": [
        "reduce",
        "elementwise",
        "vectorized",
    ],

    "MEMORY_COPY": [
        "memcpy",
        "copy",
    ],
}


family_time = {
    family: 0.0
    for family in families
}


for name, total in kernel_time.items():

    lower = name.lower()

    matched = False

    for family, words in families.items():

        if any(
            word in lower
            for word in words
        ):
            family_time[family] += total

            matched = True

            break


classified = sum(
    family_time.values()
)


for family, total_us in sorted(
    family_time.items(),
    key=lambda x: x[1],
    reverse=True,
):
    pct_all = (
        100.0
        * total_us
        / all_kernel_duration_us
        if all_kernel_duration_us
        else 0.0
    )

    print(
        f"{family}_CUDA_TIME_MS = "
        f"{total_us / 1000.0:.6f}"
    )

    print(
        f"{family}_CUDA_TIME_PERCENT = "
        f"{pct_all:.3f}%"
    )


print()
print(
    "CLASSIFIED_KERNEL_TIME_PERCENT =",
    (
        f"{100.0 * classified / all_kernel_duration_us:.3f}%"
        if all_kernel_duration_us
        else "0.000%"
    ),
)


print()
print("=" * 120)
print("TOP CPU OPERATORS")
print("=" * 120)


for rank, (
    name,
    total_us,
) in enumerate(
    sorted(
        cpu_time.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20],
    1,
):
    print(
        f"{rank:02d}. "
        f"TOTAL_MS={total_us/1000.0:.6f} "
        f"COUNT={cpu_count[name]} "
        f"NAME={name}"
    )


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print(
"""
This trace is a production vLLM execution trace.

Unlike the synthetic P1-M11A CUDA kernels, these events
come from the actual inference worker.

PyTorch Profiler can identify:

    CUDA kernel names
    CUDA kernel durations
    launch counts
    CPU operators
    relative CUDA-time contribution

It does NOT replace Nsight Compute hardware counters.

Because ERR_NVGPUCTRPERM blocks privileged counters,
this run must NOT claim measured:

    DRAM bandwidth utilization
    Tensor Core utilization
    achieved occupancy
    SM hardware-counter utilization

Those require a host that exposes NVIDIA GPU
performance counters.

The valid evidence chain on this host is therefore:

    TTFT / TPOT / throughput
            ↓
    scheduler + KV metrics
            ↓
    coarse GPU telemetry
            ↓
    production CUDA kernel names and timing
            ↓
    hardware-counter layer unavailable on host
""".strip()
)


if all_kernel_events <= 0:
    print()
    print(
        "KERNEL_TRACE_VERDICT = "
        "NO_CUDA_KERNEL_EVENTS_FOUND"
    )

    raise RuntimeError(
        "Trace files exist but no CUDA kernel events "
        "were detected."
    )


print()
print("=" * 120)
print("FINAL TRACE VERDICT")
print("=" * 120)

print(
    "TRACE_FILES = SUCCESS"
)

print(
    "CUDA_KERNEL_EVENTS = SUCCESS"
)

print(
    "PRODUCTION_VLLM_KERNEL_TIMING = SUCCESS"
)

print(
    "HARDWARE_COUNTER_LIMITATION = "
    "ERR_NVGPUCTRPERM"
)

print(
    "VERDICT = "
    "READY_FOR_P1_M12_KERNEL_ATTRIBUTION_AND_OPTIMIZATION"
)
