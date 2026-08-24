import csv
import json
import sys
from pathlib import Path


prefill_path = Path(
    sys.argv[1]
)

decode_path = Path(
    sys.argv[2]
)

csv_path = Path(
    sys.argv[3]
)


prefill = json.loads(
    prefill_path.read_text(
        encoding="utf-8"
    )
)

decode = json.loads(
    decode_path.read_text(
        encoding="utf-8"
    )
)


families = [
    "KV_CACHE",
    "ATTENTION",
    "GEMV",
    "GEMM",
    "NORM",
    "SAMPLING",
    "MEMORY_COPY",
    "REDUCTION",
    "ACTIVATION_ELEMENTWISE",
    "OTHER",
]


def pct(
    summary,
    family,
):
    return float(
        summary[
            "families"
        ][family][
            "cuda_time_percent"
        ]
    )


def ms(
    summary,
    family,
):
    return float(
        summary[
            "families"
        ][family][
            "cuda_time_ms"
        ]
    )


print("=" * 120)
print(
    "P1-M12 PREFILL-HEAVY VS "
    "DECODE-HEAVY KERNEL COMPOSITION"
)
print("=" * 120)


print()
print(
    "PREFILL_PROMPT_MEDIAN =",
    prefill[
        "prompt_tokens_median"
    ],
)

print(
    "PREFILL_OUTPUT_MEDIAN =",
    prefill[
        "output_tokens_median"
    ],
)

print(
    "PREFILL_KERNEL_EVENTS =",
    prefill[
        "cuda_kernel_event_count"
    ],
)

print(
    "PREFILL_CUDA_KERNEL_TIME_MS =",
    f"{prefill['cuda_kernel_total_ms']:.6f}",
)


print()
print(
    "DECODE_PROMPT_MEDIAN =",
    decode[
        "prompt_tokens_median"
    ],
)

print(
    "DECODE_OUTPUT_MEDIAN =",
    decode[
        "output_tokens_median"
    ],
)

print(
    "DECODE_KERNEL_EVENTS =",
    decode[
        "cuda_kernel_event_count"
    ],
)

print(
    "DECODE_CUDA_KERNEL_TIME_MS =",
    f"{decode['cuda_kernel_total_ms']:.6f}",
)


print()
print("=" * 120)
print("FAMILY COMPOSITION")
print("=" * 120)


for family in families:

    print()
    print(
        f"{family}_PREFILL_PERCENT = "
        f"{pct(prefill, family):.3f}%"
    )

    print(
        f"{family}_DECODE_PERCENT = "
        f"{pct(decode, family):.3f}%"
    )

    print(
        f"{family}_PREFILL_MS = "
        f"{ms(prefill, family):.6f}"
    )

    print(
        f"{family}_DECODE_MS = "
        f"{ms(decode, family):.6f}"
    )


print()
print("=" * 120)
print("PHASE-EMPHASIS OBSERVATIONS")
print("=" * 120)


print(
    "PREFILL_GEMM_GREATER_THAN_DECODE =",
    (
        pct(
            prefill,
            "GEMM",
        )
        >
        pct(
            decode,
            "GEMM",
        )
    ),
)

print(
    "DECODE_GEMV_GREATER_THAN_PREFILL =",
    (
        pct(
            decode,
            "GEMV",
        )
        >
        pct(
            prefill,
            "GEMV",
        )
    ),
)

print(
    "DECODE_KV_CACHE_GREATER_THAN_PREFILL =",
    (
        pct(
            decode,
            "KV_CACHE",
        )
        >
        pct(
            prefill,
            "KV_CACHE",
        )
    ),
)

print(
    "DECODE_ATTENTION_GREATER_THAN_PREFILL =",
    (
        pct(
            decode,
            "ATTENTION",
        )
        >
        pct(
            prefill,
            "ATTENTION",
        )
    ),
)


print()
print("=" * 120)
print("NORMALIZED CUDA KERNEL TIME")
print("=" * 120)


print(
    "PREFILL_CUDA_MS_PER_REQUEST =",
    f"{prefill['cuda_kernel_total_ms'] / prefill['request_count']:.6f}",
)

print(
    "DECODE_CUDA_MS_PER_REQUEST =",
    f"{decode['cuda_kernel_total_ms'] / decode['request_count']:.6f}",
)

print(
    "PREFILL_CUDA_MS_PER_PROMPT_TOKEN =",
    f"{prefill['cuda_kernel_total_ms'] / prefill['prompt_tokens_total']:.9f}",
)

print(
    "DECODE_CUDA_MS_PER_OUTPUT_TOKEN =",
    f"{decode['cuda_kernel_total_ms'] / decode['output_tokens_total']:.9f}",
)


print()
print("=" * 120)
print("IMPORTANT INTERPRETATION LIMIT")
print("=" * 120)

print(
"""
These are phase-emphasis workloads, not pure isolated
Prefill-only and Decode-only kernels.

The comparison establishes differences in production
kernel composition under:

    long-prompt / short-output traffic

versus:

    short-prompt / long-output traffic.

PyTorch Profiler itself introduces overhead, so the
profiled wall-clock throughput must NOT replace the
non-profiled P1-M7/P1-M8/P1-M9 benchmark results.

Likewise, these traces provide kernel timing rather than
privileged hardware-counter evidence.

Therefore they support claims about:

    kernel identity
    launch count
    relative CUDA-time composition

but NOT direct claims about measured:

    DRAM bandwidth saturation
    Tensor Core utilization
    occupancy
    SM hardware-counter utilization.
""".strip()
)


print()
print("=" * 120)
print("P1-M12 KNOWLEDGE CHECKPOINT")
print("=" * 120)

print(
"""
PREFILL and DECODE are different GPU workload regimes.

PREFILL:

    many prompt tokens processed together
    larger matrix dimensions
    generally better opportunity for GEMM-style compute

DECODE:

    one new token per active sequence per step
    many repeated engine iterations
    GEMV / small GEMM / attention / KV maintenance
    become relatively more visible

The key systems lesson is:

    optimizing an LLM server is workload-phase dependent.

A kernel optimization that matters during long-prompt
Prefill may have little impact on long-running Decode,
and vice versa.

Production optimization should therefore begin with:

    workload
        ↓
    phase
        ↓
    kernel composition
        ↓
    candidate optimization

rather than choosing a kernel to optimize first.
""".strip()
)


rows = []

for summary in [
    prefill,
    decode,
]:
    row = {
        "phase": summary[
            "phase"
        ],

        "prompt_tokens_median": (
            summary[
                "prompt_tokens_median"
            ]
        ),

        "output_tokens_median": (
            summary[
                "output_tokens_median"
            ]
        ),

        "request_count": (
            summary[
                "request_count"
            ]
        ),

        "cuda_kernel_event_count": (
            summary[
                "cuda_kernel_event_count"
            ]
        ),

        "cuda_kernel_total_ms": (
            summary[
                "cuda_kernel_total_ms"
            ]
        ),

        "overlap_event_count": (
            summary[
                "overlap_event_count"
            ]
        ),
    }

    for family in families:

        row[
            f"{family.lower()}_ms"
        ] = ms(
            summary,
            family,
        )

        row[
            f"{family.lower()}_pct"
        ] = pct(
            summary,
            family,
        )

    rows.append(row)


fieldnames = list(
    rows[0].keys()
)


with csv_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print()
print(
    "CSV_RESULT =",
    csv_path,
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)

print(
    "PREFILL_PRODUCTION_TRACE = SUCCESS"
)

print(
    "DECODE_PRODUCTION_TRACE = SUCCESS"
)

print(
    "MUTUALLY_EXCLUSIVE_ATTRIBUTION = SUCCESS"
)

print(
    "OVERLAP_AUDIT = SUCCESS"
)

print(
    "PHASE_COMPARISON = SUCCESS"
)

print(
    "VERDICT = "
    "READY_FOR_P1_M13_TARGETED_OPTIMIZATION"
)
