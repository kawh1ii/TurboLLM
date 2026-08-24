import collections
import csv
import gzip
import json
import sys
from pathlib import Path


phase = sys.argv[1]

profile_dir = Path(
    sys.argv[2]
)

meta_path = Path(
    sys.argv[3]
)

summary_path = Path(
    sys.argv[4]
)


RULES = [
    # IMPORTANT:
    #
    # KV must precede ATTENTION because production names
    # such as reshape_and_cache_flash_kernel contain BOTH
    # "cache" and "flash".
    (
        "KV_CACHE",
        [
            "reshape_and_cache",
            "unified_kv_cache",
            "kv_cache_update",
            "kv_cache",
            "cache_update",
            "compute_slot_mapping",
            "compute_slot_mappings",
            "gather_block_tables",
        ],
    ),

    (
        "ATTENTION",
        [
            "flash_fwd",
            "flash::",
            "varlen_fwd",
            "paged_attention",
            "attention",
        ],
    ),

    # GEMV must precede GEMM because some cuBLAS GEMV
    # names also contain broader cuBLAS/GEMM substrings.
    (
        "GEMV",
        [
            "gemvx",
            "gemv",
        ],
    ),

    (
        "GEMM",
        [
            "gemm",
            "cutlass",
            "cublaslt",
            "wmma",
            "tensorop",
        ],
    ),

    (
        "NORM",
        [
            "rms_norm",
            "rmsnorm",
            "layer_norm",
            "fused_add_rms",
            "_norm_",
        ],
    ),

    (
        "SAMPLING",
        [
            "gumbel",
            "sample",
            "sampling",
            "multinomial",
            "argmax",
        ],
    ),

    # Copy must precede generic elementwise because
    # PyTorch direct-copy kernels are implemented through
    # elementwise templates.
    (
        "MEMORY_COPY",
        [
            "memcpy",
            "direct_copy",
            "copy_kernel",
        ],
    ),

    (
        "REDUCTION",
        [
            "reduce_kernel",
            "reduction",
        ],
    ),

    (
        "ACTIVATION_ELEMENTWISE",
        [
            "silu",
            "gelu",
            "elementwise",
            "triton_poi",
            "triton_per_fused",
            "vectorized_elementwise",
        ],
    ),
]


def matching_families(name):
    lower = name.lower()

    found = []

    for family, patterns in RULES:

        if any(
            pattern in lower
            for pattern in patterns
        ):
            found.append(
                family
            )

    return found


def classify(name):
    found = matching_families(
        name
    )

    if found:
        return found[0]

    return "OTHER"


files = sorted(
    list(
        profile_dir.rglob(
            "*.json"
        )
    )
    +
    list(
        profile_dir.rglob(
            "*.json.gz"
        )
    )
)


if not files:
    raise RuntimeError(
        f"No trace files under "
        f"{profile_dir}"
    )


meta = json.loads(
    meta_path.read_text(
        encoding="utf-8",
    )
)


kernel_total_us = 0.0
kernel_event_count = 0

kernel_time = collections.defaultdict(
    float
)

kernel_count = collections.Counter()

family_time = collections.defaultdict(
    float
)

family_count = collections.Counter()

overlap_time_us = 0.0
overlap_event_count = 0

overlap_names = collections.defaultdict(
    float
)


for path in files:

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


    for event in obj.get(
        "traceEvents",
        [],
    ):

        if event.get("ph") != "X":
            continue

        category = str(
            event.get(
                "cat",
                "",
            )
        ).lower()

        if (
            "kernel" not in category
            and
            category != "cuda_kernel"
        ):
            continue

        name = str(
            event.get(
                "name",
                "UNKNOWN",
            )
        )

        try:
            duration_us = float(
                event.get(
                    "dur",
                    0.0,
                )
            )
        except Exception:
            duration_us = 0.0


        kernel_total_us += (
            duration_us
        )

        kernel_event_count += 1

        kernel_time[name] += (
            duration_us
        )

        kernel_count[name] += 1


        matches = matching_families(
            name
        )

        family = (
            matches[0]
            if matches
            else "OTHER"
        )

        family_time[family] += (
            duration_us
        )

        family_count[family] += 1


        if len(matches) > 1:

            overlap_event_count += 1

            overlap_time_us += (
                duration_us
            )

            overlap_names[
                name
            ] += duration_us


families = [
    family
    for family, _ in RULES
] + [
    "OTHER"
]


summary = {
    **meta,

    "trace_file_count": (
        len(files)
    ),

    "cuda_kernel_event_count": (
        kernel_event_count
    ),

    "cuda_kernel_total_ms": (
        kernel_total_us
        / 1000.0
    ),

    "overlap_event_count": (
        overlap_event_count
    ),

    "overlap_cuda_time_ms": (
        overlap_time_us
        / 1000.0
    ),

    "families": {},

    "top_kernels": [],
}


for family in families:

    total_us = family_time[
        family
    ]

    summary[
        "families"
    ][family] = {
        "cuda_time_ms": (
            total_us
            / 1000.0
        ),

        "cuda_time_percent": (
            100.0
            * total_us
            / kernel_total_us
            if kernel_total_us > 0
            else 0.0
        ),

        "event_count": (
            family_count[
                family
            ]
        ),
    }


top = sorted(
    kernel_time.items(),
    key=lambda x: x[1],
    reverse=True,
)[:40]


for name, total_us in top:

    matches = matching_families(
        name
    )

    summary[
        "top_kernels"
    ].append(
        {
            "name": name,

            "family": (
                matches[0]
                if matches
                else "OTHER"
            ),

            "all_matching_families": (
                matches
            ),

            "cuda_time_ms": (
                total_us
                / 1000.0
            ),

            "cuda_time_percent": (
                100.0
                * total_us
                / kernel_total_us
                if kernel_total_us > 0
                else 0.0
            ),

            "event_count": (
                kernel_count[
                    name
                ]
            ),
        }
    )


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 120)
print(
    f"P1-M12 TRACE ANALYSIS — "
    f"{phase.upper()}"
)
print("=" * 120)

print(
    "TRACE_FILE_COUNT =",
    len(files),
)

print(
    "CUDA_KERNEL_EVENT_COUNT =",
    kernel_event_count,
)

print(
    "CUDA_KERNEL_TOTAL_TIME_MS =",
    f"{kernel_total_us / 1000.0:.6f}",
)

print(
    "OVERLAP_EVENT_COUNT =",
    overlap_event_count,
)

print(
    "OVERLAP_CUDA_TIME_MS =",
    f"{overlap_time_us / 1000.0:.6f}",
)


print()
print("=" * 120)
print(
    "MUTUALLY EXCLUSIVE FAMILY ATTRIBUTION"
)
print("=" * 120)


for family in families:

    info = summary[
        "families"
    ][family]

    print(
        f"{family}_CUDA_TIME_MS = "
        f"{info['cuda_time_ms']:.6f}"
    )

    print(
        f"{family}_CUDA_TIME_PERCENT = "
        f"{info['cuda_time_percent']:.3f}%"
    )

    print(
        f"{family}_EVENT_COUNT = "
        f"{info['event_count']}"
    )


print()
print("=" * 120)
print("TOP CUDA KERNELS")
print("=" * 120)


for rank, item in enumerate(
    summary[
        "top_kernels"
    ][:30],
    1,
):

    print(
        f"{rank:02d}. "
        f"FAMILY={item['family']} "
        f"TOTAL_MS={item['cuda_time_ms']:.6f} "
        f"PCT={item['cuda_time_percent']:.3f}% "
        f"COUNT={item['event_count']} "
        f"NAME={item['name']}"
    )


print()
print("=" * 120)
print("OVERLAP AUDIT")
print("=" * 120)


for rank, (
    name,
    total_us,
) in enumerate(
    sorted(
        overlap_names.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20],
    1,
):

    print(
        f"{rank:02d}. "
        f"CHOSEN={classify(name)} "
        f"MATCHES={matching_families(name)} "
        f"TOTAL_MS={total_us / 1000.0:.6f} "
        f"NAME={name}"
    )


print()
print("=" * 120)
print("CORRECTNESS CHECK")
print("=" * 120)


reshape_matches = [
    item
    for item in summary[
        "top_kernels"
    ]
    if "reshape_and_cache" in (
        item["name"].lower()
    )
]


if reshape_matches:

    for item in reshape_matches:

        print(
            "RESHAPE_AND_CACHE_CLASSIFICATION =",
            item["family"],
        )

        print(
            "RESHAPE_AND_CACHE_ALL_MATCHES =",
            item[
                "all_matching_families"
            ],
        )

else:

    print(
        "RESHAPE_AND_CACHE_CLASSIFICATION = "
        "NOT_IN_TOP40"
    )


print(
    "SUMMARY_JSON =",
    summary_path,
)


if kernel_event_count <= 0:
    raise RuntimeError(
        "No CUDA kernel events detected."
    )


print(
    "TRACE_ANALYSIS = SUCCESS"
)
