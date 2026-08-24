#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"
VLLM="${ENV}/bin/vllm"

HF_HOME="${BASE}/caches/huggingface"

SRC_DIR="${PROJECT}/src/p1_m12"
RESULT_DIR="${PROJECT}/results/p1_m12"
ARTIFACT_DIR="${BASE}/artifacts"

HOST="127.0.0.1"
PORT="8000"
MODEL_NAME="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

TXT="${ARTIFACT_DIR}/P1_M12_PHASE_KERNEL_ATTRIBUTION_${TS}.txt"

CLIENT="${SRC_DIR}/p1_m12_workload.py"
PARSER="${SRC_DIR}/p1_m12_trace_parser.py"
COMPARE="${SRC_DIR}/p1_m12_compare.py"

PREFILL_PROFILE="${RESULT_DIR}/prefill_profile_${TS}"
DECODE_PROFILE="${RESULT_DIR}/decode_profile_${TS}"

PREFILL_META="${RESULT_DIR}/P1_M12_PREFILL_META_${TS}.json"
DECODE_META="${RESULT_DIR}/P1_M12_DECODE_META_${TS}.json"

PREFILL_SUMMARY="${RESULT_DIR}/P1_M12_PREFILL_KERNELS_${TS}.json"
DECODE_SUMMARY="${RESULT_DIR}/P1_M12_DECODE_KERNELS_${TS}.json"

PREFILL_LOG="${RESULT_DIR}/P1_M12_PREFILL_SERVER_${TS}.log"
DECODE_LOG="${RESULT_DIR}/P1_M12_DECODE_SERVER_${TS}.log"

CSV="${RESULT_DIR}/P1_M12_PHASE_COMPARISON_${TS}.csv"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}" \
    "${PREFILL_PROFILE}" \
    "${DECODE_PROFILE}"

SERVER_PID=""

LINE="========================================================================================================================"


cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "STOPPING_VLLM_SERVER = ${SERVER_PID}"

        kill -TERM -- "-${SERVER_PID}" 2>/dev/null \
            || kill -TERM "${SERVER_PID}" 2>/dev/null \
            || true

        for _ in $(seq 1 30); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL -- "-${SERVER_PID}" 2>/dev/null \
                || kill -KILL "${SERVER_PID}" 2>/dev/null \
                || true
        fi
    fi

    SERVER_PID=""
    sleep 2
}


create_client() {

cat > "${CLIENT}" <<'PY'
import concurrent.futures
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M12_HOST"]
PORT = int(os.environ["P1_M12_PORT"])
MODEL = os.environ["P1_M12_MODEL"]
MODEL_PATH = os.environ["P1_M12_MODEL_PATH"]

PHASE = os.environ["P1_M12_PHASE"]
META_PATH = Path(os.environ["P1_M12_META"])

BASE = f"http://{HOST}:{PORT}"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


def chat_tokens(text):
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": text,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return len(
        tokenizer.encode(
            rendered,
            add_special_tokens=False,
        )
    )


def output_token_count(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_prompt(
    target_tokens,
    long_output,
):
    prefix = (
        f"Unique request {uuid.uuid4().hex}. "
        "Analyze this synthetic AI infrastructure workload. "
    )

    filler = (
        "GPU inference scheduling KV cache attention "
        "memory bandwidth continuous batching kernel "
        "prefill decode throughput latency tensor compute. "
    )

    if long_output:
        suffix = (
            "Give a long detailed technical explanation. "
            "Continue discussing different systems aspects "
            "until the output limit is reached."
        )
    else:
        suffix = (
            "Give one very concise technical explanation."
        )

    lo = 0
    hi = 32

    while chat_tokens(
        prefix + filler * hi + suffix
    ) < target_tokens:

        hi *= 2

        if hi > 16384:
            raise RuntimeError(
                "Prompt construction runaway."
            )

    while lo < hi:
        mid = (lo + hi) // 2

        text = (
            prefix
            + filler * mid
            + suffix
        )

        if chat_tokens(text) < target_tokens:
            lo = mid + 1
        else:
            hi = mid

    candidates = []

    for reps in [
        max(0, lo - 2),
        max(0, lo - 1),
        lo,
        lo + 1,
    ]:
        text = (
            prefix
            + filler * reps
            + suffix
        )

        count = chat_tokens(text)

        candidates.append(
            (
                abs(count - target_tokens),
                count,
                text,
            )
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return candidates[0][2]


def request_once(
    prompt,
    max_tokens,
):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }

    start = time.perf_counter()

    with httpx.Client(
        trust_env=False,
        timeout=300.0,
    ) as client:

        response = client.post(
            BASE + "/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        obj = response.json()

    elapsed = (
        time.perf_counter()
        - start
    )

    usage = obj.get(
        "usage",
        {},
    )

    content = (
        obj.get(
            "choices",
            [{}],
        )[0]
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
    )

    prompt_tokens = int(
        usage.get(
            "prompt_tokens",
            chat_tokens(prompt),
        )
    )

    completion_tokens = int(
        usage.get(
            "completion_tokens",
            output_token_count(content),
        )
    )

    return {
        "latency_ms": elapsed * 1000.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def run_workload():

    if PHASE == "prefill":

        target_prompt = 2048
        output_target = 16
        concurrency = 8
        request_count = 16
        long_output = False

    elif PHASE == "decode":

        target_prompt = 128
        output_target = 512
        concurrency = 8
        request_count = 8
        long_output = True

    else:
        raise RuntimeError(
            f"Unknown PHASE={PHASE}"
        )

    prompts = [
        build_prompt(
            target_prompt,
            long_output,
        )
        for _ in range(
            request_count
        )
    ]

    print(
        "PHASE =",
        PHASE,
    )

    print(
        "TARGET_PROMPT_TOKENS =",
        target_prompt,
    )

    print(
        "TARGET_OUTPUT_TOKENS =",
        output_target,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "REQUEST_COUNT =",
        request_count,
    )

    start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                prompt,
                output_target,
            )
            for prompt in prompts
        ]

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            results.append(
                future.result()
            )

    wall = (
        time.perf_counter()
        - start
    )

    prompt_total = sum(
        x["prompt_tokens"]
        for x in results
    )

    output_total = sum(
        x["output_tokens"]
        for x in results
    )

    latencies = [
        x["latency_ms"]
        for x in results
    ]

    meta = {
        "phase": PHASE,
        "target_prompt_tokens": (
            target_prompt
        ),
        "target_output_tokens": (
            output_target
        ),
        "concurrency": concurrency,
        "request_count": (
            request_count
        ),
        "wall_s": wall,
        "prompt_tokens_total": (
            prompt_total
        ),
        "output_tokens_total": (
            output_total
        ),
        "prompt_tokens_median": (
            statistics.median(
                [
                    x["prompt_tokens"]
                    for x in results
                ]
            )
        ),
        "output_tokens_median": (
            statistics.median(
                [
                    x["output_tokens"]
                    for x in results
                ]
            )
        ),
        "latency_median_ms": (
            statistics.median(
                latencies
            )
        ),
        "output_tokens_per_s_profiled": (
            output_total / wall
        ),
    }

    META_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    META_PATH.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    for k, v in meta.items():

        if isinstance(
            v,
            float,
        ):
            print(
                f"{k} = {v:.6f}"
            )
        else:
            print(
                f"{k} = {v}"
            )

    print(
        "PROFILE_WORKLOAD = SUCCESS"
    )


print("=" * 120)
print(
    "P1-M12 PHASE-SEPARATED "
    "PRODUCTION WORKLOAD"
)
print("=" * 120)

run_workload()
PY
}


create_parser() {

cat > "${PARSER}" <<'PY'
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
PY
}


create_compare() {

cat > "${COMPARE}" <<'PY'
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
PY
}


start_profile_server() {
    PHASE="$1"
    PROFILE_DIR="$2"
    LOG="$3"
    META="$4"
    SUMMARY="$5"

    cleanup_server

    pkill -f "${ENV}/bin/vllm serve" \
        2>/dev/null || true

    pkill -f "vllm.*EngineCore" \
        2>/dev/null || true

    sleep 2

    rm -rf "${PROFILE_DIR}"
    mkdir -p "${PROFILE_DIR}"

    echo
    echo "${LINE}"
    echo "START ${PHASE^^} PROFILING SERVER"
    echo "${LINE}"

    PROFILER_CONFIG="$(
        "${PYTHON}" - \
            "${PROFILE_DIR}" \
        <<'PY'
import json
import sys

profile_dir = sys.argv[1]

print(
    json.dumps(
        {
            "profiler": "torch",

            "torch_profiler_dir": (
                profile_dir
            ),

            "torch_profiler_with_stack": False,

            "torch_profiler_record_shapes": False,

            "torch_profiler_with_memory": False,

            "torch_profiler_use_gzip": True,

            "torch_profiler_dump_cuda_time_total": True,

            # Decode-heavy run may require hundreds
            # of engine iterations.
            "active_iterations": 1024,
        },
        separators=(",", ":"),
    )
)
PY
    )"

    echo "PROFILER_CONFIG = ${PROFILER_CONFIG}"

    setsid \
    env \
        -u ALL_PROXY \
        -u all_proxy \
        -u HTTP_PROXY \
        -u HTTPS_PROXY \
        -u http_proxy \
        -u https_proxy \
        PATH="${PATH}" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        HF_HOME="${HF_HOME}" \
    "${VLLM}" serve \
        "${MODEL_PATH}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name "${MODEL_NAME}" \
        --dtype float16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.70 \
        --generation-config vllm \
        --max-num-seqs 32 \
        --profiler-config "${PROFILER_CONFIG}" \
        > "${LOG}" 2>&1 &

    SERVER_PID=$!

    echo "SERVER_PID = ${SERVER_PID}"
    echo "PROFILE_DIR = ${PROFILE_DIR}"
    echo "SERVER_LOG = ${LOG}"

    READY="FALSE"

    for SECOND in $(seq 1 900); do

        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then

            echo "SERVER_EXITED_EARLY = TRUE"

            cat "${LOG}" || true

            return 10
        fi

        CODE="$(
            curl \
                --noproxy '*' \
                -sS \
                -o /dev/null \
                -w '%{http_code}' \
                --connect-timeout 2 \
                --max-time 3 \
                "http://${HOST}:${PORT}/health" \
            2>/dev/null \
            || true
        )"

        if [ "${CODE}" = "200" ]; then

            READY="TRUE"

            echo "SERVER_HEALTH = READY"
            echo "SECONDS_WAITED = ${SECOND}"

            break
        fi

        if [ $((SECOND % 60)) -eq 0 ]; then

            echo "WAITING_SECOND = ${SECOND}"

            tail -n 40 \
                "${LOG}" \
                || true
        fi

        sleep 1
    done


    if [ "${READY}" != "TRUE" ]; then
        echo "SERVER_HEALTH = TIMEOUT"
        return 11
    fi


    # ----------------------------------------------------------------------------------------------
    # Warm-up outside profiler.
    # ----------------------------------------------------------------------------------------------

    echo
    echo "PRE_PROFILE_WARMUP = ${PHASE}"

    P1_M12_HOST="${HOST}" \
    P1_M12_PORT="${PORT}" \
    P1_M12_MODEL="${MODEL_NAME}" \
    P1_M12_MODEL_PATH="${MODEL_PATH}" \
    P1_M12_PHASE="${PHASE}" \
    P1_M12_META="${META}.warmup" \
    "${PYTHON}" "${CLIENT}"

    echo "PRE_PROFILE_WARMUP = SUCCESS"


    # ----------------------------------------------------------------------------------------------
    # Start profile.
    # ----------------------------------------------------------------------------------------------

    START_CODE="$(
        curl \
            --noproxy '*' \
            -sS \
            -o /dev/null \
            -w '%{http_code}' \
            -X POST \
            "http://${HOST}:${PORT}/start_profile" \
        || true
    )"

    echo "START_PROFILE_HTTP = ${START_CODE}"

    if [ "${START_CODE}" != "200" ]; then
        echo "ERROR = START_PROFILE_FAILED"
        return 12
    fi


    # ----------------------------------------------------------------------------------------------
    # Production phase-emphasis workload.
    # ----------------------------------------------------------------------------------------------

    echo
    echo "${LINE}"
    echo "PROFILED WORKLOAD = ${PHASE}"
    echo "${LINE}"

    P1_M12_HOST="${HOST}" \
    P1_M12_PORT="${PORT}" \
    P1_M12_MODEL="${MODEL_NAME}" \
    P1_M12_MODEL_PATH="${MODEL_PATH}" \
    P1_M12_PHASE="${PHASE}" \
    P1_M12_META="${META}" \
    "${PYTHON}" "${CLIENT}"


    # ----------------------------------------------------------------------------------------------
    # Stop.
    # ----------------------------------------------------------------------------------------------

    STOP_CODE="$(
        curl \
            --noproxy '*' \
            -sS \
            -o /dev/null \
            -w '%{http_code}' \
            -X POST \
            "http://${HOST}:${PORT}/stop_profile" \
        || true
    )"

    echo "STOP_PROFILE_HTTP = ${STOP_CODE}"

    if [ "${STOP_CODE}" != "200" ]; then
        echo "ERROR = STOP_PROFILE_FAILED"
        return 13
    fi

    sleep 5


    TRACE_COUNT="$(
        find "${PROFILE_DIR}" \
            -type f \
            \( \
                -name '*.json' \
                -o \
                -name '*.json.gz' \
            \) \
        | wc -l
    )"

    echo "TRACE_FILE_COUNT = ${TRACE_COUNT}"

    if [ "${TRACE_COUNT}" -eq 0 ]; then

        echo "ERROR = NO_TRACE_FILES"

        grep -Ei \
            'profil|trace|torch' \
            "${LOG}" \
            | tail -n 200 \
            || true

        return 14
    fi


    echo
    echo "${LINE}"
    echo "PARSE ${PHASE^^} TRACE"
    echo "${LINE}"

    "${PYTHON}" \
        "${PARSER}" \
        "${PHASE}" \
        "${PROFILE_DIR}" \
        "${META}" \
        "${SUMMARY}"


    echo
    echo "${PHASE^^} SERVER PROFILE EVIDENCE:"

    grep -Ei \
        'profil|torch|cudagraph|compile|jit' \
        "${LOG}" \
        | tail -n 120 \
        || true


    cleanup_server

    echo "PHASE_${PHASE^^}_COMPLETE = TRUE"
}


main() {
    trap cleanup_server EXIT
    set -e

    export CUDA_HOME="/usr/local/cuda-12.8"

    export PATH="${CUDA_HOME}/bin:${ENV}/bin:${PATH}"

    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M12"
    echo "PHASE-SEPARATED PRODUCTION KERNEL ATTRIBUTION"
    echo "${LINE}"

    echo
    echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"


    echo
    echo "${LINE}"
    echo "1. ENVIRONMENT"
    echo "${LINE}"

    "${PYTHON}" - <<'PY'
import importlib.metadata
import torch

print(
    "vllm =",
    importlib.metadata.version(
        "vllm"
    ),
)

print(
    "torch =",
    torch.__version__,
)

print(
    "torch_cuda =",
    torch.version.cuda,
)

print(
    "CUDA_AVAILABLE =",
    torch.cuda.is_available(),
)

assert (
    importlib.metadata.version(
        "vllm"
    )
    == "0.27.1+cu129"
)

assert torch.cuda.is_available()

print(
    "GPU =",
    torch.cuda.get_device_name(0),
)

print(
    "TORCH_PROFILER_AVAILABLE =",
    hasattr(
        torch,
        "profiler",
    ),
)

print(
    "ENVIRONMENT = HEALTHY"
)
PY


    echo
    echo "NCU_COUNTER_STATUS = BLOCKED_BY_HOST_ERR_NVGPUCTRPERM"

    echo "P1_M12_PROFILING_PATH = PYTORCH_KINETO"


    echo
    echo "${LINE}"
    echo "2. LOCAL MODEL"
    echo "${LINE}"

    MODEL_PATH_FILE="$(
        find \
            "${PROJECT}/results/p1_m6" \
            -maxdepth 1 \
            -type f \
            -name 'P1_M6_MODEL_PATH_*.txt' \
            -printf '%T@ %p\n' \
            2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"

    if [ -z "${MODEL_PATH_FILE}" ] \
        || [ ! -f "${MODEL_PATH_FILE}" ]
    then
        echo "ERROR = MODEL_PATH_FILE_NOT_FOUND"
        return 2
    fi

    MODEL_PATH="$(
        cat "${MODEL_PATH_FILE}"
    )"

    if [ -z "${MODEL_PATH}" ] \
        || [ ! -d "${MODEL_PATH}" ]
    then
        echo "ERROR = MODEL_PATH_INVALID"
        return 2
    fi

    echo "MODEL_PATH = ${MODEL_PATH}"


    echo
    echo "${LINE}"
    echo "3. CREATE CLIENT / PARSER"
    echo "${LINE}"

    create_client
    create_parser
    create_compare

    echo "CLIENT = ${CLIENT}"
    echo "PARSER = ${PARSER}"
    echo "COMPARE = ${COMPARE}"


    echo
    echo "${LINE}"
    echo "4. PREFILL-HEAVY PRODUCTION TRACE"
    echo "${LINE}"

    start_profile_server \
        "prefill" \
        "${PREFILL_PROFILE}" \
        "${PREFILL_LOG}" \
        "${PREFILL_META}" \
        "${PREFILL_SUMMARY}"


    echo
    echo "${LINE}"
    echo "5. DECODE-HEAVY PRODUCTION TRACE"
    echo "${LINE}"

    start_profile_server \
        "decode" \
        "${DECODE_PROFILE}" \
        "${DECODE_LOG}" \
        "${DECODE_META}" \
        "${DECODE_SUMMARY}"


    echo
    echo "${LINE}"
    echo "6. CROSS-PHASE COMPARISON"
    echo "${LINE}"

    "${PYTHON}" \
        "${COMPARE}" \
        "${PREFILL_SUMMARY}" \
        "${DECODE_SUMMARY}" \
        "${CSV}"


    echo
    echo "${LINE}"
    echo "7. STRUCTURED CSV"
    echo "${LINE}"

    cat "${CSV}"


    echo
    echo "${LINE}"
    echo "FINAL VERDICT"
    echo "${LINE}"

    echo "PREFILL_PRODUCTION_TRACE = SUCCESS"
    echo "DECODE_PRODUCTION_TRACE = SUCCESS"
    echo "OLD_M11B_ATTRIBUTION_BUG = CORRECTED"
    echo "KV_CACHE_PRIORITY_CLASSIFICATION = ENABLED"
    echo "OVERLAP_AUDIT = SUCCESS"
    echo "PHASE_SEPARATED_ATTRIBUTION = SUCCESS"
    echo "HARDWARE_COUNTER_LIMITATION = ERR_NVGPUCTRPERM"

    echo "VERDICT = READY_FOR_P1_M13_TARGETED_OPTIMIZATION"


    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "PREFILL_PROFILE_DIR = ${PREFILL_PROFILE}"
    echo "DECODE_PROFILE_DIR = ${DECODE_PROFILE}"

    echo "PREFILL_SUMMARY = ${PREFILL_SUMMARY}"
    echo "DECODE_SUMMARY = ${DECODE_SUMMARY}"

    echo "PREFILL_META = ${PREFILL_META}"
    echo "DECODE_META = ${DECODE_META}"

    echo "CSV_RESULT = ${CSV}"

    echo "CLIENT = ${CLIENT}"
    echo "PARSER = ${PARSER}"
    echo "COMPARE = ${COMPARE}"


    echo
    echo "${LINE}"
    echo "END OF P1-M12"
    echo "${LINE}"

    cleanup_server

    trap - EXIT
}


set +e

main 2>&1 | tee "${TXT}"

STATUS=${PIPESTATUS[0]}

set -e

printf '%s\n' "${TXT}"

exit "${STATUS}"
