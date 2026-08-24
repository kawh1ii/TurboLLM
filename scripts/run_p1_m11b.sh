#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"
VLLM="${ENV}/bin/vllm"

HF_HOME="${BASE}/caches/huggingface"

SRC_DIR="${PROJECT}/src/p1_m11b"
RESULT_DIR="${PROJECT}/results/p1_m11b"
ARTIFACT_DIR="${BASE}/artifacts"

HOST="127.0.0.1"
PORT="8000"
MODEL_NAME="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

PROFILE_DIR="${RESULT_DIR}/torch_profile_${TS}"
SERVER_LOG="${RESULT_DIR}/P1_M11B_SERVER_${TS}.log"
SUMMARY="${RESULT_DIR}/P1_M11B_KERNEL_SUMMARY_${TS}.txt"
CLIENT="${SRC_DIR}/p1_m11b_workload.py"
PARSER="${SRC_DIR}/p1_m11b_parse_trace.py"

TXT="${ARTIFACT_DIR}/P1_M11B_VLLM_TORCH_PROFILER_${TS}.txt"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${PROFILE_DIR}" \
    "${ARTIFACT_DIR}"

SERVER_PID=""

LINE="========================================================================================================================"


cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] \
        && kill -0 "${SERVER_PID}" 2>/dev/null
    then
        echo "STOPPING_VLLM_SERVER = TRUE"

        kill -TERM -- "-${SERVER_PID}" \
            2>/dev/null || \
        kill -TERM "${SERVER_PID}" \
            2>/dev/null || true

        for _ in $(seq 1 20); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL -- "-${SERVER_PID}" \
                2>/dev/null || \
            kill -KILL "${SERVER_PID}" \
                2>/dev/null || true
        fi
    fi

    SERVER_PID=""
}


main() {
    trap cleanup_server EXIT
    set -e

    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="${CUDA_HOME}/bin:${ENV}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M11B"
    echo "VLLM PRODUCTION KERNEL TRACING WITH PYTORCH PROFILER"
    echo "${LINE}"

    echo
    echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"


    # =================================================================================================
    # 1. RECORD P1-M11A LIMITATION
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "1. P1-M11A HARDWARE-COUNTER STATUS"
    echo "${LINE}"

    echo "NSIGHT_COMPUTE = AVAILABLE"
    echo "CUDA_NATIVE_KERNEL_EXECUTION = SUCCESS"
    echo "NCU_HARDWARE_COUNTERS = BLOCKED_BY_HOST"
    echo "NCU_ERROR = ERR_NVGPUCTRPERM"

    echo
    echo "IMPORTANT:"
    echo "The host blocks privileged NVIDIA hardware performance counters."
    echo "This does NOT prevent ordinary CUDA execution."
    echo "This does NOT imply a CUDA/vLLM failure."
    echo "P1-M11B therefore uses vLLM's supported PyTorch Profiler path."


    # =================================================================================================
    # 2. ENVIRONMENT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "2. ENVIRONMENT"
    echo "${LINE}"

    "${PYTHON}" - <<'PY'
import importlib.metadata
import torch

print("vllm =", importlib.metadata.version("vllm"))
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())

assert importlib.metadata.version("vllm") == "0.27.1+cu129"
assert torch.cuda.is_available()

print("GPU =", torch.cuda.get_device_name(0))

print(
    "TORCH_PROFILER_AVAILABLE =",
    hasattr(torch, "profiler"),
)

assert hasattr(torch, "profiler")

print("ENVIRONMENT = HEALTHY")
PY

    echo
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.free \
        --format=csv,noheader


    # =================================================================================================
    # 3. VERIFY PROFILER CLI
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "3. VERIFY VLLM PROFILER SUPPORT"
    echo "${LINE}"

    HELP="$(
        NO_COLOR=1 \
        "${VLLM}" serve --help=all 2>&1
    )"

    if echo "${HELP}" \
        | grep -q -- "--profiler-config"
    then
        echo "VLLM_PROFILER_CONFIG = SUPPORTED"
    else
        echo "VLLM_PROFILER_CONFIG = NOT_SUPPORTED"
        return 2
    fi


    # =================================================================================================
    # 4. LOCAL MODEL
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "4. LOCAL MODEL"
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
        return 3
    fi

    MODEL_PATH="$(cat "${MODEL_PATH_FILE}")"

    if [ -z "${MODEL_PATH}" ] \
        || [ ! -d "${MODEL_PATH}" ]
    then
        echo "ERROR = MODEL_PATH_INVALID"
        return 3
    fi

    echo "MODEL_PATH = ${MODEL_PATH}"


    # =================================================================================================
    # 5. CREATE PROFILE WORKLOAD
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "5. CREATE PROFILE WORKLOAD CLIENT"
    echo "${LINE}"

    cat > "${CLIENT}" <<'PY'
import concurrent.futures
import json
import os
import time

import httpx


HOST = os.environ["P1_M11B_HOST"]
PORT = int(os.environ["P1_M11B_PORT"])
MODEL = os.environ["P1_M11B_MODEL"]

BASE = f"http://{HOST}:{PORT}"


def request_once(i, max_tokens=64):
    # Unique text near the beginning avoids turning the
    # experiment into a prefix-cache benchmark.
    content = (
        f"Unique profiling request {i}. "
        "Analyze GPU inference serving architecture. "
        + (
            "Discuss scheduler KV cache attention kernels "
            "memory bandwidth batching prefill decode latency "
            "throughput CUDA graph tensor computation. "
            * 25
        )
        + "Give a detailed technical answer."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
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

    elapsed = time.perf_counter() - start

    usage = obj.get("usage", {})

    return {
        "request": i,
        "latency_ms": elapsed * 1000.0,
        "prompt_tokens": usage.get(
            "prompt_tokens",
            0,
        ),
        "output_tokens": usage.get(
            "completion_tokens",
            0,
        ),
    }


def run_batch(
    base_id,
    concurrency,
    count,
    output_tokens,
):
    start = time.perf_counter()

    rows = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                base_id + i,
                output_tokens,
            )
            for i in range(count)
        ]

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            rows.append(
                future.result()
            )

    wall = time.perf_counter() - start

    output_total = sum(
        x["output_tokens"]
        for x in rows
    )

    prompt_total = sum(
        x["prompt_tokens"]
        for x in rows
    )

    print(
        "REQUEST_COUNT =",
        count,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "WALL_S =",
        f"{wall:.6f}",
    )

    print(
        "PROMPT_TOKENS =",
        prompt_total,
    )

    print(
        "OUTPUT_TOKENS =",
        output_total,
    )

    print(
        "OUTPUT_TOKENS_PER_S =",
        f"{output_total / wall:.3f}",
    )


print("=" * 120)
print("P1-M11B PRODUCTION VLLM WORKLOAD")
print("=" * 120)

# Moderate concurrency is enough to expose real
# batched attention / GEMM / normalization / sampling kernels.
run_batch(
    base_id=100000,
    concurrency=8,
    count=16,
    output_tokens=64,
)

print("PROFILE_WORKLOAD = SUCCESS")
PY

    echo "WORKLOAD_CLIENT = ${CLIENT}"


    # =================================================================================================
    # 6. CREATE TRACE PARSER
    # =================================================================================================

    cat > "${PARSER}" <<'PY'
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
PY

    echo "TRACE_PARSER = ${PARSER}"


    # =================================================================================================
    # 7. CLEAN OLD SERVER
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "7. CLEAN STALE VLLM SERVER"
    echo "${LINE}"

    pkill -f "${ENV}/bin/vllm serve" \
        2>/dev/null || true

    pkill -f "vllm.*EngineCore" \
        2>/dev/null || true

    sleep 2

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free \
        --format=csv,noheader


    # =================================================================================================
    # 8. START SERVER WITH TORCH PROFILER ENABLED
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "8. START VLLM WITH TORCH PROFILER"
    echo "${LINE}"

    PROFILER_CONFIG="$(
        "${PYTHON}" - \
            "${PROFILE_DIR}" \
        <<'PY'
import json
import sys

profile_dir = sys.argv[1]

config = {
    "profiler": "torch",

    "torch_profiler_dir": (
        profile_dir
    ),

    # Keep profiling overhead controlled.
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,

    "torch_profiler_use_gzip": True,

    "torch_profiler_dump_cuda_time_total": True,

    # Enough engine iterations for the bounded workload.
    "active_iterations": 64,
}

print(
    json.dumps(
        config,
        separators=(",", ":"),
    )
)
PY
    )"

    echo "PROFILER_CONFIG = ${PROFILER_CONFIG}"

    rm -f "${SERVER_LOG}"

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
        > "${SERVER_LOG}" 2>&1 &

    SERVER_PID=$!

    echo "SERVER_PID = ${SERVER_PID}"
    echo "SERVER_LOG = ${SERVER_LOG}"
    echo "PROFILE_DIR = ${PROFILE_DIR}"


    # =================================================================================================
    # 9. WAIT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "9. WAIT FOR SERVER"
    echo "${LINE}"

    READY="FALSE"

    for SECOND in $(seq 1 900); do

        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "SERVER_EXITED_EARLY = TRUE"

            cat "${SERVER_LOG}" || true

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

            tail -n 50 \
                "${SERVER_LOG}" \
                || true
        fi

        sleep 1
    done


    if [ "${READY}" != "TRUE" ]; then
        echo "SERVER_HEALTH = TIMEOUT"

        return 11
    fi


    # =================================================================================================
    # 10. VERIFY PROFILING ENDPOINTS
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "10. PROFILING API PRECHECK"
    echo "${LINE}"

    # Warmup BEFORE profiling so compilation/cold start is
    # not confused with steady-state kernel execution.
    echo "PRE_PROFILE_WARMUP:"

    P1_M11B_HOST="${HOST}" \
    P1_M11B_PORT="${PORT}" \
    P1_M11B_MODEL="${MODEL_NAME}" \
    "${PYTHON}" "${CLIENT}"

    echo "PRE_PROFILE_WARMUP = SUCCESS"


    # =================================================================================================
    # 11. START PROFILE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "11. START TORCH PROFILE"
    echo "${LINE}"

    START_BODY="${RESULT_DIR}/start_profile_${TS}.txt"

    START_CODE="$(
        curl \
            --noproxy '*' \
            -sS \
            -o "${START_BODY}" \
            -w '%{http_code}' \
            -X POST \
            "http://${HOST}:${PORT}/start_profile" \
        || true
    )"

    echo "START_PROFILE_HTTP = ${START_CODE}"

    cat "${START_BODY}" || true
    echo

    if [ "${START_CODE}" != "200" ]; then
        echo "ERROR = START_PROFILE_FAILED"

        return 12
    fi


    # =================================================================================================
    # 12. PROFILE REAL WORKLOAD
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "12. RUN PROFILED PRODUCTION WORKLOAD"
    echo "${LINE}"

    P1_M11B_HOST="${HOST}" \
    P1_M11B_PORT="${PORT}" \
    P1_M11B_MODEL="${MODEL_NAME}" \
    "${PYTHON}" "${CLIENT}"


    # =================================================================================================
    # 13. STOP PROFILE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "13. STOP TORCH PROFILE"
    echo "${LINE}"

    STOP_BODY="${RESULT_DIR}/stop_profile_${TS}.txt"

    STOP_CODE="$(
        curl \
            --noproxy '*' \
            -sS \
            -o "${STOP_BODY}" \
            -w '%{http_code}' \
            -X POST \
            "http://${HOST}:${PORT}/stop_profile" \
        || true
    )"

    echo "STOP_PROFILE_HTTP = ${STOP_CODE}"

    cat "${STOP_BODY}" || true
    echo

    if [ "${STOP_CODE}" != "200" ]; then
        echo "ERROR = STOP_PROFILE_FAILED"

        return 13
    fi


    # Flush trace files.
    sleep 10


    # =================================================================================================
    # 14. TRACE FILES
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "14. GENERATED PROFILER ARTIFACTS"
    echo "${LINE}"

    find "${PROFILE_DIR}" \
        -type f \
        -printf '%s bytes  %p\n' \
        | sort \
        || true

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

    echo
    echo "TRACE_FILE_COUNT = ${TRACE_COUNT}"


    if [ "${TRACE_COUNT}" -eq 0 ]; then

        echo
        echo "SERVER PROFILE LOG EVIDENCE:"

        grep -Ei \
            'profil|kineto|trace|torch' \
            "${SERVER_LOG}" \
            | tail -n 200 \
            || true

        echo "ERROR = NO_PROFILER_TRACE"

        return 14
    fi


    # =================================================================================================
    # 15. PARSE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "15. PRODUCTION CUDA KERNEL ANALYSIS"
    echo "${LINE}"

    "${PYTHON}" \
        "${PARSER}" \
        "${PROFILE_DIR}" \
        | tee "${SUMMARY}"


    # =================================================================================================
    # 16. SERVER LOG
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "16. SERVER PROFILER EVIDENCE"
    echo "${LINE}"

    grep -Ei \
        'profil|trace|torch|cudagraph|compile|JIT' \
        "${SERVER_LOG}" \
        | tail -n 250 \
        || true


    # =================================================================================================
    # 17. FINAL
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "P1-M11B KNOWLEDGE CHECKPOINT"
    echo "${LINE}"

    echo "
P1-M11A established that the AutoDL host does not expose
privileged NVIDIA hardware performance counters.

P1-M11B therefore profiles the production vLLM execution
path using PyTorch/Kineto tracing.

VALID EVIDENCE ON THIS HOST:

    application latency / throughput
    scheduler metrics
    KV cache metrics
    GPU telemetry
    CUDA kernel names
    CUDA kernel launch counts
    CUDA kernel durations
    CPU operator timing

NOT AVAILABLE ON THIS HOST:

    Nsight Compute DRAM hardware counters
    Tensor Core counter utilization
    achieved occupancy counters
    detailed SM hardware counter breakdown

This boundary must be preserved when interpreting results.
"


    echo
    echo "${LINE}"
    echo "FINAL VERDICT"
    echo "${LINE}"

    echo "P1_M11A_NCU_ATTEMPT = COMPLETE"
    echo "NCU_COUNTER_PERMISSION = BLOCKED_BY_HOST"
    echo "PYTORCH_PROFILER = SUCCESS"
    echo "VLLM_PRODUCTION_TRACE = SUCCESS"
    echo "CUDA_KERNEL_TIMING = SUCCESS"
    echo "KERNEL_ATTRIBUTION = SUCCESS"

    echo "VERDICT = READY_FOR_P1_M12_KERNEL_ATTRIBUTION_AND_OPTIMIZATION"

    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "PROFILE_DIR = ${PROFILE_DIR}"
    echo "TRACE_SUMMARY = ${SUMMARY}"
    echo "WORKLOAD_CLIENT = ${CLIENT}"
    echo "TRACE_PARSER = ${PARSER}"
    echo "SERVER_LOG = ${SERVER_LOG}"

    echo
    echo "${LINE}"
    echo "END OF P1-M11B"
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
