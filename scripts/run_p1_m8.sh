#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

VLLM_ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${VLLM_ENV}/bin/python"
VLLM="${VLLM_ENV}/bin/vllm"

SRC_DIR="${PROJECT}/src/p1_m8"
RESULT_DIR="${PROJECT}/results/p1_m8"
ARTIFACT_DIR="${BASE}/artifacts"

HF_HOME="${BASE}/caches/huggingface"

HOST="127.0.0.1"
PORT="8000"
MODEL_NAME="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M8_OPTIMIZED_VS_EAGER_${TS}.txt"
CSV="${RESULT_DIR}/P1_M8_COMPARISON_${TS}.csv"

CLIENT="${SRC_DIR}/p1_m8_client.py"
COMPARE="${SRC_DIR}/p1_m8_compare.py"

EAGER_LOG="${RESULT_DIR}/P1_M8_EAGER_SERVER_${TS}.log"
OPT_LOG="${RESULT_DIR}/P1_M8_OPTIMIZED_SERVER_${TS}.log"

EAGER_METRICS="${RESULT_DIR}/P1_M8_EAGER_METRICS_${TS}.txt"
OPT_METRICS="${RESULT_DIR}/P1_M8_OPTIMIZED_METRICS_${TS}.txt"

SERVER_PID=""

LINE="========================================================================================================================"


cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] \
        && kill -0 "${SERVER_PID}" 2>/dev/null
    then

        echo "STOPPING_SERVER_PID = ${SERVER_PID}"

        kill -TERM -- "-${SERVER_PID}" \
            2>/dev/null || \
        kill -TERM "${SERVER_PID}" \
            2>/dev/null || true

        for _ in $(seq 1 30); do
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

    sleep 2
}


main() {
    trap cleanup_server EXIT
    set -e

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M8"
    echo "OPTIMIZED VLLM VS EAGER VLLM"
    echo "${LINE}"

    echo
    echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"


    # =================================================================================================
    # 1. ENVIRONMENT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "1. ENVIRONMENT"
    echo "${LINE}"

    export PATH="${VLLM_ENV}/bin:/usr/local/cuda-12.8/bin:${PATH}"

    "${PYTHON}" - <<'PY'
import importlib.metadata
import shutil
import torch

print("vllm =", importlib.metadata.version("vllm"))
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())
print("ninja =", shutil.which("ninja"))

assert importlib.metadata.version("vllm") == "0.27.1+cu129"
assert torch.cuda.is_available()
assert str(torch.version.cuda).startswith("12.9")
assert shutil.which("ninja") is not None

print("GPU =", torch.cuda.get_device_name(0))
print("ENVIRONMENT = HEALTHY")
PY

    echo
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit \
        --format=csv,noheader


    # =================================================================================================
    # 2. LOCAL MODEL
    # =================================================================================================

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

    MODEL_PATH="$(cat "${MODEL_PATH_FILE}")"

    if [ -z "${MODEL_PATH}" ] \
        || [ ! -d "${MODEL_PATH}" ]
    then
        echo "ERROR = MODEL_PATH_INVALID"
        return 2
    fi

    echo "MODEL_PATH = ${MODEL_PATH}"


    # =================================================================================================
    # 3. CLIENT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "3. CREATE MATCHED BENCHMARK CLIENT"
    echo "${LINE}"

    cat > "${CLIENT}" <<'PY'
import concurrent.futures
import csv
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M8_HOST"]
PORT = int(os.environ["P1_M8_PORT"])
MODEL = os.environ["P1_M8_MODEL"]
MODEL_PATH = os.environ["P1_M8_MODEL_PATH"]

MODE = os.environ["P1_M8_MODE"]

CSV_PATH = Path(
    os.environ["P1_M8_CSV"]
)

STARTUP_S = float(
    os.environ["P1_M8_STARTUP_S"]
)

GPU_MEMORY_MIB = float(
    os.environ["P1_M8_GPU_MEMORY_MIB"]
)

BASE = f"http://{HOST}:{PORT}"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


def percentile(values, q):
    xs = sorted(values)

    if not xs:
        return float("nan")

    pos = (len(xs) - 1) * q

    lo = int(pos)
    hi = min(
        lo + 1,
        len(xs) - 1,
    )

    frac = pos - lo

    return (
        xs[lo] * (1 - frac)
        + xs[hi] * frac
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


def completion_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_prompt(
    target=512,
    request_id=0,
):
    # Put uniqueness near the beginning so ordinary
    # prefix-cache reuse does not dominate this test.
    prefix = (
        f"Unique request {request_id:06d}. "
        "Analyze the following synthetic AI infrastructure context. "
    )

    filler = (
        "GPU inference scheduling memory bandwidth KV cache "
        "continuous batching attention latency throughput kernel. "
    )

    suffix = (
        "Give a concise technical explanation."
    )

    lo = 0
    hi = 64

    while (
        chat_tokens(
            prefix
            + filler * hi
            + suffix
        )
        < target
    ):
        hi *= 2

    while lo < hi:
        mid = (lo + hi) // 2

        text = (
            prefix
            + filler * mid
            + suffix
        )

        if chat_tokens(text) < target:
            lo = mid + 1
        else:
            hi = mid

    candidates = []

    for n in [
        max(0, lo - 2),
        max(0, lo - 1),
        lo,
        lo + 1,
    ]:
        text = (
            prefix
            + filler * n
            + suffix
        )

        count = chat_tokens(text)

        candidates.append(
            (
                abs(count - target),
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
    max_tokens=64,
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
        "stream": True,
    }

    start = time.perf_counter()

    first = None

    output = []

    with httpx.Client(
        trust_env=False,
        timeout=180.0,
    ) as client:

        with client.stream(
            "POST",
            BASE
            + "/v1/chat/completions",
            json=payload,
        ) as response:

            if response.status_code != 200:

                body = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

                raise RuntimeError(
                    f"HTTP {response.status_code}: {body}"
                )

            for line in response.iter_lines():

                if not line:
                    continue

                if not line.startswith(
                    "data: "
                ):
                    continue

                body = line[6:].strip()

                if body == "[DONE]":
                    break

                try:
                    obj = json.loads(body)
                except Exception:
                    continue

                choices = obj.get(
                    "choices",
                    [],
                )

                if not choices:
                    continue

                content = (
                    choices[0]
                    .get(
                        "delta",
                        {},
                    )
                    .get(
                        "content",
                        "",
                    )
                )

                if content:

                    if first is None:
                        first = (
                            time.perf_counter()
                            - start
                        )

                    output.append(content)

    total = (
        time.perf_counter()
        - start
    )

    if first is None:
        raise RuntimeError(
            "No streamed output token."
        )

    out_text = "".join(output)

    n_out = max(
        1,
        completion_tokens(out_text),
    )

    ttft_ms = first * 1000.0
    total_ms = total * 1000.0

    if n_out > 1:
        tpot_ms = (
            total_ms - ttft_ms
        ) / (n_out - 1)
    else:
        tpot_ms = float("nan")

    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "tpot_ms": tpot_ms,
        "prompt_tokens": chat_tokens(prompt),
        "output_tokens": n_out,
    }


def execute_group(
    name,
    concurrency,
    count,
    request_id_base,
):
    prompts = [
        build_prompt(
            target=512,
            request_id=(
                request_id_base + i
            ),
        )
        for i in range(count)
    ]

    wall_start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                prompt,
                64,
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
        - wall_start
    )

    ttft = [
        x["ttft_ms"]
        for x in results
    ]

    latency = [
        x["total_ms"]
        for x in results
    ]

    tpot = [
        x["tpot_ms"]
        for x in results
        if x["tpot_ms"] == x["tpot_ms"]
    ]

    prompt_tokens = sum(
        x["prompt_tokens"]
        for x in results
    )

    output_tokens = sum(
        x["output_tokens"]
        for x in results
    )

    row = {
        "mode": MODE,
        "workload": name,
        "startup_s": STARTUP_S,
        "gpu_memory_ready_mib": (
            GPU_MEMORY_MIB
        ),
        "concurrency": concurrency,
        "request_count": count,
        "median_ttft_ms": (
            statistics.median(ttft)
        ),
        "p95_ttft_ms": (
            percentile(
                ttft,
                0.95,
            )
        ),
        "median_tpot_ms": (
            statistics.median(tpot)
        ),
        "median_latency_ms": (
            statistics.median(
                latency
            )
        ),
        "p95_latency_ms": (
            percentile(
                latency,
                0.95,
            )
        ),
        "request_throughput_rps": (
            count / wall
        ),
        "prompt_tokens_per_s": (
            prompt_tokens / wall
        ),
        "output_tokens_per_s": (
            output_tokens / wall
        ),
        "total_tokens_per_s": (
            (
                prompt_tokens
                + output_tokens
            )
            / wall
        ),
    }

    print()
    print("-" * 120)

    for k, v in row.items():
        if isinstance(v, float):
            print(
                f"{k} = {v:.6f}"
            )
        else:
            print(
                f"{k} = {v}"
            )

    return row


print("=" * 120)
print(
    f"P1-M8 MATCHED BENCHMARK — {MODE}"
)
print("=" * 120)

print(
    "STARTUP_S =",
    STARTUP_S,
)

print(
    "GPU_MEMORY_READY_MIB =",
    GPU_MEMORY_MIB,
)


# Warmup each workload shape before measurement.
print()
print("WARMUP")

request_once(
    build_prompt(
        target=512,
        request_id=900001,
    ),
    64,
)

print(
    "WARMUP_SINGLE = SUCCESS"
)

warm_prompts = [
    build_prompt(
        target=512,
        request_id=910000 + i,
    )
    for i in range(16)
]

with concurrent.futures.ThreadPoolExecutor(
    max_workers=16
) as executor:

    futures = [
        executor.submit(
            request_once,
            p,
            64,
        )
        for p in warm_prompts
    ]

    for f in (
        concurrent.futures.as_completed(
            futures
        )
    ):
        f.result()

print(
    "WARMUP_CONCURRENT = SUCCESS"
)


rows = []

rows.append(
    execute_group(
        name="single_c1",
        concurrency=1,
        count=6,
        request_id_base=100000,
    )
)

rows.append(
    execute_group(
        name="batch_c8",
        concurrency=8,
        count=16,
        request_id_base=200000,
    )
)

rows.append(
    execute_group(
        name="batch_c16",
        concurrency=16,
        count=32,
        request_id_base=300000,
    )
)


fields = [
    "mode",
    "workload",
    "startup_s",
    "gpu_memory_ready_mib",
    "concurrency",
    "request_count",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_tpot_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "request_throughput_rps",
    "prompt_tokens_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
]

write_header = (
    not CSV_PATH.exists()
    or CSV_PATH.stat().st_size == 0
)

with CSV_PATH.open(
    "a",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    if write_header:
        writer.writeheader()

    writer.writerows(rows)

print()
print(
    "MODE_BENCHMARK = SUCCESS"
)

print(
    "CSV_RESULT =",
    CSV_PATH,
)
PY


    # =================================================================================================
    # 4. COMPARISON SCRIPT
    # =================================================================================================

    cat > "${COMPARE}" <<'PY'
import csv
import sys
from collections import defaultdict


path = sys.argv[1]

rows = []

with open(
    path,
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


data = defaultdict(dict)

for row in rows:
    data[row["workload"]][
        row["mode"]
    ] = row


print("=" * 120)
print("P1-M8 EAGER VS OPTIMIZED COMPARISON")
print("=" * 120)


for workload in [
    "single_c1",
    "batch_c8",
    "batch_c16",
]:
    eager = data[workload][
        "eager"
    ]

    optimized = data[workload][
        "optimized"
    ]


    def num(row, name):
        return float(
            row[name]
        )


    eager_ttft = num(
        eager,
        "median_ttft_ms",
    )

    opt_ttft = num(
        optimized,
        "median_ttft_ms",
    )


    eager_tpot = num(
        eager,
        "median_tpot_ms",
    )

    opt_tpot = num(
        optimized,
        "median_tpot_ms",
    )


    eager_out = num(
        eager,
        "output_tokens_per_s",
    )

    opt_out = num(
        optimized,
        "output_tokens_per_s",
    )


    eager_rps = num(
        eager,
        "request_throughput_rps",
    )

    opt_rps = num(
        optimized,
        "request_throughput_rps",
    )


    eager_latency = num(
        eager,
        "median_latency_ms",
    )

    opt_latency = num(
        optimized,
        "median_latency_ms",
    )


    print()
    print("-" * 120)

    print(
        "WORKLOAD =",
        workload,
    )

    print(
        "EAGER_MEDIAN_TTFT_MS =",
        f"{eager_ttft:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_TTFT_MS =",
        f"{opt_ttft:.3f}",
    )

    print(
        "TTFT_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_ttft / opt_ttft:.3f}x",
    )


    print(
        "EAGER_MEDIAN_TPOT_MS =",
        f"{eager_tpot:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_TPOT_MS =",
        f"{opt_tpot:.3f}",
    )

    print(
        "TPOT_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_tpot / opt_tpot:.3f}x",
    )


    print(
        "EAGER_OUTPUT_TOKENS_S =",
        f"{eager_out:.3f}",
    )

    print(
        "OPTIMIZED_OUTPUT_TOKENS_S =",
        f"{opt_out:.3f}",
    )

    print(
        "OUTPUT_THROUGHPUT_SPEEDUP =",
        f"{opt_out / eager_out:.3f}x",
    )


    print(
        "EAGER_REQUESTS_S =",
        f"{eager_rps:.3f}",
    )

    print(
        "OPTIMIZED_REQUESTS_S =",
        f"{opt_rps:.3f}",
    )

    print(
        "REQUEST_THROUGHPUT_SPEEDUP =",
        f"{opt_rps / eager_rps:.3f}x",
    )


    print(
        "EAGER_MEDIAN_LATENCY_MS =",
        f"{eager_latency:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_LATENCY_MS =",
        f"{opt_latency:.3f}",
    )

    print(
        "LATENCY_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_latency / opt_latency:.3f}x",
    )


eager_startup = float(
    data["single_c1"][
        "eager"
    ][
        "startup_s"
    ]
)

opt_startup = float(
    data["single_c1"][
        "optimized"
    ][
        "startup_s"
    ]
)


print()
print("=" * 120)
print("STARTUP")
print("=" * 120)

print(
    "EAGER_STARTUP_S =",
    f"{eager_startup:.3f}",
)

print(
    "OPTIMIZED_STARTUP_S =",
    f"{opt_startup:.3f}",
)

print(
    "OPTIMIZED_STARTUP_OVER_EAGER =",
    f"{opt_startup / eager_startup:.3f}x",
)


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print(
"""
Eager mode disables torch.compile and CUDA Graph execution.

Optimized mode allows vLLM's default optimization path.

The comparison therefore captures a systems trade-off:

    initialization / compilation cost
              versus
    steady-state serving performance

Do not judge serving systems only by startup time.

Long-running production servers may amortize compilation
cost over many requests, so steady-state TTFT, TPOT and
throughput are the more important runtime metrics.

Likewise, an optimization is not automatically beneficial
for every workload shape.

The impact may differ between:

    single-request latency
    moderate concurrency
    high concurrency
""".strip()
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)

print(
    "EAGER_BASELINE = SUCCESS"
)

print(
    "OPTIMIZED_BASELINE = SUCCESS"
)

print(
    "MATCHED_WORKLOAD_COMPARISON = SUCCESS"
)

print(
    "VERDICT = READY_FOR_P1_M9_PROFILING_AND_BOTTLENECK_ANALYSIS"
)
PY


    # =================================================================================================
    # 5. SERVER RUNNER
    # =================================================================================================

    run_mode() {
        MODE="$1"

        if [ "${MODE}" = "eager" ]; then
            LOG="${EAGER_LOG}"
            METRICS="${EAGER_METRICS}"
            EXTRA_ARGS=(--enforce-eager)
        else
            LOG="${OPT_LOG}"
            METRICS="${OPT_METRICS}"
            EXTRA_ARGS=()
        fi

        echo
        echo "${LINE}"
        echo "RUN MODE = ${MODE}"
        echo "${LINE}"

        cleanup_server

        pkill -f "${VLLM_ENV}/bin/vllm serve" \
            2>/dev/null || true

        pkill -f "vllm.*EngineCore" \
            2>/dev/null || true

        sleep 2

        echo "GPU_BEFORE_START:"

        nvidia-smi \
            --query-gpu=name,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader

        rm -f "${LOG}"

        START_NS="$(date +%s%N)"

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
            "${EXTRA_ARGS[@]}" \
            > "${LOG}" 2>&1 &

        SERVER_PID=$!

        echo "SERVER_PID = ${SERVER_PID}"
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
                break
            fi

            if [ $((SECOND % 60)) -eq 0 ]; then

                echo
                echo "WAITING_${MODE}_SECOND = ${SECOND}"

                nvidia-smi \
                    --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
                    --format=csv,noheader \
                    || true

                tail -n 40 "${LOG}" || true
            fi

            sleep 1
        done

        if [ "${READY}" != "TRUE" ]; then
            echo "SERVER_HEALTH = TIMEOUT"
            cat "${LOG}" || true
            return 11
        fi

        END_NS="$(date +%s%N)"

        STARTUP_S="$(
            "${PYTHON}" - \
                "${START_NS}" \
                "${END_NS}" \
            <<'PY'
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

print(
    (end - start) / 1e9
)
PY
        )"

        GPU_MEM="$(
            nvidia-smi \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits \
            | head -n 1 \
            | tr -d ' '
        )"

        echo "SERVER_HEALTH = READY"
        echo "MODE = ${MODE}"
        echo "STARTUP_S = ${STARTUP_S}"
        echo "GPU_MEMORY_READY_MIB = ${GPU_MEM}"

        sleep 2

        P1_M8_HOST="${HOST}" \
        P1_M8_PORT="${PORT}" \
        P1_M8_MODEL="${MODEL_NAME}" \
        P1_M8_MODEL_PATH="${MODEL_PATH}" \
        P1_M8_MODE="${MODE}" \
        P1_M8_CSV="${CSV}" \
        P1_M8_STARTUP_S="${STARTUP_S}" \
        P1_M8_GPU_MEMORY_MIB="${GPU_MEM}" \
        "${PYTHON}" "${CLIENT}"

        curl \
            --noproxy '*' \
            -sS \
            "http://${HOST}:${PORT}/metrics" \
            > "${METRICS}" \
            || true

        echo
        echo "${MODE} PERFORMANCE LOG:"

        grep -Ei \
            'compile|cudagraph|capture|eager|jit|warmup|model loading|init engine' \
            "${LOG}" \
            | tail -n 150 \
            || true

        cleanup_server

        echo "MODE_${MODE}_COMPLETE = TRUE"
    }


    # =================================================================================================
    # 6. EAGER
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "4. EAGER BASELINE"
    echo "${LINE}"

    rm -f "${CSV}"

    run_mode eager


    # =================================================================================================
    # 7. OPTIMIZED
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "5. OPTIMIZED DEFAULT VLLM"
    echo "${LINE}"

    run_mode optimized


    # =================================================================================================
    # 8. CSV
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "6. COMPLETE STRUCTURED RESULTS"
    echo "${LINE}"

    cat "${CSV}"


    # =================================================================================================
    # 9. COMPARE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "7. MATCHED COMPARISON"
    echo "${LINE}"

    "${PYTHON}" \
        "${COMPARE}" \
        "${CSV}"


    # =================================================================================================
    # 10. SERVER LOG EVIDENCE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "8. EAGER SERVER OPTIMIZATION EVIDENCE"
    echo "${LINE}"

    grep -Ei \
        'enforce eager|compile|cudagraph|capture|jit|warmup|init engine' \
        "${EAGER_LOG}" \
        | tail -n 200 \
        || true


    echo
    echo "${LINE}"
    echo "9. OPTIMIZED SERVER OPTIMIZATION EVIDENCE"
    echo "${LINE}"

    grep -Ei \
        'enforce eager|compile|cudagraph|capture|jit|warmup|init engine' \
        "${OPT_LOG}" \
        | tail -n 250 \
        || true


    # =================================================================================================
    # 11. FINAL
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "P1-M8 KNOWLEDGE CHECKPOINT"
    echo "${LINE}"

    echo "
EAGER MODE

    easier startup/debugging
    no torch.compile
    no CUDA Graph execution

OPTIMIZED MODE

    additional initialization / compilation cost
    potential kernel/graph optimization
    designed for steady-state serving

THE SYSTEMS QUESTION IS NOT:

    Which mode starts faster?

The real question is:

    Does the additional setup cost produce
    better steady-state serving performance?

We evaluate that under identical:

    model
    GPU
    prompt size
    output size
    concurrency
    request count

This is a matched systems experiment.
"

    echo
    echo "${LINE}"
    echo "FINAL VERDICT"
    echo "${LINE}"

    echo "EAGER_SERVER = SUCCESS"
    echo "OPTIMIZED_SERVER = SUCCESS"
    echo "MATCHED_SINGLE_REQUEST = SUCCESS"
    echo "MATCHED_CONCURRENCY_8 = SUCCESS"
    echo "MATCHED_CONCURRENCY_16 = SUCCESS"
    echo "STRUCTURED_COMPARISON = SUCCESS"
    echo "VERDICT = READY_FOR_P1_M9_PROFILING_AND_BOTTLENECK_ANALYSIS"

    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "CSV_RESULT = ${CSV}"
    echo "CLIENT = ${CLIENT}"
    echo "COMPARE_SCRIPT = ${COMPARE}"
    echo "EAGER_LOG = ${EAGER_LOG}"
    echo "OPTIMIZED_LOG = ${OPT_LOG}"
    echo "EAGER_METRICS = ${EAGER_METRICS}"
    echo "OPTIMIZED_METRICS = ${OPT_METRICS}"

    echo
    echo "${LINE}"
    echo "END OF P1-M8"
    echo "${LINE}"

    cleanup_server

    trap - EXIT
}


# =================================================================================================
# SYNCHRONOUS TEE
#
# Unlike:
#
#     exec > >(tee ...)
#
# this pipeline waits for tee to finish before printing the
# final path. Therefore the final terminal line is deterministic.
# =================================================================================================

set +e

main 2>&1 | tee "${TXT}"

STATUS=${PIPESTATUS[0]}

set -e

printf '%s\n' "${TXT}"

exit "${STATUS}"
