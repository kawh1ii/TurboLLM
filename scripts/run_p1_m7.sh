#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

VLLM_ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${VLLM_ENV}/bin/python"
VLLM="${VLLM_ENV}/bin/vllm"

RESULT_DIR="${PROJECT}/results/p1_m7"
SRC_DIR="${PROJECT}/src/p1_m7"
ARTIFACT_DIR="${BASE}/artifacts"

HF_HOME="${BASE}/caches/huggingface"

HOST="127.0.0.1"
PORT="8000"
SERVED_MODEL="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${RESULT_DIR}" \
    "${SRC_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M7_VLLM_SERVING_CHARACTERIZATION_${TS}.txt"
CSV="${RESULT_DIR}/P1_M7_CHARACTERIZATION_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/P1_M7_VLLM_SERVER_${TS}.log"
METRICS="${RESULT_DIR}/P1_M7_FINAL_METRICS_${TS}.txt"

CLIENT="${SRC_DIR}/p1_m7_characterization.py"

FIFO="/tmp/p1_m7_log_${TS}_$$.fifo"

SERVER_PID=""
TEE_PID=""

# -------------------------------------------------------------------------------------------------
# Reliable logging:
# all experiment output -> terminal + TXT
# final terminal line -> TXT absolute path only
# -------------------------------------------------------------------------------------------------

exec 3>&1
exec 4>&2

mkfifo "${FIFO}"

tee "${TXT}" < "${FIFO}" >&3 &
TEE_PID=$!

exec > "${FIFO}" 2>&1

rm -f "${FIFO}"

finish() {
    STATUS=$?
    trap - EXIT

    echo
    echo "========================================================================================================================"
    echo "FINAL CLEANUP"
    echo "========================================================================================================================"

    if [ -n "${SERVER_PID:-}" ]; then
        if kill -0 "${SERVER_PID}" 2>/dev/null; then

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
    fi

    echo
    echo "GPU_AFTER_CLEANUP:"

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader \
        2>/dev/null || true

    echo
    echo "========================================================================================================================"
    echo "SCRIPT COMPLETION"
    echo "========================================================================================================================"
    echo "EXIT_CODE = ${STATUS}"

    if [ "${STATUS}" -eq 0 ]; then
        echo "SCRIPT_STATUS = SUCCESS"
    else
        echo "SCRIPT_STATUS = FAILED"
    fi

    # Close writer side so tee gets EOF.
    exec 1>&-
    exec 2>&-

    wait "${TEE_PID}" 2>/dev/null || true

    # Restore untouched terminal.
    exec 1>&3
    exec 2>&4

    exec 3>&-
    exec 4>&-

    # STRICT FINAL TERMINAL LINE.
    printf '%s\n' "${TXT}"

    exit "${STATUS}"
}

trap finish EXIT

set -e

LINE="========================================================================================================================"

echo "${LINE}"
echo "AI INFRA PROJECT 1 - P1-M7"
echo "VLLM SERVING CHARACTERIZATION"
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

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR = VLLM_ENV_NOT_FOUND"
    exit 1
fi

if [ ! -x "${VLLM}" ]; then
    echo "ERROR = VLLM_CLI_NOT_FOUND"
    exit 1
fi

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
echo "GPU:"

nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit \
    --format=csv,noheader


# =================================================================================================
# 2. FIND LOCAL MODEL
# =================================================================================================

echo
echo "${LINE}"
echo "2. LOCATE LOCAL MODEL SNAPSHOT"
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
    exit 2
fi

MODEL_PATH="$(
    cat "${MODEL_PATH_FILE}"
)"

if [ -z "${MODEL_PATH}" ] \
    || [ ! -d "${MODEL_PATH}" ]
then
    echo "ERROR = LOCAL_MODEL_PATH_INVALID"
    exit 2
fi

echo "MODEL_PATH_FILE = ${MODEL_PATH_FILE}"
echo "MODEL_PATH = ${MODEL_PATH}"

du -sh "${MODEL_PATH}" || true


# =================================================================================================
# 3. CLEAN STALE SERVER
# =================================================================================================

echo
echo "${LINE}"
echo "3. CLEAN STALE VLLM PROCESSES"
echo "${LINE}"

pkill -f "${VLLM_ENV}/bin/vllm serve" \
    2>/dev/null || true

pkill -f "vllm.*EngineCore" \
    2>/dev/null || true

sleep 2

echo "STALE_PROCESS_CLEANUP = COMPLETE"

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader


# =================================================================================================
# 4. START CONTROLLED SERVER
# =================================================================================================

echo
echo "${LINE}"
echo "4. START VLLM SERVER"
echo "${LINE}"

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
    --served-model-name "${SERVED_MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.70 \
    --enforce-eager \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!

echo "SERVER_PID = ${SERVER_PID}"
echo "SERVER_LOG = ${SERVER_LOG}"
echo "MODE = OFFLINE + EAGER"
echo "PREFIX_CACHING = VLLM_DEFAULT"


# =================================================================================================
# 5. WAIT FOR HEALTH
# =================================================================================================

echo
echo "${LINE}"
echo "5. WAIT FOR SERVER HEALTH"
echo "${LINE}"

READY="FALSE"

for SECOND in $(seq 1 600); do

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then

        echo "SERVER_PROCESS_EXITED = TRUE"
        echo
        cat "${SERVER_LOG}" || true

        exit 3
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

    if [ $((SECOND % 30)) -eq 0 ]; then

        echo
        echo "WAITING_SECOND = ${SECOND}"

        nvidia-smi \
            --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
            --format=csv,noheader \
            || true

        tail -n 30 "${SERVER_LOG}" || true
    fi

    sleep 1

done

if [ "${READY}" != "TRUE" ]; then

    echo "SERVER_HEALTH = TIMEOUT"
    cat "${SERVER_LOG}" || true

    exit 4
fi


# =================================================================================================
# 6. READY STATE
# =================================================================================================

echo
echo "${LINE}"
echo "6. SERVER READY STATE"
echo "${LINE}"

curl \
    --noproxy '*' \
    -sS \
    "http://${HOST}:${PORT}/v1/models"

echo

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader


# =================================================================================================
# 7. CREATE CHARACTERIZATION CLIENT
# =================================================================================================

echo
echo "${LINE}"
echo "7. CREATE CHARACTERIZATION CLIENT"
echo "${LINE}"

cat > "${CLIENT}" <<'PY'
import concurrent.futures
import csv
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M7_HOST"]
PORT = int(os.environ["P1_M7_PORT"])
MODEL = os.environ["P1_M7_MODEL"]
MODEL_PATH = os.environ["P1_M7_MODEL_PATH"]
CSV_PATH = Path(os.environ["P1_M7_CSV"])

BASE_URL = f"http://{HOST}:{PORT}"

LINE = "=" * 120

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

records = []


def percentile(values, q):
    xs = sorted(values)

    if not xs:
        return float("nan")

    p = (len(xs) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(xs) - 1)
    frac = p - lo

    return (
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


def chat_token_count(content):
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    return len(ids)


def construct_prompt(
    target_tokens,
    nonce,
    shared_prefix=None,
):
    if shared_prefix is None:
        prefix = (
            f"Request identifier {nonce}. "
            "Read the following synthetic context and answer briefly. "
        )
    else:
        prefix = (
            f"{shared_prefix} "
            f"Request identifier {nonce}. "
        )

    filler = (
        "system inference memory scheduling batching "
        "attention cache throughput latency gpu "
    )

    lo = 0
    hi = max(32, target_tokens * 4)

    while lo < hi:
        mid = (lo + hi) // 2

        candidate = (
            prefix
            + (filler * mid)
            + "Answer with one concise technical explanation."
        )

        count = chat_token_count(candidate)

        if count < target_tokens:
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
            + (filler * n)
            + "Answer with one concise technical explanation."
        )

        count = chat_token_count(text)

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

    _, actual, text = candidates[0]

    return text, actual


def stream_request(
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
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    first_content_s = None
    start = time.perf_counter()

    usage = {}

    with httpx.Client(
        trust_env=False,
        timeout=180.0,
    ) as client:

        with client.stream(
            "POST",
            BASE_URL
            + "/v1/chat/completions",
            json=payload,
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                body = line[6:].strip()

                if body == "[DONE]":
                    break

                try:
                    obj = json.loads(body)
                except Exception:
                    continue

                if obj.get("usage"):
                    usage = obj["usage"]

                choices = obj.get(
                    "choices",
                    [],
                )

                if not choices:
                    continue

                delta = (
                    choices[0]
                    .get(
                        "delta",
                        {},
                    )
                )

                content = delta.get(
                    "content",
                    "",
                )

                if (
                    first_content_s is None
                    and content
                ):
                    first_content_s = (
                        time.perf_counter()
                        - start
                    )

    total_s = (
        time.perf_counter()
        - start
    )

    if first_content_s is None:
        raise RuntimeError(
            "No streamed output token observed."
        )

    prompt_tokens = int(
        usage.get(
            "prompt_tokens",
            0,
        )
    )

    completion_tokens = int(
        usage.get(
            "completion_tokens",
            0,
        )
    )

    if completion_tokens > 1:
        tpot_ms = (
            (
                total_s
                - first_content_s
            )
            / (
                completion_tokens
                - 1
            )
            * 1000.0
        )
    else:
        tpot_ms = float("nan")

    return {
        "ttft_ms": (
            first_content_s
            * 1000.0
        ),
        "total_latency_ms": (
            total_s
            * 1000.0
        ),
        "tpot_ms": tpot_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_group(
    experiment,
    target_prompt_tokens,
    max_tokens,
    concurrency,
    request_count,
    prefix_mode,
):
    shared_prefix = None

    if prefix_mode == "shared":

        # Large identical prefix deliberately creates
        # a prefix-cache reuse opportunity.
        shared_prefix = (
            "This shared prefix is intentionally identical "
            "across all requests. "
            + (
                "large language model serving systems use "
                "schedulers kv caches attention kernels and batching. "
                * 50
            )
        )

    prompts = []

    expected_prompt_lengths = []

    for i in range(request_count):

        nonce = uuid.uuid4().hex

        text, actual = construct_prompt(
            target_prompt_tokens,
            nonce=nonce,
            shared_prefix=shared_prefix,
        )

        prompts.append(text)
        expected_prompt_lengths.append(actual)

    # Warm up exactly the workload shape,
    # but not the same text.
    warm_prompt, _ = construct_prompt(
        target_prompt_tokens,
        nonce=uuid.uuid4().hex,
        shared_prefix=(
            shared_prefix
            if prefix_mode == "shared"
            else None
        ),
    )

    _ = stream_request(
        warm_prompt,
        max_tokens=max_tokens,
    )

    wall_start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                stream_request,
                prompt,
                max_tokens,
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

    wall_s = (
        time.perf_counter()
        - wall_start
    )

    ttfts = [
        x["ttft_ms"]
        for x in results
    ]

    latencies = [
        x["total_latency_ms"]
        for x in results
    ]

    tpots = [
        x["tpot_ms"]
        for x in results
        if x["tpot_ms"]
        == x["tpot_ms"]
    ]

    total_prompt_tokens = sum(
        x["prompt_tokens"]
        for x in results
    )

    total_output_tokens = sum(
        x["completion_tokens"]
        for x in results
    )

    row = {
        "experiment": experiment,
        "target_prompt_tokens": (
            target_prompt_tokens
        ),
        "observed_prompt_tokens_median": (
            statistics.median(
                [
                    x["prompt_tokens"]
                    for x in results
                ]
            )
        ),
        "max_output_tokens": max_tokens,
        "concurrency": concurrency,
        "request_count": request_count,
        "prefix_mode": prefix_mode,
        "wall_s": wall_s,
        "median_ttft_ms": (
            statistics.median(ttfts)
        ),
        "p95_ttft_ms": (
            percentile(
                ttfts,
                0.95,
            )
        ),
        "median_latency_ms": (
            statistics.median(
                latencies
            )
        ),
        "p95_latency_ms": (
            percentile(
                latencies,
                0.95,
            )
        ),
        "median_tpot_ms": (
            statistics.median(tpots)
            if tpots
            else float("nan")
        ),
        "request_throughput_rps": (
            request_count / wall_s
        ),
        "prompt_tokens_per_s": (
            total_prompt_tokens
            / wall_s
        ),
        "output_tokens_per_s": (
            total_output_tokens
            / wall_s
        ),
        "total_tokens_per_s": (
            (
                total_prompt_tokens
                + total_output_tokens
            )
            / wall_s
        ),
    }

    records.append(row)

    print()
    print("-" * 120)

    for key, value in row.items():
        if isinstance(value, float):
            print(
                f"{key} = {value:.6f}"
            )
        else:
            print(
                f"{key} = {value}"
            )

    return row


print(LINE)
print("P1-M7 VLLM SERVING CHARACTERIZATION")
print(LINE)

print(
    "MODEL =",
    MODEL,
)

print(
    "LOCAL_MODEL =",
    MODEL_PATH,
)


# =================================================================================================
# EXPERIMENT 1 — PROMPT LENGTH
# =================================================================================================

print()
print(LINE)
print("EXPERIMENT 1 — PROMPT LENGTH / PREFILL PRESSURE")
print(LINE)

for prompt_tokens in [
    128,
    512,
    2048,
]:
    run_group(
        experiment="prompt_length",
        target_prompt_tokens=prompt_tokens,
        max_tokens=32,
        concurrency=1,
        request_count=4,
        prefix_mode="unique",
    )


# =================================================================================================
# EXPERIMENT 2 — CONCURRENCY
# =================================================================================================

print()
print(LINE)
print("EXPERIMENT 2 — CONCURRENCY / CONTINUOUS BATCHING")
print(LINE)

for concurrency, requests in [
    (1, 8),
    (4, 12),
    (8, 16),
    (16, 32),
]:
    run_group(
        experiment="concurrency",
        target_prompt_tokens=512,
        max_tokens=64,
        concurrency=concurrency,
        request_count=requests,
        prefix_mode="unique",
    )


# =================================================================================================
# EXPERIMENT 3 — OUTPUT LENGTH
# =================================================================================================

print()
print(LINE)
print("EXPERIMENT 3 — OUTPUT LENGTH / DECODE PRESSURE")
print(LINE)

for output_tokens in [
    16,
    64,
    128,
]:
    run_group(
        experiment="output_length",
        target_prompt_tokens=512,
        max_tokens=output_tokens,
        concurrency=4,
        request_count=8,
        prefix_mode="unique",
    )


# =================================================================================================
# EXPERIMENT 4 — PREFIX CACHE
# =================================================================================================

print()
print(LINE)
print("EXPERIMENT 4 — PREFIX CACHE REUSE")
print(LINE)

run_group(
    experiment="prefix_cache",
    target_prompt_tokens=2048,
    max_tokens=32,
    concurrency=8,
    request_count=16,
    prefix_mode="unique",
)

run_group(
    experiment="prefix_cache",
    target_prompt_tokens=2048,
    max_tokens=32,
    concurrency=8,
    request_count=16,
    prefix_mode="shared",
)


# =================================================================================================
# SAVE
# =================================================================================================

print()
print(LINE)
print("SAVE STRUCTURED RESULTS")
print(LINE)

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
    "experiment",
    "target_prompt_tokens",
    "observed_prompt_tokens_median",
    "max_output_tokens",
    "concurrency",
    "request_count",
    "prefix_mode",
    "wall_s",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "median_tpot_ms",
    "request_throughput_rps",
    "prompt_tokens_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
]

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(records)

print(
    "CSV_RESULT =",
    CSV_PATH,
)


print()
print(LINE)
print("P1-M7 KNOWLEDGE CHECKPOINT")
print(LINE)

print(
r"""
P1-M7 separates four major serving dimensions.

1. PROMPT LENGTH

Longer prompts increase Prefill work.

Watch especially:

    TTFT
    prompt tokens/s

TTFT is usually much more sensitive to prompt length
than steady-state decode TPOT.


2. CONCURRENCY

Concurrency gives vLLM more independent sequences
that can be scheduled together.

Increasing concurrency may improve:

    output token throughput
    GPU utilization
    request throughput

but can worsen:

    TTFT
    per-request latency

This is the core latency/throughput trade-off.


3. OUTPUT LENGTH

Longer generation increases Decode work.

Watch:

    TPOT
    output tokens/s
    total latency

If prompt length remains fixed while output grows,
the workload increasingly emphasizes Decode.


4. PREFIX CACHING

Repeated prefixes can reuse previously computed KV state.

Therefore a benchmark that sends the exact same prompt
many times may substantially overestimate ordinary
continuous-batching gains.

A controlled benchmark should distinguish:

    unique-prefix workload

from:

    shared-prefix workload


TTFT

Time To First Token:

    request arrival
        ↓
    queueing
        ↓
    scheduling
        ↓
    prefill
        ↓
    first generated token


TPOT

Time Per Output Token approximates Decode speed after TTFT.

For N generated tokens:

    TPOT ≈
    (total latency - TTFT)
    ----------------------
           N - 1


IMPORTANT SYSTEMS LESSON

There is no single number called:

    "LLM serving performance"

A serving system must be characterized across:

    workload shape
    prompt length
    output length
    concurrency
    cache reuse
    latency percentiles
    throughput


This is why real AI Infra benchmarking requires
a workload matrix rather than one tokens/s number.
""".strip()
)


print()
print(LINE)
print("FINAL VERDICT")
print(LINE)

print("PROMPT_LENGTH_SWEEP = SUCCESS")
print("CONCURRENCY_SWEEP = SUCCESS")
print("OUTPUT_LENGTH_SWEEP = SUCCESS")
print("PREFIX_CACHE_CHARACTERIZATION = SUCCESS")
print("TTFT_CHARACTERIZATION = SUCCESS")
print("TPOT_CHARACTERIZATION = SUCCESS")
print("STRUCTURED_CSV_RESULT = SUCCESS")
print(
    "VERDICT = "
    "READY_FOR_P1_M8_OPTIMIZED_VS_EAGER_VLLM"
)
PY

echo "CLIENT_SOURCE = ${CLIENT}"


# =================================================================================================
# 8. RUN CHARACTERIZATION
# =================================================================================================

echo
echo "${LINE}"
echo "8. RUN SERVING CHARACTERIZATION"
echo "${LINE}"

P1_M7_HOST="${HOST}" \
P1_M7_PORT="${PORT}" \
P1_M7_MODEL="${SERVED_MODEL}" \
P1_M7_MODEL_PATH="${MODEL_PATH}" \
P1_M7_CSV="${CSV}" \
"${PYTHON}" "${CLIENT}"


# =================================================================================================
# 9. CSV
# =================================================================================================

echo
echo "${LINE}"
echo "9. COMPLETE CSV"
echo "${LINE}"

cat "${CSV}"


# =================================================================================================
# 10. VLLM METRICS
# =================================================================================================

echo
echo "${LINE}"
echo "10. VLLM METRICS SNAPSHOT"
echo "${LINE}"

curl \
    --noproxy '*' \
    -sS \
    "http://${HOST}:${PORT}/metrics" \
    > "${METRICS}" \
    || true

grep -E \
    'vllm:|prefix|cache|request|token|time_to_first|inter_token|generation' \
    "${METRICS}" \
    | tail -n 150 \
    || true


# =================================================================================================
# 11. SERVER LOG CHECKPOINTS
# =================================================================================================

echo
echo "${LINE}"
echo "11. SERVER LOG PERFORMANCE CHECKPOINTS"
echo "${LINE}"

grep -E \
    'Avg prompt throughput|Avg generation throughput|Running:|Waiting:|GPU KV cache usage|Prefix cache hit rate|JIT compilation|KV cache size|Maximum concurrency' \
    "${SERVER_LOG}" \
    | tail -n 200 \
    || true


# =================================================================================================
# 12. FINAL VERDICT
# =================================================================================================

echo
echo "${LINE}"
echo "FINAL VERDICT"
echo "${LINE}"

HEALTH="$(
    curl \
        --noproxy '*' \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        "http://${HOST}:${PORT}/health" \
    || true
)"

echo "SERVER_HEALTH_HTTP = ${HEALTH}"

if \
    [ "${HEALTH}" = "200" ] \
    && grep -q \
        'VERDICT = READY_FOR_P1_M8_OPTIMIZED_VS_EAGER_VLLM' \
        "${TXT}"
then

    echo "VLLM_SERVER = SUCCESS"
    echo "PROMPT_LENGTH_CHARACTERIZATION = SUCCESS"
    echo "CONCURRENCY_CHARACTERIZATION = SUCCESS"
    echo "OUTPUT_LENGTH_CHARACTERIZATION = SUCCESS"
    echo "PREFIX_CACHE_CHARACTERIZATION = SUCCESS"
    echo "VERDICT = READY_FOR_P1_M8_OPTIMIZED_VS_EAGER_VLLM"

else

    echo "VERDICT = P1_M7_REQUIRES_FURTHER_DIAGNOSIS"
    exit 5

fi


echo
echo "${LINE}"
echo "ARTIFACTS"
echo "${LINE}"

echo "CLIENT_SOURCE = ${CLIENT}"
echo "CSV_RESULT = ${CSV}"
echo "SERVER_LOG = ${SERVER_LOG}"
echo "METRICS = ${METRICS}"

echo
echo "${LINE}"
echo "END OF P1-M7 RESULT"
echo "${LINE}"
