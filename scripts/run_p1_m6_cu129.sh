#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

BASE_ENV="${BASE}/envs/project1"
VLLM_ENV="${BASE}/envs/project1-vllm-0.27.1"

SRC_DIR="${PROJECT}/src/p1_m6"
RESULT_DIR="${PROJECT}/results/p1_m6"
ARTIFACT_DIR="${BASE}/artifacts"

HF_HOME="${BASE}/caches/huggingface"
PIP_CACHE_DIR="${BASE}/caches/pip"

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL="qwen25-05b"

HOST="127.0.0.1"
PORT="8000"

VLLM_VERSION="0.27.1+cu129"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}" \
    "${HF_HOME}" \
    "${PIP_CACHE_DIR}"

TXT="${ARTIFACT_DIR}/P1_M6_VLLM_SERVING_BASELINE_${TS}.txt"
CSV="${RESULT_DIR}/P1_M6_SERVING_BENCHMARK_${TS}.csv"

SERVER_LOG="${RESULT_DIR}/P1_M6_VLLM_SERVER_${TS}.log"
MODELS_JSON="${RESULT_DIR}/P1_M6_MODELS_${TS}.json"
SINGLE_JSON="${RESULT_DIR}/P1_M6_SINGLE_RESPONSE_${TS}.json"

BENCHMARK_PY="${SRC_DIR}/vllm_serving_baseline.py"

SERVER_PID=""

exec 3>&1
exec > >(tee "${TXT}") 2>&1


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
    echo "GPU_AFTER_SERVER_STOP:"
    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader \
        2>/dev/null || true

    if [ -f "${SERVER_LOG}" ]; then
        echo
        echo "========================================================================================================================"
        echo "COMPLETE VLLM SERVER LOG"
        echo "========================================================================================================================"

        cat "${SERVER_LOG}" || true
    fi

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

    exec 1>&3 2>&3

    printf '%s\n' "${TXT}"

    exec 3>&-

    exit "${STATUS}"
}

trap finish EXIT

set -e

LINE="========================================================================================================================"

echo "${LINE}"
echo "AI INFRA PROJECT 1 - P1-M6"
echo "VLLM SERVING BASELINE"
echo "${LINE}"

echo
echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "MODEL_ID = ${MODEL_ID}"
echo "SERVED_MODEL = ${SERVED_MODEL}"
echo "VLLM_VERSION = ${VLLM_VERSION}"
echo "HOST = ${HOST}"
echo "PORT = ${PORT}"


# =================================================================================================
# 1. Proxy
# =================================================================================================

echo
echo "${LINE}"
echo "1. NETWORK / PROXY"
echo "${LINE}"

if [ -f /root/autodl-tmp/mihomo/proxy_env.sh ]; then
    source /root/autodl-tmp/mihomo/proxy_env.sh
fi

export HF_HOME
export PIP_CACHE_DIR

export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu129"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "HTTPS_PROXY = ${HTTPS_PROXY:-NOT_SET}"
echo "HF_HOME = ${HF_HOME}"
echo "PIP_CACHE_DIR = ${PIP_CACHE_DIR}"
echo "PIP_INDEX_URL = ${PIP_INDEX_URL}"
echo "PIP_EXTRA_INDEX_URL = ${PIP_EXTRA_INDEX_URL}"

GH="$(
    curl \
        -L \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        --connect-timeout 10 \
        --max-time 30 \
        https://github.com \
    || true
)"

HF="$(
    curl \
        -L \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        --connect-timeout 10 \
        --max-time 30 \
        https://huggingface.co \
    || true
)"

PYPI="$(
    curl \
        -L \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        --connect-timeout 10 \
        --max-time 30 \
        https://pypi.org \
    || true
)"

echo "GITHUB_HTTP = ${GH}"
echo "HUGGINGFACE_HTTP = ${HF}"
echo "PYPI_HTTP = ${PYPI}"


# =================================================================================================
# 2. Host environment
# =================================================================================================

echo
echo "${LINE}"
echo "2. HOST GPU / DRIVER"
echo "${LINE}"

nvidia-smi

echo
echo "GPU_QUERY:"

nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit,compute_cap \
    --format=csv,noheader


# =================================================================================================
# 3. Dedicated vLLM environment
# =================================================================================================

echo
echo "${LINE}"
echo "3. CREATE / REUSE DEDICATED VLLM ENVIRONMENT"
echo "${LINE}"

if [ ! -x "${BASE_ENV}/bin/python" ]; then
    echo "ERROR = BASE_PROJECT1_ENV_NOT_FOUND"
    exit 1
fi

if [ ! -x "${VLLM_ENV}/bin/python" ]; then

    echo "VLLM_ENV = CREATE_NEW"
    echo "SOURCE_PYTHON = ${BASE_ENV}/bin/python"

    "${BASE_ENV}/bin/python" \
        -m venv \
        "${VLLM_ENV}"

else

    echo "VLLM_ENV = EXISTING"

fi

PYTHON="${VLLM_ENV}/bin/python"
PIP="${PYTHON} -m pip"
VLLM="${VLLM_ENV}/bin/vllm"

echo "VLLM_ENV_PATH = ${VLLM_ENV}"

"${PYTHON}" --version

echo
"${PYTHON}" -m pip --version


# =================================================================================================
# 4. Install vLLM only if needed
# =================================================================================================

echo
echo "${LINE}"
echo "4. VLLM INSTALLATION"
echo "${LINE}"

INSTALLED_VERSION="$(
"${PYTHON}" - <<'PY' 2>/dev/null || true
try:
    import importlib.metadata
    print(importlib.metadata.version("vllm"))
except Exception:
    pass
PY
)"

echo "CURRENT_VLLM_VERSION = ${INSTALLED_VERSION:-NOT_INSTALLED}"

if [ "${INSTALLED_VERSION:-}" != "${VLLM_VERSION}" ]; then

    echo "ACTION = INSTALL_VLLM_${VLLM_VERSION}"

    "${PYTHON}" -m pip install \
        --index-url https://pypi.org/simple \
        --upgrade \
        pip \
        setuptools \
        wheel

    "${PYTHON}" -m pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        "vllm==${VLLM_VERSION}"

else

    echo "ACTION = REUSE_EXISTING_VLLM"

fi


# =================================================================================================
# 5. Verify installation
# =================================================================================================

echo
echo "${LINE}"
echo "5. VLLM ENVIRONMENT VERIFICATION"
echo "${LINE}"

"${PYTHON}" - <<'PY'
import importlib.metadata
import sys

import torch
import vllm

print("PYTHON =", sys.version.replace("\n", " "))
print("VLLM =", importlib.metadata.version("vllm"))
print("TORCH =", torch.__version__)
print("TORCH_CUDA =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in vLLM environment")

print("GPU =", torch.cuda.get_device_name(0))
print("GPU_COUNT =", torch.cuda.device_count())
print(
    "VRAM_GIB =",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        3,
    ),
)
PY

echo
echo "PIP_CHECK:"

"${PYTHON}" -m pip check

echo
echo "VLLM_CLI_VERSION:"

"${VLLM}" --version

echo
echo "VLLM_SERVE_HELP_CHECK:"

"${VLLM}" serve --help 2>&1 | head -n 50


# =================================================================================================
# 6. Freeze environment
# =================================================================================================

echo
echo "${LINE}"
echo "6. FREEZE VLLM RUNTIME ENVIRONMENT"
echo "${LINE}"

FREEZE_FILE="${PROJECT}/docs/project1_vllm_0.27.1_environment_pip_freeze.txt"

"${PYTHON}" -m pip freeze \
    > "${FREEZE_FILE}"

echo "ENVIRONMENT_FREEZE = ${FREEZE_FILE}"


# =================================================================================================
# 7. Start vLLM OpenAI-compatible server
# =================================================================================================

echo
echo "${LINE}"
echo "7. START VLLM OPENAI-COMPATIBLE SERVER"
echo "${LINE}"

echo "GPU_BEFORE_SERVER:"

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader

rm -f "${SERVER_LOG}"

setsid \
env \
    HF_HOME="${HF_HOME}" \
    HTTP_PROXY="${HTTP_PROXY:-}" \
    HTTPS_PROXY="${HTTPS_PROXY:-}" \
    http_proxy="${http_proxy:-}" \
    https_proxy="${https_proxy:-}" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
"${VLLM}" serve \
    "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL}" \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.70 \
    > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!

echo "VLLM_SERVER_PID = ${SERVER_PID}"
echo "VLLM_SERVER_LOG = ${SERVER_LOG}"


# =================================================================================================
# 8. Wait for health
# =================================================================================================

echo
echo "${LINE}"
echo "8. WAIT FOR SERVER HEALTH"
echo "${LINE}"

READY="FALSE"

for SECOND in $(seq 1 300); do

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then

        echo "VLLM_PROCESS_EXITED_EARLY = TRUE"

        echo
        echo "SERVER_LOG_SO_FAR:"
        cat "${SERVER_LOG}" || true

        exit 2
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
        || true
    )"

    if [ "${CODE}" = "200" ]; then
        READY="TRUE"
        echo "SERVER_HEALTH = READY"
        echo "SECONDS_WAITED_APPROX = ${SECOND}"
        break
    fi

    if [ $((SECOND % 10)) -eq 0 ]; then
        echo "WAITING_SECOND = ${SECOND}"
        tail -n 20 "${SERVER_LOG}" || true
    fi

    sleep 1

done

if [ "${READY}" != "TRUE" ]; then

    echo "SERVER_HEALTH = TIMEOUT"

    echo
    cat "${SERVER_LOG}" || true

    exit 3
fi


# =================================================================================================
# 9. Server GPU state
# =================================================================================================

echo
echo "${LINE}"
echo "9. GPU AFTER VLLM SERVER LOAD"
echo "${LINE}"

nvidia-smi

echo
echo "GPU_QUERY:"

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader


# =================================================================================================
# 10. OpenAI API model listing
# =================================================================================================

echo
echo "${LINE}"
echo "10. OPENAI API — MODEL LIST"
echo "${LINE}"

curl \
    --noproxy '*' \
    -sS \
    "http://${HOST}:${PORT}/v1/models" \
    > "${MODELS_JSON}"

cat "${MODELS_JSON}"

echo


# =================================================================================================
# 11. Single request
# =================================================================================================

echo
echo "${LINE}"
echo "11. OPENAI API — SINGLE CHAT COMPLETION"
echo "${LINE}"

curl \
    --noproxy '*' \
    -sS \
    -X POST \
    "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen25-05b",
        "messages": [
            {
                "role": "user",
                "content": "In one short sentence, explain what KV cache is."
            }
        ],
        "temperature": 0,
        "max_tokens": 48
    }' \
    > "${SINGLE_JSON}"

cat "${SINGLE_JSON}"

echo


# =================================================================================================
# 12. Write benchmark client
# =================================================================================================

echo
echo "${LINE}"
echo "12. CREATE SERVING BENCHMARK CLIENT"
echo "${LINE}"

cat > "${BENCHMARK_PY}" <<'PY'
import concurrent.futures
import csv
import json
import os
import statistics
import time
from pathlib import Path

import requests


HOST = os.environ["P1_M6_HOST"]
PORT = int(os.environ["P1_M6_PORT"])
MODEL = os.environ["P1_M6_MODEL"]
CSV_PATH = Path(os.environ["P1_M6_CSV"])

BASE_URL = f"http://{HOST}:{PORT}"

SESSION = requests.Session()
SESSION.trust_env = False

records = []

PROMPT = (
    "Explain in concise technical terms why batching can "
    "increase GPU throughput during LLM inference."
)


def percentile(values, q):
    if not values:
        return float("nan")

    xs = sorted(values)

    position = (len(xs) - 1) * q

    low = int(position)
    high = min(low + 1, len(xs) - 1)

    fraction = position - low

    return (
        xs[low] * (1 - fraction)
        + xs[high] * fraction
    )


def request_once():
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    start = time.perf_counter()

    response = SESSION.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()

    data = response.json()

    usage = data.get("usage") or {}

    return {
        "latency_s": elapsed,
        "prompt_tokens": int(
            usage.get("prompt_tokens", 0)
        ),
        "completion_tokens": int(
            usage.get("completion_tokens", 0)
        ),
        "total_tokens": int(
            usage.get("total_tokens", 0)
        ),
    }


def streaming_ttft():
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
    }

    start = time.perf_counter()

    first_content_time = None
    chunk_count = 0

    response = SESSION.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        decoded = line.decode(
            "utf-8",
            errors="replace",
        )

        if not decoded.startswith("data: "):
            continue

        payload_text = decoded[6:].strip()

        if payload_text == "[DONE]":
            break

        try:
            item = json.loads(
                payload_text
            )
        except Exception:
            continue

        chunk_count += 1

        choices = item.get(
            "choices",
            [],
        )

        if not choices:
            continue

        delta = choices[0].get(
            "delta",
            {},
        )

        content = delta.get(
            "content",
            "",
        )

        if (
            first_content_time is None
            and content
        ):
            first_content_time = (
                time.perf_counter()
                - start
            )

    total = (
        time.perf_counter()
        - start
    )

    if first_content_time is None:
        raise RuntimeError(
            "No content token observed in stream."
        )

    return {
        "ttft_s": first_content_time,
        "total_s": total,
        "chunks": chunk_count,
    }


def concurrency_benchmark(
    concurrency,
    request_count,
):
    wall_start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once
            )
            for _ in range(request_count)
        ]

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            results.append(
                future.result()
            )

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    latencies = [
        r["latency_s"]
        for r in results
    ]

    completion_tokens = sum(
        r["completion_tokens"]
        for r in results
    )

    total_tokens = sum(
        r["total_tokens"]
        for r in results
    )

    request_throughput = (
        request_count / wall_time
    )

    output_token_throughput = (
        completion_tokens / wall_time
        if wall_time > 0
        else 0
    )

    total_token_throughput = (
        total_tokens / wall_time
        if wall_time > 0
        else 0
    )

    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "wall_time_s": wall_time,
        "median_latency_s": statistics.median(
            latencies
        ),
        "p95_latency_s": percentile(
            latencies,
            0.95,
        ),
        "request_throughput_rps": (
            request_throughput
        ),
        "completion_tokens": (
            completion_tokens
        ),
        "output_token_throughput": (
            output_token_throughput
        ),
        "total_token_throughput": (
            total_token_throughput
        ),
    }


LINE = "=" * 120

print(LINE)
print("P1-M6 VLLM SERVING CLIENT BENCHMARK")
print(LINE)

print("BASE_URL =", BASE_URL)
print("MODEL =", MODEL)


print()
print(LINE)
print("1. WARMUP")
print(LINE)

for i in range(2):
    result = request_once()

    print(
        f"WARMUP_{i+1}_LATENCY_S = "
        f"{result['latency_s']:.6f}"
    )

print("WARMUP = SUCCESS")


print()
print(LINE)
print("2. STREAMING / TTFT")
print(LINE)

ttft = streaming_ttft()

print(
    "STREAM_TTFT_MS =",
    round(
        ttft["ttft_s"] * 1000,
        3,
    ),
)

print(
    "STREAM_TOTAL_MS =",
    round(
        ttft["total_s"] * 1000,
        3,
    ),
)

print(
    "STREAM_CHUNKS =",
    ttft["chunks"],
)

records.append(
    {
        "experiment": "streaming",
        "concurrency": 1,
        "request_count": 1,
        "wall_time_s": ttft["total_s"],
        "median_latency_s": ttft["total_s"],
        "p95_latency_s": ttft["total_s"],
        "request_throughput_rps": (
            1.0 / ttft["total_s"]
        ),
        "completion_tokens": 0,
        "output_token_throughput": 0,
        "total_token_throughput": 0,
        "ttft_ms": (
            ttft["ttft_s"] * 1000
        ),
    }
)


print()
print(LINE)
print("3. CONCURRENCY SWEEP")
print(LINE)

print(
    "concurrency"
    "\t"
    "requests"
    "\t"
    "wall_s"
    "\t"
    "median_latency_s"
    "\t"
    "p95_latency_s"
    "\t"
    "req/s"
    "\t"
    "output_tok/s"
)


for concurrency, count in [
    (1, 8),
    (4, 8),
    (8, 16),
]:

    r = concurrency_benchmark(
        concurrency,
        count,
    )

    print(
        f"{r['concurrency']}"
        f"\t"
        f"{r['request_count']}"
        f"\t"
        f"{r['wall_time_s']:.6f}"
        f"\t"
        f"{r['median_latency_s']:.6f}"
        f"\t"
        f"{r['p95_latency_s']:.6f}"
        f"\t"
        f"{r['request_throughput_rps']:.3f}"
        f"\t"
        f"{r['output_token_throughput']:.3f}"
    )

    r["experiment"] = (
        "concurrency_sweep"
    )

    r["ttft_ms"] = ""

    records.append(r)


print()
print(LINE)
print("4. SAVE CSV")
print(LINE)

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fieldnames = [
    "experiment",
    "concurrency",
    "request_count",
    "wall_time_s",
    "median_latency_s",
    "p95_latency_s",
    "request_throughput_rps",
    "completion_tokens",
    "output_token_throughput",
    "total_token_throughput",
    "ttft_ms",
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

print("CSV_RESULT =", CSV_PATH)


print()
print(LINE)
print("P1-M6 KNOWLEDGE CHECKPOINT")
print(LINE)

print(
r"""
FROM MODEL.GENERATE() TO SERVING

P1-M5 used a model directly inside one Python process.

P1-M6 introduces a real serving system:

Client
  ↓
HTTP / OpenAI-compatible API
  ↓
vLLM Server
  ↓
Request Scheduler
  ↓
Model Executor
  ↓
KV Cache
  ↓
CUDA Kernels
  ↓
GPU


WHY SERVING IS DIFFERENT

A single model.generate() call only answers:

    How fast can this one workload run?

A serving system must additionally handle:

    many independent requests
    arrival timing
    batching
    queueing
    scheduling
    KV Cache allocation
    streaming output
    latency / throughput trade-offs


TTFT

TTFT = Time To First Token

It includes work before the first generated token becomes
visible to the client.

It is especially important for interactive applications.


TOTAL LATENCY

Total latency measures the complete request.

For streaming workloads:

    TTFT
        !=
    total latency


CONCURRENCY

Concurrency = number of requests active at the same time.

Increasing concurrency can give the scheduler more
independent work to batch together.

This often improves:

    GPU utilization
    total token throughput

But it may also increase:

    queueing delay
    per-request latency
    TTFT


THIS IS THE CENTRAL SERVING TRADE-OFF:

    latency
        vs
    throughput


WHY vLLM EXISTS

A production LLM system does not want the GPU to process
requests independently one by one.

It wants to efficiently organize many sequences through:

    continuous batching
    scheduling
    KV Cache management
    PagedAttention
    prefix reuse
    chunked prefill

P1-M6 establishes the serving baseline.

The next milestone will systematically vary workload shape
and concurrency to characterize the system.
""".strip()
)


print()
print(LINE)
print("FINAL VERDICT")
print(LINE)

print("OPENAI_API_SERVER = SUCCESS")
print("SINGLE_REQUEST = SUCCESS")
print("STREAMING_TTFT = SUCCESS")
print("CONCURRENCY_SWEEP = SUCCESS")
print("STRUCTURED_CSV_RESULT = SUCCESS")
print(
    "VERDICT = "
    "READY_FOR_P1_M7_VLLM_SERVING_CHARACTERIZATION"
)
PY

echo "BENCHMARK_CLIENT = ${BENCHMARK_PY}"


# =================================================================================================
# 13. Run benchmark
# =================================================================================================

echo
echo "${LINE}"
echo "13. RUN SERVING BENCHMARK"
echo "${LINE}"

P1_M6_HOST="${HOST}" \
P1_M6_PORT="${PORT}" \
P1_M6_MODEL="${SERVED_MODEL}" \
P1_M6_CSV="${CSV}" \
"${PYTHON}" "${BENCHMARK_PY}"


# =================================================================================================
# 14. Include CSV in TXT
# =================================================================================================

echo
echo "${LINE}"
echo "14. CSV CONTENT"
echo "${LINE}"

cat "${CSV}"


# =================================================================================================
# 15. Server metrics / final state
# =================================================================================================

echo
echo "${LINE}"
echo "15. FINAL SERVER / GPU STATE"
echo "${LINE}"

echo "HEALTH_HTTP = $(
    curl \
        --noproxy '*' \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        "http://${HOST}:${PORT}/health"
)"

echo
echo "GPU_QUERY:"

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader


# =================================================================================================
# 16. Artifacts
# =================================================================================================

echo
echo "${LINE}"
echo "16. ARTIFACTS"
echo "${LINE}"

echo "VLLM_ENV = ${VLLM_ENV}"
echo "ENVIRONMENT_FREEZE = ${FREEZE_FILE}"
echo "BENCHMARK_CLIENT = ${BENCHMARK_PY}"
echo "CSV_RESULT = ${CSV}"
echo "MODELS_JSON = ${MODELS_JSON}"
echo "SINGLE_RESPONSE_JSON = ${SINGLE_JSON}"
echo "SERVER_LOG = ${SERVER_LOG}"
echo "TXT_RESULT = ${TXT}"


# =================================================================================================
# 17. Final verdict
# =================================================================================================

echo
echo "${LINE}"
echo "FINAL VERDICT"
echo "${LINE}"

if grep -q \
    "VERDICT = READY_FOR_P1_M7_VLLM_SERVING_CHARACTERIZATION" \
    "${TXT}"
then
    echo "VLLM_INSTALL = SUCCESS"
    echo "VLLM_SERVER = SUCCESS"
    echo "OPENAI_COMPATIBLE_API = SUCCESS"
    echo "STREAMING = SUCCESS"
    echo "CONCURRENCY_BENCHMARK = SUCCESS"
    echo "VERDICT = READY_FOR_P1_M7_VLLM_SERVING_CHARACTERIZATION"
else
    echo "VERDICT = P1_M6_REQUIRES_FURTHER_DIAGNOSIS"
    exit 4
fi

echo
echo "${LINE}"
echo "END OF P1-M6 RESULT"
echo "${LINE}"
