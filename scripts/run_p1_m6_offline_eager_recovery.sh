#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"
ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"
VLLM="${ENV}/bin/vllm"

HF_HOME="${BASE}/caches/huggingface"

RESULT_DIR="${PROJECT}/results/p1_m6"
ARTIFACT_DIR="${BASE}/artifacts"
SRC_DIR="${PROJECT}/src/p1_m6"

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL="qwen25-05b"

HOST="127.0.0.1"
PORT="8000"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}" \
    "${SRC_DIR}" \
    "${HF_HOME}"

TXT="${ARTIFACT_DIR}/P1_M6_OFFLINE_EAGER_RECOVERY_${TS}.txt"
SERVER_LOG="${RESULT_DIR}/P1_M6_OFFLINE_EAGER_SERVER_${TS}.log"
CSV="${RESULT_DIR}/P1_M6_OFFLINE_EAGER_BENCHMARK_${TS}.csv"
MODEL_PATH_FILE="${RESULT_DIR}/P1_M6_MODEL_PATH_${TS}.txt"
MODELS_JSON="${RESULT_DIR}/P1_M6_MODELS_${TS}.json"
SINGLE_JSON="${RESULT_DIR}/P1_M6_SINGLE_${TS}.json"
BENCH_PY="${SRC_DIR}/p1_m6_offline_eager_benchmark.py"

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

            echo "STOPPING_SERVER = TRUE"

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

    if [ -f "${SERVER_LOG}" ]; then
        echo
        echo "========================================================================================================================"
        echo "COMPLETE SERVER LOG"
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
echo "OFFLINE MODEL + EAGER VLLM RECOVERY"
echo "${LINE}"

echo
echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "MODEL_ID = ${MODEL_ID}"
echo "VLLM_ENV = ${ENV}"


# =================================================================================================
# 1. ENVIRONMENT
# =================================================================================================

echo
echo "${LINE}"
echo "1. VERIFY VLLM ENVIRONMENT"
echo "${LINE}"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR = PYTHON_NOT_FOUND"
    exit 1
fi

if [ ! -x "${VLLM}" ]; then
    echo "ERROR = VLLM_CLI_NOT_FOUND"
    exit 1
fi

"${PYTHON}" - <<'PY'
import importlib.metadata
import torch
import vllm

print("vllm =", importlib.metadata.version("vllm"))
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())

assert importlib.metadata.version("vllm") == "0.27.1+cu129"
assert torch.cuda.is_available()
assert str(torch.version.cuda).startswith("12.9")

print("GPU =", torch.cuda.get_device_name(0))
print("ENVIRONMENT_CHECK = SUCCESS")
PY

"${PYTHON}" -m pip check


# =================================================================================================
# 2. PROXY
# =================================================================================================

echo
echo "${LINE}"
echo "2. PREPARE HTTP-ONLY DOWNLOAD ENVIRONMENT"
echo "${LINE}"

if [ -f /root/autodl-tmp/mihomo/proxy_env.sh ]; then
    source /root/autodl-tmp/mihomo/proxy_env.sh
fi

echo "HTTP_PROXY = ${HTTP_PROXY:-NOT_SET}"
echo "HTTPS_PROXY = ${HTTPS_PROXY:-NOT_SET}"
echo "ALL_PROXY_BEFORE_DOWNLOAD = ${ALL_PROXY:-NOT_SET}"

echo
echo "HUGGINGFACE_HTTP_TEST:"

HF_CODE="$(
    env \
        -u ALL_PROXY \
        -u all_proxy \
        HTTP_PROXY="http://127.0.0.1:7890" \
        HTTPS_PROXY="http://127.0.0.1:7890" \
        curl \
            -L \
            -sS \
            -o /dev/null \
            -w '%{http_code}' \
            --connect-timeout 15 \
            --max-time 30 \
            https://huggingface.co \
    || true
)"

echo "HUGGINGFACE_HTTP = ${HF_CODE}"

if [ "${HF_CODE}" != "200" ]; then
    echo "ERROR = HUGGINGFACE_NOT_REACHABLE"
    exit 2
fi


# =================================================================================================
# 3. INSPECT EXISTING HF CACHE
# =================================================================================================

echo
echo "${LINE}"
echo "3. INSPECT EXISTING MODEL CACHE"
echo "${LINE}"

echo "HF_HOME = ${HF_HOME}"

find "${HF_HOME}" \
    -type f \
    \( \
        -name '*.safetensors' \
        -o -name '*.json' \
        -o -name '*.model' \
        -o -name '*.txt' \
    \) \
    -printf '%s %p\n' \
    2>/dev/null \
    | sort -nr \
    | head -n 40 \
    || true

echo
echo "TOTAL_HF_CACHE:"

du -sh "${HF_HOME}" || true


# =================================================================================================
# 4. MATERIALIZE COMPLETE MODEL SNAPSHOT
# =================================================================================================

echo
echo "${LINE}"
echo "4. MATERIALIZE COMPLETE QWEN SNAPSHOT"
echo "${LINE}"

env \
    -u ALL_PROXY \
    -u all_proxy \
    HTTP_PROXY="http://127.0.0.1:7890" \
    HTTPS_PROXY="http://127.0.0.1:7890" \
    http_proxy="http://127.0.0.1:7890" \
    https_proxy="http://127.0.0.1:7890" \
    HF_HOME="${HF_HOME}" \
    HF_HUB_DISABLE_XET=1 \
    MODEL_ID="${MODEL_ID}" \
    MODEL_PATH_FILE="${MODEL_PATH_FILE}" \
    "${PYTHON}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download


repo = os.environ["MODEL_ID"]
out_file = Path(
    os.environ["MODEL_PATH_FILE"]
)

print("REPO =", repo)


def valid_snapshot(path):
    path = Path(path)

    required_any_weights = (
        list(path.glob("*.safetensors"))
        + list(path.glob("*.bin"))
    )

    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "tokenizer.model",
        path / "vocab.json",
    ]

    has_config = (
        path / "config.json"
    ).exists()

    has_tokenizer = any(
        p.exists()
        for p in tokenizer_candidates
    )

    weight_bytes = sum(
        p.stat().st_size
        for p in required_any_weights
        if p.is_file()
    )

    print(
        "SNAPSHOT_PATH =",
        path,
    )

    print(
        "HAS_CONFIG =",
        has_config,
    )

    print(
        "HAS_TOKENIZER =",
        has_tokenizer,
    )

    print(
        "WEIGHT_FILE_COUNT =",
        len(required_any_weights),
    )

    print(
        "WEIGHT_BYTES =",
        weight_bytes,
    )

    return (
        has_config
        and has_tokenizer
        and len(required_any_weights) > 0
        and weight_bytes > 500_000_000
    )


cached_path = None

try:
    cached_path = snapshot_download(
        repo_id=repo,
        local_files_only=True,
    )

    print(
        "LOCAL_SNAPSHOT_DISCOVERY = SUCCESS"
    )

except Exception as e:
    print(
        "LOCAL_SNAPSHOT_DISCOVERY = INCOMPLETE"
    )
    print(
        "LOCAL_DISCOVERY_EXCEPTION =",
        repr(e),
    )


if (
    cached_path is not None
    and valid_snapshot(cached_path)
):
    final_path = cached_path

    print(
        "MODEL_DOWNLOAD_REQUIRED = FALSE"
    )

else:
    print(
        "MODEL_DOWNLOAD_REQUIRED = TRUE"
    )

    print(
        "DOWNLOADING_COMPLETE_MODEL_SNAPSHOT"
    )

    final_path = snapshot_download(
        repo_id=repo,
    )

    if not valid_snapshot(final_path):
        raise RuntimeError(
            "Downloaded snapshot failed validation."
        )


out_file.write_text(
    str(final_path),
    encoding="utf-8",
)

print(
    "MODEL_SNAPSHOT_READY = SUCCESS"
)

print(
    "FINAL_MODEL_PATH =",
    final_path,
)
PY

MODEL_PATH="$(
    cat "${MODEL_PATH_FILE}"
)"

if [ -z "${MODEL_PATH}" ] \
    || [ ! -d "${MODEL_PATH}" ]
then
    echo "ERROR = MODEL_PATH_INVALID"
    exit 3
fi

echo
echo "MODEL_PATH = ${MODEL_PATH}"

echo
echo "MODEL_SNAPSHOT_CONTENTS:"

find "${MODEL_PATH}" \
    -maxdepth 1 \
    -type f \
    -printf '%12s  %f\n' \
    | sort \
    || true

echo
echo "MODEL_SNAPSHOT_SIZE:"

du -sh "${MODEL_PATH}" || true


# =================================================================================================
# 5. VERIFY MODEL CAN BE OPENED OFFLINE
# =================================================================================================

echo
echo "${LINE}"
echo "5. OFFLINE TRANSFORMERS CONFIG TEST"
echo "${LINE}"

env \
    -u ALL_PROXY \
    -u all_proxy \
    -u HTTP_PROXY \
    -u HTTPS_PROXY \
    -u http_proxy \
    -u https_proxy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LOCAL_MODEL="${MODEL_PATH}" \
    "${PYTHON}" - <<'PY'
import os

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

path = os.environ[
    "LOCAL_MODEL"
]

config = AutoConfig.from_pretrained(
    path,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True,
)

print(
    "MODEL_TYPE =",
    config.model_type,
)

print(
    "VOCAB_SIZE =",
    config.vocab_size,
)

print(
    "TOKENIZER_CLASS =",
    tokenizer.__class__.__name__,
)

print(
    "OFFLINE_MODEL_METADATA = SUCCESS"
)
PY


# =================================================================================================
# 6. CLEAN STALE SERVER
# =================================================================================================

echo
echo "${LINE}"
echo "6. CLEAN STALE VLLM PROCESSES"
echo "${LINE}"

pkill -f \
    "${ENV}/bin/vllm serve" \
    2>/dev/null \
    || true

sleep 2

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader


# =================================================================================================
# 7. START OFFLINE + EAGER SERVER
# =================================================================================================

echo
echo "${LINE}"
echo "7. START OFFLINE EAGER VLLM SERVER"
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

echo
echo "SERVER_PROCESS_ENVIRONMENT_MODE:"
echo "NETWORK = OFFLINE"
echo "ENFORCE_EAGER = TRUE"
echo "TORCH_COMPILE = DISABLED"
echo "CUDA_GRAPHS = DISABLED"


# =================================================================================================
# 8. WAIT FOR HEALTH
# =================================================================================================

echo
echo "${LINE}"
echo "8. WAIT FOR SERVER HEALTH"
echo "${LINE}"

READY="FALSE"

for SECOND in $(seq 1 600); do

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then

        echo
        echo "SERVER_PROCESS_EXITED = TRUE"

        echo
        echo "SERVER_LOG:"
        cat "${SERVER_LOG}" || true

        exit 4
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

    if [ $((SECOND % 20)) -eq 0 ]; then

        echo
        echo "------------------------------------------------------------------------------------------------------------------------"
        echo "WAITING_SECOND = ${SECOND}"
        echo "------------------------------------------------------------------------------------------------------------------------"

        echo
        echo "GPU:"
        nvidia-smi \
            --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
            --format=csv,noheader \
            || true

        echo
        echo "PROCESS:"
        ps \
            -eo pid,ppid,stat,%cpu,%mem,etime,cmd \
            | grep -E \
                'vllm|EngineCore|APIServer' \
            | grep -v grep \
            || true

        echo
        echo "LATEST_SERVER_LOG:"
        tail -n 40 \
            "${SERVER_LOG}" \
            || true

    fi

    sleep 1
done

if [ "${READY}" != "TRUE" ]; then

    echo "SERVER_HEALTH = TIMEOUT"

    echo
    echo "FINAL_PROCESS_STATE:"
    ps \
        -eo pid,ppid,stat,%cpu,%mem,etime,cmd \
        | grep -E \
            'vllm|EngineCore|APIServer' \
        | grep -v grep \
        || true

    echo
    echo "FINAL_GPU_STATE:"
    nvidia-smi || true

    echo
    echo "FINAL_SERVER_LOG:"
    cat "${SERVER_LOG}" || true

    exit 5
fi


# =================================================================================================
# 9. READY STATE
# =================================================================================================

echo
echo "${LINE}"
echo "9. SERVER READY STATE"
echo "${LINE}"

nvidia-smi

echo
echo "GPU_QUERY:"

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
    --format=csv,noheader


# =================================================================================================
# 10. API TEST
# =================================================================================================

echo
echo "${LINE}"
echo "10. OPENAI-COMPATIBLE API TEST"
echo "${LINE}"

curl \
    --noproxy '*' \
    -sS \
    "http://${HOST}:${PORT}/v1/models" \
    > "${MODELS_JSON}"

cat "${MODELS_JSON}"

echo

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
                "content": "Explain KV cache in one concise technical sentence."
            }
        ],
        "temperature": 0,
        "max_tokens": 48
    }' \
    > "${SINGLE_JSON}"

echo
cat "${SINGLE_JSON}"

echo


# =================================================================================================
# 11. BENCHMARK CLIENT
# =================================================================================================

echo
echo "${LINE}"
echo "11. CREATE SERVING BENCHMARK"
echo "${LINE}"

cat > "${BENCH_PY}" <<'PY'
import concurrent.futures
import csv
import json
import os
import statistics
import time

import httpx


HOST = os.environ["HOST"]
PORT = os.environ["PORT"]
MODEL = os.environ["MODEL"]
CSV_PATH = os.environ["CSV_PATH"]

BASE = f"http://{HOST}:{PORT}"

PROMPT = (
    "Explain why continuous batching can improve "
    "LLM inference throughput."
)


def percentile(xs, q):
    ys = sorted(xs)

    if not ys:
        return float("nan")

    p = (len(ys) - 1) * q

    lo = int(p)
    hi = min(
        lo + 1,
        len(ys) - 1,
    )

    f = p - lo

    return (
        ys[lo] * (1 - f)
        + ys[hi] * f
    )


def one_request():
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

    with httpx.Client(
        trust_env=False,
        timeout=120.0,
    ) as client:

        t0 = time.perf_counter()

        r = client.post(
            BASE
            + "/v1/chat/completions",
            json=payload,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        r.raise_for_status()

        data = r.json()

    usage = (
        data.get("usage")
        or {}
    )

    return {
        "latency_s": elapsed,
        "prompt_tokens": int(
            usage.get(
                "prompt_tokens",
                0,
            )
        ),
        "completion_tokens": int(
            usage.get(
                "completion_tokens",
                0,
            )
        ),
    }


def ttft():
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

    first = None
    chunks = 0

    with httpx.Client(
        trust_env=False,
        timeout=120.0,
    ) as client:

        start = time.perf_counter()

        with client.stream(
            "POST",
            BASE
            + "/v1/chat/completions",
            json=payload,
        ) as response:

            response.raise_for_status()

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
                    obj = json.loads(
                        body
                    )
                except Exception:
                    continue

                chunks += 1

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

                if (
                    first is None
                    and content
                ):
                    first = (
                        time.perf_counter()
                        - start
                    )

        total = (
            time.perf_counter()
            - start
        )

    if first is None:
        raise RuntimeError(
            "No streamed content observed"
        )

    return (
        first,
        total,
        chunks,
    )


def run_group(
    concurrency,
    count,
):
    start = time.perf_counter()

    values = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as ex:

        futures = [
            ex.submit(
                one_request
            )
            for _ in range(count)
        ]

        for f in (
            concurrent.futures
            .as_completed(
                futures
            )
        ):
            values.append(
                f.result()
            )

    wall = (
        time.perf_counter()
        - start
    )

    latencies = [
        x["latency_s"]
        for x in values
    ]

    output_tokens = sum(
        x["completion_tokens"]
        for x in values
    )

    return {
        "concurrency": concurrency,
        "requests": count,
        "wall_s": wall,
        "median_latency_s": (
            statistics.median(
                latencies
            )
        ),
        "p95_latency_s": (
            percentile(
                latencies,
                0.95,
            )
        ),
        "requests_per_s": (
            count / wall
        ),
        "output_tokens": (
            output_tokens
        ),
        "output_tokens_per_s": (
            output_tokens / wall
        ),
    }


print(
    "=" * 120
)

print(
    "P1-M6 VLLM OFFLINE EAGER SERVING BENCHMARK"
)

print(
    "=" * 120
)

print()
print("WARMUP")

for i in range(2):
    x = one_request()

    print(
        f"WARMUP_{i+1}_MS = "
        f"{x['latency_s'] * 1000:.3f}"
    )


print()
print("STREAMING")

first, total, chunks = ttft()

print(
    "TTFT_MS =",
    round(
        first * 1000,
        3,
    ),
)

print(
    "STREAM_TOTAL_MS =",
    round(
        total * 1000,
        3,
    ),
)

print(
    "STREAM_CHUNKS =",
    chunks,
)


print()
print("CONCURRENCY SWEEP")

rows = []

for concurrency, count in [
    (1, 4),
    (4, 8),
    (8, 16),
]:

    row = run_group(
        concurrency,
        count,
    )

    rows.append(row)

    print()
    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "REQUESTS =",
        count,
    )

    print(
        "WALL_S =",
        round(
            row["wall_s"],
            6,
        ),
    )

    print(
        "MEDIAN_LATENCY_MS =",
        round(
            row[
                "median_latency_s"
            ]
            * 1000,
            3,
        ),
    )

    print(
        "P95_LATENCY_MS =",
        round(
            row[
                "p95_latency_s"
            ]
            * 1000,
            3,
        ),
    )

    print(
        "REQUESTS_PER_S =",
        round(
            row[
                "requests_per_s"
            ],
            3,
        ),
    )

    print(
        "OUTPUT_TOKENS_PER_S =",
        round(
            row[
                "output_tokens_per_s"
            ],
            3,
        ),
    )


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "concurrency",
            "requests",
            "wall_s",
            "median_latency_s",
            "p95_latency_s",
            "requests_per_s",
            "output_tokens",
            "output_tokens_per_s",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print(
    "CSV_RESULT =",
    CSV_PATH,
)

print()
print(
    "BENCHMARK = SUCCESS"
)
PY


# =================================================================================================
# 12. RUN BENCHMARK
# =================================================================================================

echo
echo "${LINE}"
echo "12. RUN SERVING BENCHMARK"
echo "${LINE}"

HOST="${HOST}" \
PORT="${PORT}" \
MODEL="${SERVED_MODEL}" \
CSV_PATH="${CSV}" \
"${PYTHON}" "${BENCH_PY}"


# =================================================================================================
# 13. RESULTS
# =================================================================================================

echo
echo "${LINE}"
echo "13. STRUCTURED RESULTS"
echo "${LINE}"

cat "${CSV}"

echo
echo "${LINE}"
echo "14. SERVER LOG CHECKPOINTS"
echo "${LINE}"

grep -E \
    'Loading model|Loading weights|Model loading|KV cache|GPU KV|CUDAGraph|torch.compile|eager|Application startup complete|Starting vLLM|Uvicorn|INFO' \
    "${SERVER_LOG}" \
    | tail -n 120 \
    || true


# =================================================================================================
# 15. FINAL VERDICT
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
echo "LOCAL_MODEL_SNAPSHOT = SUCCESS"
echo "OFFLINE_MODEL_LOAD = SUCCESS"
echo "ENFORCE_EAGER = TRUE"
echo "OPENAI_API = SUCCESS"
echo "SERVING_BENCHMARK = SUCCESS"

if [ "${HEALTH}" = "200" ]; then

    echo "VERDICT = READY_FOR_P1_M7_VLLM_SERVING_CHARACTERIZATION"

else

    echo "VERDICT = P1_M6_REQUIRES_FURTHER_DIAGNOSIS"
    exit 6

fi

echo
echo "${LINE}"
echo "ARTIFACTS"
echo "${LINE}"

echo "MODEL_PATH_FILE = ${MODEL_PATH_FILE}"
echo "SERVER_LOG = ${SERVER_LOG}"
echo "CSV_RESULT = ${CSV}"
echo "MODELS_JSON = ${MODELS_JSON}"
echo "SINGLE_JSON = ${SINGLE_JSON}"
echo "BENCHMARK_SOURCE = ${BENCH_PY}"

echo
echo "${LINE}"
echo "END OF P1-M6 RECOVERY"
echo "${LINE}"
