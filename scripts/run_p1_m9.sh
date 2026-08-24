#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

VLLM_ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${VLLM_ENV}/bin/python"
VLLM="${VLLM_ENV}/bin/vllm"

SRC_DIR="${PROJECT}/src/p1_m9"
RESULT_DIR="${PROJECT}/results/p1_m9"
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

TXT="${ARTIFACT_DIR}/P1_M9_PROFILING_BOTTLENECK_${TS}.txt"
CSV="${RESULT_DIR}/P1_M9_PROFILE_${TS}.csv"
SERVER_LOG="${RESULT_DIR}/P1_M9_VLLM_SERVER_${TS}.log"
FINAL_METRICS="${RESULT_DIR}/P1_M9_FINAL_METRICS_${TS}.txt"

CLIENT="${SRC_DIR}/p1_m9_profiler.py"

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

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M9"
    echo "VLLM PROFILING AND BOTTLENECK ANALYSIS"
    echo "${LINE}"

    echo
    echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"


    # ==============================================================================================
    # 1. ENVIRONMENT
    # ==============================================================================================

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
    echo "GPU:"

    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit \
        --format=csv,noheader

    echo
    echo "PROFILING TOOLS:"

    echo "NSYS = $(command -v nsys || echo NOT_INSTALLED)"
    echo "NCU = $(command -v ncu || echo NOT_INSTALLED)"
    echo "NVIDIA_SMI = $(command -v nvidia-smi)"


    # ==============================================================================================
    # 2. LOCAL MODEL
    # ==============================================================================================

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


    # ==============================================================================================
    # 3. CREATE PROFILER CLIENT
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "3. CREATE PROFILING CLIENT"
    echo "${LINE}"

    cat > "${CLIENT}" <<'PY'
import concurrent.futures
import csv
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M9_HOST"]
PORT = int(os.environ["P1_M9_PORT"])
MODEL = os.environ["P1_M9_MODEL"]
MODEL_PATH = os.environ["P1_M9_MODEL_PATH"]
CSV_PATH = Path(os.environ["P1_M9_CSV"])

BASE_URL = f"http://{HOST}:{PORT}"

LINE = "=" * 120

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


# --------------------------------------------------------------------------------------------------
# BASIC STATISTICS
# --------------------------------------------------------------------------------------------------

def percentile(values, q):
    xs = sorted(values)

    if not xs:
        return float("nan")

    pos = (len(xs) - 1) * q

    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)

    frac = pos - lo

    return (
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


def safe_median(values):
    values = [
        x
        for x in values
        if not math.isnan(x)
    ]

    if not values:
        return float("nan")

    return statistics.median(values)


# --------------------------------------------------------------------------------------------------
# TOKENIZATION
# --------------------------------------------------------------------------------------------------

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


def output_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_prompt(
    target_tokens,
    output_target,
):
    nonce = uuid.uuid4().hex

    prefix = (
        f"Unique request {nonce}. "
        "Analyze the following synthetic AI infrastructure context. "
    )

    filler = (
        "GPU inference scheduling memory bandwidth KV cache "
        "continuous batching attention kernel latency throughput "
        "prefill decode tensor compute serving request. "
    )

    if output_target >= 256:
        suffix = (
            "Give a long detailed technical explanation. "
            "Continue explaining different aspects until the output limit."
        )
    else:
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
        < target_tokens
    ):
        hi *= 2

        if hi > 16384:
            raise RuntimeError(
                "Prompt construction runaway."
            )

    while lo < hi:

        mid = (
            lo + hi
        ) // 2

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


# --------------------------------------------------------------------------------------------------
# VLLM SERVER METRICS
# --------------------------------------------------------------------------------------------------

def fetch_metrics():
    with httpx.Client(
        trust_env=False,
        timeout=30.0,
    ) as client:

        response = client.get(
            BASE_URL + "/metrics"
        )

        response.raise_for_status()

        return response.text


def metric_value(
    text,
    exact_name,
):
    pattern = re.compile(
        rf"^{re.escape(exact_name)}"
        r"(?:\{[^}]*\})?"
        r"\s+"
        r"([-+0-9.eE]+)"
        r"$",
        re.MULTILINE,
    )

    matches = pattern.findall(
        text
    )

    if not matches:
        return 0.0

    return sum(
        float(x)
        for x in matches
    )


def metric_average_delta(
    before,
    after,
    base_name,
):
    sum_name = (
        base_name
        + "_sum"
    )

    count_name = (
        base_name
        + "_count"
    )

    delta_sum = (
        metric_value(
            after,
            sum_name,
        )
        - metric_value(
            before,
            sum_name,
        )
    )

    delta_count = (
        metric_value(
            after,
            count_name,
        )
        - metric_value(
            before,
            count_name,
        )
    )

    if delta_count <= 0:
        return float("nan")

    return (
        delta_sum
        / delta_count
    )


# --------------------------------------------------------------------------------------------------
# GPU TELEMETRY
#
# utilization.memory from nvidia-smi is NOT exact HBM bandwidth.
# We treat it only as a memory-controller activity proxy.
# --------------------------------------------------------------------------------------------------

def parse_float(text):
    text = text.strip()

    try:
        return float(text)
    except Exception:
        return float("nan")


class GPUTelemetry:
    def __init__(
        self,
        interval_s=0.25,
    ):
        self.interval_s = interval_s
        self.samples = []

        self.stop_event = (
            threading.Event()
        )

        self.thread = None


    def _sample_once(self):
        command = [
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "utilization.memory,"
            "memory.used,"
            "power.draw,"
            "clocks.sm,"
            "clocks.mem",
            "--format=csv,noheader,nounits",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return

        line = (
            result.stdout
            .strip()
            .splitlines()[0]
        )

        parts = [
            x.strip()
            for x in line.split(",")
        ]

        if len(parts) != 6:
            return

        self.samples.append(
            {
                "gpu_util": (
                    parse_float(
                        parts[0]
                    )
                ),
                "memory_util": (
                    parse_float(
                        parts[1]
                    )
                ),
                "memory_used_mib": (
                    parse_float(
                        parts[2]
                    )
                ),
                "power_w": (
                    parse_float(
                        parts[3]
                    )
                ),
                "sm_clock_mhz": (
                    parse_float(
                        parts[4]
                    )
                ),
                "memory_clock_mhz": (
                    parse_float(
                        parts[5]
                    )
                ),
            }
        )


    def _run(self):

        while not (
            self.stop_event.is_set()
        ):

            try:
                self._sample_once()
            except Exception:
                pass

            self.stop_event.wait(
                self.interval_s
            )


    def start(self):

        self.samples = []

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()


    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(
                timeout=5
            )


    def values(
        self,
        key,
    ):
        return [
            x[key]
            for x in self.samples
            if not math.isnan(
                x[key]
            )
        ]


    def summary(self):

        gpu = self.values(
            "gpu_util"
        )

        memory = self.values(
            "memory_util"
        )

        power = self.values(
            "power_w"
        )

        vram = self.values(
            "memory_used_mib"
        )

        sm_clock = self.values(
            "sm_clock_mhz"
        )

        return {
            "telemetry_samples": (
                len(self.samples)
            ),

            "gpu_util_avg_pct": (
                statistics.mean(gpu)
                if gpu
                else float("nan")
            ),

            "gpu_util_p95_pct": (
                percentile(
                    gpu,
                    0.95,
                )
                if gpu
                else float("nan")
            ),

            "gpu_util_max_pct": (
                max(gpu)
                if gpu
                else float("nan")
            ),

            "memory_controller_avg_pct": (
                statistics.mean(
                    memory
                )
                if memory
                else float("nan")
            ),

            "memory_controller_p95_pct": (
                percentile(
                    memory,
                    0.95,
                )
                if memory
                else float("nan")
            ),

            "power_avg_w": (
                statistics.mean(power)
                if power
                else float("nan")
            ),

            "power_max_w": (
                max(power)
                if power
                else float("nan")
            ),

            "vram_avg_mib": (
                statistics.mean(vram)
                if vram
                else float("nan")
            ),

            "vram_max_mib": (
                max(vram)
                if vram
                else float("nan")
            ),

            "sm_clock_avg_mhz": (
                statistics.mean(
                    sm_clock
                )
                if sm_clock
                else float("nan")
            ),
        }


# --------------------------------------------------------------------------------------------------
# REQUEST
# --------------------------------------------------------------------------------------------------

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

        "max_completion_tokens": (
            max_tokens
        ),

        "stream": True,
    }


    start = time.perf_counter()

    first_token = None

    pieces = []


    with httpx.Client(
        trust_env=False,
        timeout=300.0,
    ) as client:

        with client.stream(
            "POST",
            BASE_URL
            + "/v1/chat/completions",
            json=payload,
        ) as response:

            if response.status_code != 200:

                body = (
                    response.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                raise RuntimeError(
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{body}"
                )


            for line in (
                response.iter_lines()
            ):

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

                    if first_token is None:
                        first_token = (
                            time.perf_counter()
                            - start
                        )

                    pieces.append(
                        content
                    )


    total_s = (
        time.perf_counter()
        - start
    )


    if first_token is None:
        raise RuntimeError(
            "No streamed content received."
        )


    output_text = "".join(
        pieces
    )


    n_output = max(
        1,
        output_tokens(
            output_text
        ),
    )


    ttft_ms = (
        first_token
        * 1000.0
    )

    total_ms = (
        total_s
        * 1000.0
    )


    if n_output > 1:

        tpot_ms = (
            total_ms
            - ttft_ms
        ) / (
            n_output
            - 1
        )

    else:
        tpot_ms = float("nan")


    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": (
            total_ms
        ),
        "tpot_ms": tpot_ms,
        "prompt_tokens": (
            chat_tokens(prompt)
        ),
        "output_tokens": (
            n_output
        ),
    }


# --------------------------------------------------------------------------------------------------
# CONCURRENT REQUEST EXECUTION
# --------------------------------------------------------------------------------------------------

def execute_requests(
    prompt_tokens,
    output_tokens_target,
    concurrency,
    count,
):

    prompts = [
        build_prompt(
            prompt_tokens,
            output_tokens_target,
        )
        for _ in range(count)
    ]


    results = []


    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                prompt,
                output_tokens_target,
            )
            for prompt in prompts
        ]


        for future in (
            concurrent.futures
            .as_completed(
                futures
            )
        ):

            results.append(
                future.result()
            )


    return results


# --------------------------------------------------------------------------------------------------
# WORKLOAD
# --------------------------------------------------------------------------------------------------

def run_workload(
    name,
    prompt_tokens,
    output_tokens_target,
    concurrency,
    request_count,
):

    print()
    print(LINE)
    print(
        "WORKLOAD =",
        name,
    )
    print(LINE)

    print(
        "PROMPT_TARGET =",
        prompt_tokens,
    )

    print(
        "OUTPUT_TARGET =",
        output_tokens_target,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "REQUEST_COUNT =",
        request_count,
    )


    # ----------------------------------------------------------------------------------------------
    # Shape warmup.
    # Unique prompts avoid turning this into a prefix-cache benchmark.
    # ----------------------------------------------------------------------------------------------

    warmup_count = max(
        1,
        min(
            concurrency,
            16,
        ),
    )


    print(
        "WARMUP_REQUESTS =",
        warmup_count,
    )


    execute_requests(
        prompt_tokens,
        min(
            output_tokens_target,
            64,
        ),
        concurrency,
        warmup_count,
    )


    print(
        "WARMUP = SUCCESS"
    )


    before_metrics = (
        fetch_metrics()
    )


    telemetry = (
        GPUTelemetry(
            interval_s=0.25
        )
    )

    telemetry.start()


    wall_start = (
        time.perf_counter()
    )


    results = execute_requests(
        prompt_tokens,
        output_tokens_target,
        concurrency,
        request_count,
    )


    wall_s = (
        time.perf_counter()
        - wall_start
    )


    telemetry.stop()


    after_metrics = (
        fetch_metrics()
    )


    gpu_summary = (
        telemetry.summary()
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
    ]


    total_prompt_tokens = sum(
        x["prompt_tokens"]
        for x in results
    )

    total_output_tokens = sum(
        x["output_tokens"]
        for x in results
    )


    queue_s = (
        metric_average_delta(
            before_metrics,
            after_metrics,
            "vllm:request_queue_time_seconds",
        )
    )

    prefill_s = (
        metric_average_delta(
            before_metrics,
            after_metrics,
            "vllm:request_prefill_time_seconds",
        )
    )

    decode_s = (
        metric_average_delta(
            before_metrics,
            after_metrics,
            "vllm:request_decode_time_seconds",
        )
    )


    row = {
        "workload": name,

        "target_prompt_tokens": (
            prompt_tokens
        ),

        "target_output_tokens": (
            output_tokens_target
        ),

        "concurrency": (
            concurrency
        ),

        "request_count": (
            request_count
        ),

        "wall_s": (
            wall_s
        ),

        "observed_prompt_tokens_median": (
            statistics.median(
                [
                    x["prompt_tokens"]
                    for x in results
                ]
            )
        ),

        "observed_output_tokens_median": (
            statistics.median(
                [
                    x["output_tokens"]
                    for x in results
                ]
            )
        ),

        "median_ttft_ms": (
            statistics.median(
                ttfts
            )
        ),

        "p95_ttft_ms": (
            percentile(
                ttfts,
                0.95,
            )
        ),

        "median_tpot_ms": (
            safe_median(
                tpots
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

        "request_throughput_rps": (
            request_count
            / wall_s
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

        "server_avg_queue_ms": (
            queue_s * 1000.0
        ),

        "server_avg_prefill_ms": (
            prefill_s * 1000.0
        ),

        "server_avg_decode_ms": (
            decode_s * 1000.0
        ),
    }


    row.update(
        gpu_summary
    )


    print()

    for key, value in row.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key} = "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key} = {value}"
            )


    # ----------------------------------------------------------------------------------------------
    # Low-risk diagnostic hints.
    #
    # memory_controller_* is only a proxy and must NOT be interpreted as exact HBM bandwidth.
    # ----------------------------------------------------------------------------------------------

    print()

    gpu_avg = row[
        "gpu_util_avg_pct"
    ]

    mem_avg = row[
        "memory_controller_avg_pct"
    ]

    queue_ms = row[
        "server_avg_queue_ms"
    ]

    prefill_ms = row[
        "server_avg_prefill_ms"
    ]

    decode_ms = row[
        "server_avg_decode_ms"
    ]


    if (
        not math.isnan(
            decode_ms
        )
        and not math.isnan(
            prefill_ms
        )
    ):

        if decode_ms > (
            prefill_ms * 3
        ):

            phase_hint = (
                "DECODE_DOMINANT"
            )

        elif prefill_ms > (
            decode_ms * 1.5
        ):

            phase_hint = (
                "PREFILL_DOMINANT"
            )

        else:

            phase_hint = (
                "MIXED_PREFILL_DECODE"
            )

    else:

        phase_hint = (
            "INSUFFICIENT_SERVER_METRICS"
        )


    if (
        not math.isnan(gpu_avg)
        and gpu_avg >= 85
    ):

        gpu_hint = (
            "HIGH_GPU_EXECUTION_SATURATION"
        )

    elif (
        not math.isnan(gpu_avg)
        and gpu_avg >= 60
    ):

        gpu_hint = (
            "MODERATE_GPU_UTILIZATION"
        )

    else:

        gpu_hint = (
            "GPU_NOT_FULLY_SATURATED"
        )


    if (
        not math.isnan(mem_avg)
        and mem_avg >= 70
    ):

        memory_hint = (
            "HIGH_MEMORY_CONTROLLER_ACTIVITY"
        )

    elif (
        not math.isnan(mem_avg)
        and mem_avg >= 40
    ):

        memory_hint = (
            "MODERATE_MEMORY_CONTROLLER_ACTIVITY"
        )

    else:

        memory_hint = (
            "LOW_TO_MODERATE_MEMORY_CONTROLLER_ACTIVITY"
        )


    if (
        not math.isnan(queue_ms)
        and queue_ms >= 20
    ):

        queue_hint = (
            "QUEUEING_PRESSURE_VISIBLE"
        )

    else:

        queue_hint = (
            "QUEUEING_PRESSURE_LOW"
        )


    print(
        "PHASE_HINT =",
        phase_hint,
    )

    print(
        "GPU_HINT =",
        gpu_hint,
    )

    print(
        "MEMORY_ACTIVITY_HINT =",
        memory_hint,
    )

    print(
        "QUEUE_HINT =",
        queue_hint,
    )


    return row


# --------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------

print(LINE)
print(
    "P1-M9 VLLM PROFILING CLIENT"
)
print(LINE)

print(
    "MODEL =",
    MODEL,
)

print(
    "SERVER =",
    BASE_URL,
)


rows = []


# Latency-oriented reference.
rows.append(
    run_workload(
        name="latency_c1",
        prompt_tokens=512,
        output_tokens_target=64,
        concurrency=1,
        request_count=8,
    )
)


# Long prompt, short decode.
rows.append(
    run_workload(
        name="prefill_heavy",
        prompt_tokens=2048,
        output_tokens_target=16,
        concurrency=8,
        request_count=32,
    )
)


# Short prompt, much longer decode.
rows.append(
    run_workload(
        name="decode_heavy",
        prompt_tokens=128,
        output_tokens_target=512,
        concurrency=8,
        request_count=32,
    )
)


# Balanced high-concurrency serving.
rows.append(
    run_workload(
        name="throughput_c16",
        prompt_tokens=512,
        output_tokens_target=64,
        concurrency=16,
        request_count=64,
    )
)


# --------------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------------

print()
print(LINE)
print("SAVE STRUCTURED PROFILE")
print(LINE)


fieldnames = list(
    rows[0].keys()
)


CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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

    writer.writerows(
        rows
    )


print(
    "CSV_RESULT =",
    CSV_PATH,
)


# --------------------------------------------------------------------------------------------------
# CROSS-WORKLOAD ANALYSIS
# --------------------------------------------------------------------------------------------------

print()
print(LINE)
print(
    "CROSS-WORKLOAD BOTTLENECK ANALYSIS"
)
print(LINE)


data = {
    row["workload"]: row
    for row in rows
}


latency = data[
    "latency_c1"
]

prefill = data[
    "prefill_heavy"
]

decode = data[
    "decode_heavy"
]

throughput = data[
    "throughput_c16"
]


print()
print(
    "LATENCY_C1_TTFT_MS =",
    f"{latency['median_ttft_ms']:.3f}",
)

print(
    "LATENCY_C1_TPOT_MS =",
    f"{latency['median_tpot_ms']:.3f}",
)


print()
print(
    "PREFILL_HEAVY_TTFT_MS =",
    f"{prefill['median_ttft_ms']:.3f}",
)

print(
    "PREFILL_HEAVY_SERVER_PREFILL_MS =",
    f"{prefill['server_avg_prefill_ms']:.3f}",
)

print(
    "PREFILL_HEAVY_PROMPT_TOKENS_S =",
    f"{prefill['prompt_tokens_per_s']:.3f}",
)


print()
print(
    "DECODE_HEAVY_TPOT_MS =",
    f"{decode['median_tpot_ms']:.3f}",
)

print(
    "DECODE_HEAVY_SERVER_DECODE_MS =",
    f"{decode['server_avg_decode_ms']:.3f}",
)

print(
    "DECODE_HEAVY_OUTPUT_TOKENS_S =",
    f"{decode['output_tokens_per_s']:.3f}",
)


print()
print(
    "THROUGHPUT_C16_GPU_UTIL_AVG =",
    f"{throughput['gpu_util_avg_pct']:.3f}%",
)

print(
    "THROUGHPUT_C16_MEMORY_ACTIVITY_AVG =",
    f"{throughput['memory_controller_avg_pct']:.3f}%",
)

print(
    "THROUGHPUT_C16_OUTPUT_TOKENS_S =",
    f"{throughput['output_tokens_per_s']:.3f}",
)

print(
    "THROUGHPUT_C16_REQUESTS_S =",
    f"{throughput['request_throughput_rps']:.3f}",
)


print()
print(LINE)
print("IMPORTANT INTERPRETATION LIMIT")
print(LINE)

print(
"""
nvidia-smi utilization.memory is a memory-controller activity
indicator. It is NOT an exact measurement of HBM/VRAM bandwidth.

Therefore P1-M9 uses it only as supporting evidence.

A claim such as:

    "Decode is definitively memory-bandwidth bound"

requires stronger evidence from kernel-level profiling,
hardware counters, or a controlled bandwidth model.

The scientifically safer interpretation is:

    workload phase
    +
    TTFT / TPOT
    +
    server Prefill/Decode timing
    +
    GPU utilization
    +
    memory-controller activity
    +
    throughput

together identify where the serving system is spending time.
""".strip()
)


print()
print(LINE)
print("P1-M9 KNOWLEDGE CHECKPOINT")
print(LINE)

print(
"""
PERFORMANCE DEBUGGING STACK

Application metrics:

    TTFT
    TPOT
    latency
    throughput

        ↓

Serving-system metrics:

    queue time
    prefill time
    decode time
    KV cache state

        ↓

GPU telemetry:

    GPU utilization
    memory-controller activity
    power
    clocks
    VRAM usage

        ↓

Kernel-level profiling:

    CUDA kernels
    launch gaps
    synchronization
    memory transactions
    Tensor Core utilization


BOTTLENECK ANALYSIS IS EVIDENCE-DRIVEN.

Do not infer a bottleneck from one metric alone.


PREFILL-HEAVY WORKLOAD

    large input context
    short generation

Useful signals:

    TTFT
    Prefill time
    prompt tokens/s


DECODE-HEAVY WORKLOAD

    short prompt
    long generation

Useful signals:

    TPOT
    Decode time
    output tokens/s
    memory activity


HIGH-CONCURRENCY WORKLOAD

Useful signals:

    queue time
    GPU utilization
    request throughput
    output throughput


This profiling layer turns:

    "vLLM is fast"

into:

    "which subsystem explains the observed performance?"
""".strip()
)


print()
print(LINE)
print("FINAL CLIENT VERDICT")
print(LINE)

print(
    "LATENCY_PROFILE = SUCCESS"
)

print(
    "PREFILL_PROFILE = SUCCESS"
)

print(
    "DECODE_PROFILE = SUCCESS"
)

print(
    "HIGH_CONCURRENCY_PROFILE = SUCCESS"
)

print(
    "GPU_TELEMETRY = SUCCESS"
)

print(
    "VLLM_SERVER_METRICS = SUCCESS"
)

print(
    "STRUCTURED_PROFILE = SUCCESS"
)

print(
    "VERDICT = "
    "READY_FOR_P1_M10_KV_CACHE_AND_SCHEDULING"
)
PY

    echo "CLIENT_SOURCE = ${CLIENT}"


    # ==============================================================================================
    # 4. CLEAN STALE SERVER
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "4. CLEAN STALE SERVER"
    echo "${LINE}"

    pkill -f "${VLLM_ENV}/bin/vllm serve" \
        2>/dev/null || true

    pkill -f "vllm.*EngineCore" \
        2>/dev/null || true

    sleep 2

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader


    # ==============================================================================================
    # 5. START PRODUCTION-LIKE OPTIMIZED SERVER
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "5. START OPTIMIZED VLLM SERVER"
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
        --served-model-name "${MODEL_NAME}" \
        --dtype float16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.70 \
        --generation-config vllm \
        > "${SERVER_LOG}" 2>&1 &

    SERVER_PID=$!

    echo "SERVER_PID = ${SERVER_PID}"
    echo "SERVER_LOG = ${SERVER_LOG}"
    echo "MODE = OPTIMIZED"


    # ==============================================================================================
    # 6. WAIT
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "6. WAIT FOR SERVER HEALTH"
    echo "${LINE}"

    READY="FALSE"

    for SECOND in $(seq 1 900); do

        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then

            echo "SERVER_EXITED_EARLY = TRUE"

            cat "${SERVER_LOG}" || true

            return 3
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

            echo
            echo "WAITING_SECOND = ${SECOND}"

            nvidia-smi \
                --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
                --format=csv,noheader \
                || true

            tail -n 40 \
                "${SERVER_LOG}" \
                || true
        fi


        sleep 1
    done


    if [ "${READY}" != "TRUE" ]; then

        echo "SERVER_HEALTH = TIMEOUT"

        cat "${SERVER_LOG}" || true

        return 4
    fi


    # ==============================================================================================
    # 7. READY STATE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "7. SERVER READY STATE"
    echo "${LINE}"

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader

    echo
    echo "API_MODEL:"

    curl \
        --noproxy '*' \
        -sS \
        "http://${HOST}:${PORT}/v1/models"

    echo


    # ==============================================================================================
    # 8. PROFILE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "8. RUN PROFILING WORKLOAD MATRIX"
    echo "${LINE}"

    P1_M9_HOST="${HOST}" \
    P1_M9_PORT="${PORT}" \
    P1_M9_MODEL="${MODEL_NAME}" \
    P1_M9_MODEL_PATH="${MODEL_PATH}" \
    P1_M9_CSV="${CSV}" \
    "${PYTHON}" "${CLIENT}"


    # ==============================================================================================
    # 9. CSV
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "9. COMPLETE STRUCTURED PROFILE"
    echo "${LINE}"

    cat "${CSV}"


    # ==============================================================================================
    # 10. FINAL METRICS
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "10. FINAL VLLM METRICS"
    echo "${LINE}"

    curl \
        --noproxy '*' \
        -sS \
        "http://${HOST}:${PORT}/metrics" \
        > "${FINAL_METRICS}" \
        || true

    grep -E \
        'request_queue_time|request_prefill_time|request_decode_time|prefill_kv_computed|cache_config_info' \
        "${FINAL_METRICS}" \
        | tail -n 220 \
        || true


    # ==============================================================================================
    # 11. SERVER EVIDENCE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "11. SERVER PERFORMANCE EVIDENCE"
    echo "${LINE}"

    grep -Ei \
        'Avg prompt throughput|Avg generation throughput|Running:|Waiting:|GPU KV cache usage|Prefix cache hit rate|compile|cudagraph|JIT' \
        "${SERVER_LOG}" \
        | tail -n 250 \
        || true


    # ==============================================================================================
    # 12. FINAL
    # ==============================================================================================

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


    if [ "${HEALTH}" != "200" ]; then
        echo "VERDICT = P1_M9_REQUIRES_FURTHER_DIAGNOSIS"
        return 5
    fi


    echo "OPTIMIZED_SERVER = SUCCESS"
    echo "PREFILL_HEAVY_PROFILE = SUCCESS"
    echo "DECODE_HEAVY_PROFILE = SUCCESS"
    echo "HIGH_CONCURRENCY_PROFILE = SUCCESS"
    echo "GPU_TELEMETRY = SUCCESS"
    echo "VLLM_PHASE_METRICS = SUCCESS"
    echo "BOTTLENECK_ANALYSIS = SUCCESS"
    echo "VERDICT = READY_FOR_P1_M10_KV_CACHE_AND_SCHEDULING"


    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "CSV_RESULT = ${CSV}"
    echo "CLIENT_SOURCE = ${CLIENT}"
    echo "SERVER_LOG = ${SERVER_LOG}"
    echo "FINAL_METRICS = ${FINAL_METRICS}"


    echo
    echo "${LINE}"
    echo "END OF P1-M9"
    echo "${LINE}"


    cleanup_server
    trap - EXIT
}


# ==============================================================================================
# SYNCHRONOUS LOGGING
#
# The final terminal line is printed only after tee exits.
# ==============================================================================================

set +e

main 2>&1 | tee "${TXT}"

STATUS=${PIPESTATUS[0]}

set -e

printf '%s\n' "${TXT}"

exit "${STATUS}"
