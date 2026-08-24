#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"
VLLM="${ENV}/bin/vllm"

HF_HOME="${BASE}/caches/huggingface"

SRC_DIR="${PROJECT}/src/p1_m10"
RESULT_DIR="${PROJECT}/results/p1_m10"
ARTIFACT_DIR="${BASE}/artifacts"

HOST="127.0.0.1"
PORT="8000"
MODEL_NAME="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M10_KV_CACHE_SCHEDULING_${TS}.txt"

CSV="${RESULT_DIR}/P1_M10_RESULTS_${TS}.csv"

CLIENT="${SRC_DIR}/p1_m10_client.py"
COMPARE="${SRC_DIR}/p1_m10_compare.py"

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

    sleep 2
}


create_client() {

cat > "${CLIENT}" <<'PY'
import concurrent.futures
import csv
import json
import math
import os
import re
import statistics
import threading
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M10_HOST"]
PORT = int(os.environ["P1_M10_PORT"])

MODEL = os.environ["P1_M10_MODEL"]
MODEL_PATH = os.environ["P1_M10_MODEL_PATH"]

MODE = os.environ["P1_M10_MODE"]
PLAN = os.environ["P1_M10_PLAN"]

MAX_NUM_SEQS = int(
    os.environ["P1_M10_MAX_NUM_SEQS"]
)

PREFIX_CACHING = (
    os.environ["P1_M10_PREFIX_CACHING"]
)

GPU_BLOCKS_OVERRIDE = int(
    os.environ["P1_M10_GPU_BLOCKS_OVERRIDE"]
)

SERVER_KV_TOKENS = int(
    float(
        os.environ["P1_M10_SERVER_KV_TOKENS"]
    )
)

SERVER_MAX_CONCURRENCY = float(
    os.environ[
        "P1_M10_SERVER_MAX_CONCURRENCY"
    ]
)

CSV_PATH = Path(
    os.environ["P1_M10_CSV"]
)

BASE_URL = f"http://{HOST}:{PORT}"

LINE = "=" * 120

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


# ==================================================================================================
# TOKENIZATION
# ==================================================================================================

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


def plain_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_to_target(
    fixed_prefix,
    target_tokens,
    output_target,
):
    filler = (
        " GPU inference scheduler KV cache attention "
        "continuous batching memory latency throughput "
        "kernel prefill decode serving system."
    )

    if output_target >= 128:
        suffix = (
            " Give a detailed technical explanation and "
            "continue until the requested output limit."
        )
    else:
        suffix = (
            " Give a concise technical explanation."
        )

    lo = 0
    hi = 32

    while (
        chat_tokens(
            fixed_prefix
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
            fixed_prefix
            + filler * mid
            + suffix
        )

        if chat_tokens(text) < target_tokens:
            lo = mid + 1
        else:
            hi = mid

    options = []

    for repeats in [
        max(0, lo - 2),
        max(0, lo - 1),
        lo,
        lo + 1,
    ]:
        text = (
            fixed_prefix
            + filler * repeats
            + suffix
        )

        count = chat_tokens(text)

        options.append(
            (
                abs(
                    count
                    - target_tokens
                ),
                count,
                text,
            )
        )

    options.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return options[0][2]


def unique_prompt(
    target_tokens,
    output_target,
):
    return build_to_target(
        (
            f"Unique request {uuid.uuid4().hex}. "
            "Analyze this synthetic AI infrastructure workload."
        ),
        target_tokens,
        output_target,
    )


def build_shared_prefix(
    target_plain_tokens=1500,
):
    base = (
        "This is intentionally shared reusable context "
        "for a prefix cache experiment. "
    )

    filler = (
        "LLM serving scheduler KV cache GPU attention "
        "continuous batching prefill decode memory kernel. "
    )

    lo = 0
    hi = 32

    while (
        plain_tokens(
            base + filler * hi
        )
        < target_plain_tokens
    ):
        hi *= 2

    while lo < hi:
        mid = (
            lo + hi
        ) // 2

        text = (
            base
            + filler * mid
        )

        if (
            plain_tokens(text)
            < target_plain_tokens
        ):
            lo = mid + 1
        else:
            hi = mid

    options = []

    for repeats in [
        max(0, lo - 2),
        max(0, lo - 1),
        lo,
        lo + 1,
    ]:
        text = (
            base
            + filler * repeats
        )

        n = plain_tokens(text)

        options.append(
            (
                abs(
                    n
                    - target_plain_tokens
                ),
                n,
                text,
            )
        )

    options.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return options[0][2]


SHARED_PREFIX = build_shared_prefix()

print(
    "SHARED_PREFIX_PLAIN_TOKENS =",
    plain_tokens(
        SHARED_PREFIX
    ),
)


def shared_prompt(
    target_tokens=2048,
    output_target=32,
):
    # The unique portion appears AFTER the long shared
    # prefix so the preceding token blocks are reusable.
    fixed = (
        SHARED_PREFIX
        + " "
        + f"Unique tail {uuid.uuid4().hex}. "
    )

    return build_to_target(
        fixed,
        target_tokens,
        output_target,
    )


# ==================================================================================================
# PROMETHEUS METRICS
# ==================================================================================================

def get_metrics():
    with httpx.Client(
        trust_env=False,
        timeout=20.0,
    ) as client:

        response = client.get(
            BASE_URL
            + "/metrics"
        )

        response.raise_for_status()

        return response.text


def metric_value(
    text,
    name,
):
    names = [name]

    if not name.endswith(
        (
            "_total",
            "_sum",
            "_count",
        )
    ):
        names.append(
            name
            + "_total"
        )

    for candidate in names:

        pattern = re.compile(
            rf"^{re.escape(candidate)}"
            r"(?:\{[^}]*\})?"
            r"\s+"
            r"([-+0-9.eE]+)"
            r"$",
            re.MULTILINE,
        )

        matches = (
            pattern.findall(text)
        )

        if matches:
            return sum(
                float(x)
                for x in matches
            )

    return 0.0


def metric_delta(
    before,
    after,
    name,
):
    return (
        metric_value(
            after,
            name,
        )
        - metric_value(
            before,
            name,
        )
    )


def histogram_average_delta(
    before,
    after,
    base,
):
    sum_delta = metric_delta(
        before,
        after,
        base + "_sum",
    )

    count_delta = metric_delta(
        before,
        after,
        base + "_count",
    )

    if count_delta <= 0:
        return float("nan")

    return (
        sum_delta
        / count_delta
    )


# ==================================================================================================
# LIVE SCHEDULER TELEMETRY
# ==================================================================================================

class SchedulerTelemetry:

    def __init__(
        self,
        interval=0.10,
    ):
        self.interval = interval
        self.samples = []

        self.stop_event = (
            threading.Event()
        )

        self.thread = None


    def run(self):

        while not self.stop_event.is_set():

            try:
                text = get_metrics()

                self.samples.append(
                    {
                        "running": metric_value(
                            text,
                            "vllm:num_requests_running",
                        ),

                        "waiting": metric_value(
                            text,
                            "vllm:num_requests_waiting",
                        ),

                        "kv": metric_value(
                            text,
                            "vllm:kv_cache_usage_perc",
                        ),
                    }
                )

            except Exception:
                pass

            self.stop_event.wait(
                self.interval
            )


    def start(self):

        self.samples = []

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.run,
            daemon=True,
        )

        self.thread.start()


    def stop(self):

        self.stop_event.set()

        if self.thread:
            self.thread.join(
                timeout=5
            )


    def summary(self):

        if not self.samples:

            return {
                "telemetry_samples": 0,
                "peak_running_requests": float("nan"),
                "peak_waiting_requests": float("nan"),
                "peak_kv_cache_usage_pct": float("nan"),
            }

        return {
            "telemetry_samples": (
                len(self.samples)
            ),

            "peak_running_requests": max(
                x["running"]
                for x in self.samples
            ),

            "peak_waiting_requests": max(
                x["waiting"]
                for x in self.samples
            ),

            "peak_kv_cache_usage_pct": (
                max(
                    x["kv"]
                    for x in self.samples
                )
                * 100.0
            ),
        }


# ==================================================================================================
# REQUEST
# ==================================================================================================

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

    first = None

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

                    if first is None:
                        first = (
                            time.perf_counter()
                            - start
                        )

                    pieces.append(
                        content
                    )

    total = (
        time.perf_counter()
        - start
    )

    if first is None:
        raise RuntimeError(
            "No streamed content."
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
        first
        * 1000.0
    )

    total_ms = (
        total
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
        "latency_ms": total_ms,
        "tpot_ms": tpot_ms,
        "prompt_tokens": (
            chat_tokens(prompt)
        ),
        "output_tokens": (
            n_output
        ),
    }


def execute(
    prompts,
    max_tokens,
    concurrency,
):
    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                p,
                max_tokens,
            )
            for p in prompts
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


# ==================================================================================================
# STATISTICS
# ==================================================================================================

def percentile(
    values,
    q,
):
    xs = sorted(values)

    if not xs:
        return float("nan")

    p = (
        len(xs) - 1
    ) * q

    lo = int(p)

    hi = min(
        lo + 1,
        len(xs) - 1,
    )

    frac = (
        p - lo
    )

    return (
        xs[lo]
        * (1.0 - frac)
        + xs[hi]
        * frac
    )


# ==================================================================================================
# WORKLOAD
# ==================================================================================================

def run_experiment(
    experiment,
    target_prompt_tokens,
    target_output_tokens,
    concurrency,
    request_count,
    shared=False,
):
    print()
    print(LINE)
    print(
        "EXPERIMENT =",
        experiment,
    )
    print(LINE)

    print(
        "MODE =",
        MODE,
    )

    print(
        "TARGET_PROMPT_TOKENS =",
        target_prompt_tokens,
    )

    print(
        "TARGET_OUTPUT_TOKENS =",
        target_output_tokens,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "REQUEST_COUNT =",
        request_count,
    )

    print(
        "SHARED_PREFIX =",
        shared,
    )


    # ----------------------------------------------------------------------------------------------
    # Warm-up.
    #
    # Prefix experiment deliberately warms the actual shared
    # prefix so subsequent requests can hit the cache.
    # Metrics are captured only AFTER this warm-up.
    # ----------------------------------------------------------------------------------------------

    if shared:

        warm = shared_prompt(
            target_prompt_tokens,
            target_output_tokens,
        )

    else:

        warm = unique_prompt(
            target_prompt_tokens,
            target_output_tokens,
        )

    request_once(
        warm,
        min(
            target_output_tokens,
            64,
        ),
    )

    print(
        "WARMUP = SUCCESS"
    )


    if shared:

        prompts = [
            shared_prompt(
                target_prompt_tokens,
                target_output_tokens,
            )
            for _ in range(
                request_count
            )
        ]

    else:

        prompts = [
            unique_prompt(
                target_prompt_tokens,
                target_output_tokens,
            )
            for _ in range(
                request_count
            )
        ]


    before = get_metrics()

    telemetry = SchedulerTelemetry(
        interval=0.10
    )

    telemetry.start()

    wall_start = (
        time.perf_counter()
    )

    results = execute(
        prompts,
        target_output_tokens,
        concurrency,
    )

    wall_s = (
        time.perf_counter()
        - wall_start
    )

    telemetry.stop()

    after = get_metrics()


    ttft = [
        x["ttft_ms"]
        for x in results
    ]

    latency = [
        x["latency_ms"]
        for x in results
    ]

    tpot = [
        x["tpot_ms"]
        for x in results
        if not math.isnan(
            x["tpot_ms"]
        )
    ]

    prompt_total = sum(
        x["prompt_tokens"]
        for x in results
    )

    output_total = sum(
        x["output_tokens"]
        for x in results
    )


    prefix_queries = metric_delta(
        before,
        after,
        "vllm:prefix_cache_queries",
    )

    prefix_hits = metric_delta(
        before,
        after,
        "vllm:prefix_cache_hits",
    )


    if prefix_queries > 0:

        prefix_hit_rate = (
            prefix_hits
            / prefix_queries
        )

    else:

        prefix_hit_rate = 0.0


    preemptions = metric_delta(
        before,
        after,
        "vllm:num_preemptions",
    )


    queue_s = (
        histogram_average_delta(
            before,
            after,
            "vllm:request_queue_time_seconds",
        )
    )


    prefill_s = (
        histogram_average_delta(
            before,
            after,
            "vllm:request_prefill_time_seconds",
        )
    )


    decode_s = (
        histogram_average_delta(
            before,
            after,
            "vllm:request_decode_time_seconds",
        )
    )


    computed_kv_tokens = metric_delta(
        before,
        after,
        "vllm:request_prefill_kv_computed_tokens_sum",
    )


    row = {
        "mode": MODE,

        "experiment": experiment,

        "server_kv_cache_tokens": (
            SERVER_KV_TOKENS
        ),

        "server_max_concurrency_4096": (
            SERVER_MAX_CONCURRENCY
        ),

        "max_num_seqs": (
            MAX_NUM_SEQS
        ),

        "prefix_caching": (
            PREFIX_CACHING
        ),

        "gpu_blocks_override": (
            GPU_BLOCKS_OVERRIDE
        ),

        "target_prompt_tokens": (
            target_prompt_tokens
        ),

        "target_output_tokens": (
            target_output_tokens
        ),

        "concurrency": concurrency,

        "request_count": (
            request_count
        ),

        "wall_s": wall_s,

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
                ttft
            )
        ),

        "p95_ttft_ms": (
            percentile(
                ttft,
                0.95,
            )
        ),

        "median_tpot_ms": (
            statistics.median(
                tpot
            )
            if tpot
            else float("nan")
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
            request_count
            / wall_s
        ),

        "prompt_tokens_per_s": (
            prompt_total
            / wall_s
        ),

        "output_tokens_per_s": (
            output_total
            / wall_s
        ),

        "server_avg_queue_ms": (
            queue_s
            * 1000.0
        ),

        "server_avg_prefill_ms": (
            prefill_s
            * 1000.0
        ),

        "server_avg_decode_ms": (
            decode_s
            * 1000.0
        ),

        "num_preemptions": (
            preemptions
        ),

        "prefix_cache_queries_tokens": (
            prefix_queries
        ),

        "prefix_cache_hits_tokens": (
            prefix_hits
        ),

        "prefix_cache_hit_rate": (
            prefix_hit_rate
        ),

        "prefill_kv_computed_tokens": (
            computed_kv_tokens
        ),
    }


    row.update(
        telemetry.summary()
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


    return row


# ==================================================================================================
# PLAN
# ==================================================================================================

print(LINE)
print(
    "P1-M10 KV CACHE / SCHEDULER CLIENT"
)
print(LINE)

print(
    "MODE =",
    MODE,
)

print(
    "PLAN =",
    PLAN,
)

print(
    "SERVER_KV_CACHE_TOKENS =",
    SERVER_KV_TOKENS,
)

print(
    "SERVER_MAX_CONCURRENCY_4096 =",
    SERVER_MAX_CONCURRENCY,
)

print(
    "MAX_NUM_SEQS =",
    MAX_NUM_SEQS,
)

print(
    "PREFIX_CACHING =",
    PREFIX_CACHING,
)

print(
    "GPU_BLOCKS_OVERRIDE =",
    GPU_BLOCKS_OVERRIDE,
)


rows = []


if PLAN in (
    "baseline",
    "scheduler",
):

    rows.append(
        run_experiment(
            experiment="scheduler_pressure",
            target_prompt_tokens=512,
            target_output_tokens=64,
            concurrency=32,
            request_count=64,
            shared=False,
        )
    )


if PLAN in (
    "baseline",
    "kv_pressure",
):

    rows.append(
        run_experiment(
            experiment="kv_pressure",
            target_prompt_tokens=2048,
            target_output_tokens=128,
            concurrency=8,
            request_count=8,
            shared=False,
        )
    )


if PLAN in (
    "baseline",
    "prefix",
):

    rows.append(
        run_experiment(
            experiment="prefix_cache",
            target_prompt_tokens=2048,
            target_output_tokens=32,
            concurrency=8,
            request_count=16,
            shared=True,
        )
    )


# ==================================================================================================
# APPEND CSV
# ==================================================================================================

fields = [
    "mode",
    "experiment",
    "server_kv_cache_tokens",
    "server_max_concurrency_4096",
    "max_num_seqs",
    "prefix_caching",
    "gpu_blocks_override",
    "target_prompt_tokens",
    "target_output_tokens",
    "concurrency",
    "request_count",
    "wall_s",
    "observed_prompt_tokens_median",
    "observed_output_tokens_median",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_tpot_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "request_throughput_rps",
    "prompt_tokens_per_s",
    "output_tokens_per_s",
    "server_avg_queue_ms",
    "server_avg_prefill_ms",
    "server_avg_decode_ms",
    "num_preemptions",
    "prefix_cache_queries_tokens",
    "prefix_cache_hits_tokens",
    "prefix_cache_hit_rate",
    "prefill_kv_computed_tokens",
    "telemetry_samples",
    "peak_running_requests",
    "peak_waiting_requests",
    "peak_kv_cache_usage_pct",
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

    writer.writerows(
        rows
    )


print()
print(
    "MODE_RESULT = SUCCESS"
)

print(
    "CSV_RESULT =",
    CSV_PATH,
)
PY
}


create_compare() {

cat > "${COMPARE}" <<'PY'
import csv
import sys


path = sys.argv[1]

with open(
    path,
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


def get(
    mode,
    experiment,
):
    for row in rows:
        if (
            row["mode"] == mode
            and
            row["experiment"]
            == experiment
        ):
            return row

    raise RuntimeError(
        f"Missing row "
        f"{mode}/{experiment}"
    )


def f(
    row,
    key,
):
    return float(
        row[key]
    )


baseline_sched = get(
    "baseline",
    "scheduler_pressure",
)

sched4 = get(
    "sched4",
    "scheduler_pressure",
)


baseline_kv = get(
    "baseline",
    "kv_pressure",
)

tinykv = get(
    "tinykv",
    "kv_pressure",
)


cache_on = get(
    "baseline",
    "prefix_cache",
)

cache_off = get(
    "nocache",
    "prefix_cache",
)


print("=" * 120)
print(
    "P1-M10 KV CACHE / SCHEDULING COMPARISON"
)
print("=" * 120)


# ==================================================================================================
# SCHEDULER
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT A — SCHEDULER ADMISSION"
)
print("=" * 120)

print(
    "BASELINE_MAX_NUM_SEQS =",
    baseline_sched[
        "max_num_seqs"
    ],
)

print(
    "CONSTRAINED_MAX_NUM_SEQS =",
    sched4[
        "max_num_seqs"
    ],
)

print(
    "BASELINE_PEAK_RUNNING =",
    f(
        baseline_sched,
        "peak_running_requests",
    ),
)

print(
    "SCHED4_PEAK_RUNNING =",
    f(
        sched4,
        "peak_running_requests",
    ),
)

print(
    "BASELINE_PEAK_WAITING =",
    f(
        baseline_sched,
        "peak_waiting_requests",
    ),
)

print(
    "SCHED4_PEAK_WAITING =",
    f(
        sched4,
        "peak_waiting_requests",
    ),
)

print(
    "BASELINE_TTFT_MS =",
    f(
        baseline_sched,
        "median_ttft_ms",
    ),
)

print(
    "SCHED4_TTFT_MS =",
    f(
        sched4,
        "median_ttft_ms",
    ),
)

print(
    "BASELINE_OUTPUT_TOKENS_S =",
    f(
        baseline_sched,
        "output_tokens_per_s",
    ),
)

print(
    "SCHED4_OUTPUT_TOKENS_S =",
    f(
        sched4,
        "output_tokens_per_s",
    ),
)


scheduler_effect = (
    f(
        sched4,
        "peak_waiting_requests",
    )
    >
    f(
        baseline_sched,
        "peak_waiting_requests",
    )
)


print(
    "SCHEDULER_WAITING_INCREASE_OBSERVED =",
    scheduler_effect,
)


# ==================================================================================================
# KV PRESSURE
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT B — KV CACHE PRESSURE"
)
print("=" * 120)

print(
    "BASELINE_KV_CACHE_TOKENS =",
    baseline_kv[
        "server_kv_cache_tokens"
    ],
)

print(
    "TINYKV_KV_CACHE_TOKENS =",
    tinykv[
        "server_kv_cache_tokens"
    ],
)

print(
    "BASELINE_MAX_CONCURRENCY_4096 =",
    baseline_kv[
        "server_max_concurrency_4096"
    ],
)

print(
    "TINYKV_MAX_CONCURRENCY_4096 =",
    tinykv[
        "server_max_concurrency_4096"
    ],
)

print(
    "BASELINE_PREEMPTIONS =",
    f(
        baseline_kv,
        "num_preemptions",
    ),
)

print(
    "TINYKV_PREEMPTIONS =",
    f(
        tinykv,
        "num_preemptions",
    ),
)

print(
    "BASELINE_PEAK_KV_USAGE_PCT =",
    f(
        baseline_kv,
        "peak_kv_cache_usage_pct",
    ),
)

print(
    "TINYKV_PEAK_KV_USAGE_PCT =",
    f(
        tinykv,
        "peak_kv_cache_usage_pct",
    ),
)

print(
    "BASELINE_TTFT_MS =",
    f(
        baseline_kv,
        "median_ttft_ms",
    ),
)

print(
    "TINYKV_TTFT_MS =",
    f(
        tinykv,
        "median_ttft_ms",
    ),
)

print(
    "BASELINE_OUTPUT_TOKENS_S =",
    f(
        baseline_kv,
        "output_tokens_per_s",
    ),
)

print(
    "TINYKV_OUTPUT_TOKENS_S =",
    f(
        tinykv,
        "output_tokens_per_s",
    ),
)


kv_capacity_reduced = (
    f(
        tinykv,
        "server_kv_cache_tokens",
    )
    <
    f(
        baseline_kv,
        "server_kv_cache_tokens",
    )
)


kv_pressure_observed = (
    f(
        tinykv,
        "num_preemptions",
    )
    > 0
    or
    f(
        tinykv,
        "peak_waiting_requests",
    )
    >
    f(
        baseline_kv,
        "peak_waiting_requests",
    )
)


print(
    "KV_CAPACITY_REDUCTION_OBSERVED =",
    kv_capacity_reduced,
)

print(
    "KV_PRESSURE_OR_PREEMPTION_OBSERVED =",
    kv_pressure_observed,
)


# ==================================================================================================
# PREFIX CACHE
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT C — PREFIX CACHE"
)
print("=" * 120)

print(
    "CACHE_ON_QUERIES =",
    f(
        cache_on,
        "prefix_cache_queries_tokens",
    ),
)

print(
    "CACHE_ON_HITS =",
    f(
        cache_on,
        "prefix_cache_hits_tokens",
    ),
)

print(
    "CACHE_ON_HIT_RATE =",
    f(
        cache_on,
        "prefix_cache_hit_rate",
    ),
)

print(
    "CACHE_OFF_QUERIES =",
    f(
        cache_off,
        "prefix_cache_queries_tokens",
    ),
)

print(
    "CACHE_OFF_HITS =",
    f(
        cache_off,
        "prefix_cache_hits_tokens",
    ),
)

print(
    "CACHE_ON_PREFILL_KV_COMPUTED =",
    f(
        cache_on,
        "prefill_kv_computed_tokens",
    ),
)

print(
    "CACHE_OFF_PREFILL_KV_COMPUTED =",
    f(
        cache_off,
        "prefill_kv_computed_tokens",
    ),
)

print(
    "CACHE_ON_TTFT_MS =",
    f(
        cache_on,
        "median_ttft_ms",
    ),
)

print(
    "CACHE_OFF_TTFT_MS =",
    f(
        cache_off,
        "median_ttft_ms",
    ),
)

print(
    "CACHE_ON_PROMPT_TOKENS_S =",
    f(
        cache_on,
        "prompt_tokens_per_s",
    ),
)

print(
    "CACHE_OFF_PROMPT_TOKENS_S =",
    f(
        cache_off,
        "prompt_tokens_per_s",
    ),
)


prefix_hits_observed = (
    f(
        cache_on,
        "prefix_cache_hits_tokens",
    )
    > 0
)


prefix_compute_reduced = (
    f(
        cache_on,
        "prefill_kv_computed_tokens",
    )
    <
    f(
        cache_off,
        "prefill_kv_computed_tokens",
    )
)


print(
    "PREFIX_CACHE_HITS_OBSERVED =",
    prefix_hits_observed,
)

print(
    "PREFIX_CACHE_REDUCED_COMPUTE =",
    prefix_compute_reduced,
)


# ==================================================================================================
# KNOWLEDGE
# ==================================================================================================

print()
print("=" * 120)
print(
    "P1-M10 KNOWLEDGE CHECKPOINT"
)
print("=" * 120)

print(
"""
KV CACHE CAPACITY

The KV cache stores attention K/V state for active
and reusable token prefixes.

Reducing KV capacity does not change model weights,
but it changes how many sequence tokens can coexist.


SCHEDULER ADMISSION

max_num_seqs limits the number of active sequences
the scheduler may process in one iteration.

If incoming concurrency exceeds this limit:

    running requests are capped
    waiting requests increase

This creates an explicit throughput / queueing trade-off.


KV PRESSURE AND PREEMPTION

When active sequence state cannot fit comfortably in
the KV cache, the serving engine must delay work,
evict/recompute state, or preempt sequences.

Therefore KV capacity is not merely a memory statistic.

It directly participates in scheduling.


PREFIX CACHE

Prefix caching reuses already-computed KV blocks
for matching prompt prefixes.

A correct experiment must:

    create a genuinely shared prefix
    warm that prefix first
    measure cache hits
    measure newly-computed KV tokens

rather than infer caching from latency alone.


SYSTEMS VIEW

Scheduler + KV cache form one coupled resource manager:

      request admission
            ↓
      active sequences
            ↓
      KV block allocation
            ↓
      batching / execution
            ↓
      latency + throughput
""".strip()
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)


print(
    "SCHEDULER_EFFECT =",
    scheduler_effect,
)

print(
    "KV_CAPACITY_REDUCTION =",
    kv_capacity_reduced,
)

print(
    "KV_PRESSURE_EFFECT =",
    kv_pressure_observed,
)

print(
    "PREFIX_HITS =",
    prefix_hits_observed,
)

print(
    "PREFIX_COMPUTE_REDUCTION =",
    prefix_compute_reduced,
)


if (
    scheduler_effect
    and
    kv_capacity_reduced
    and
    kv_pressure_observed
    and
    prefix_hits_observed
    and
    prefix_compute_reduced
):

    print(
        "VERDICT = "
        "READY_FOR_P1_M11_KERNEL_LEVEL_PROFILING"
    )

else:

    print(
        "VERDICT = "
        "P1_M10_BEHAVIOR_REQUIRES_REVIEW"
    )
PY
}


start_server() {
    MODE="$1"
    MAX_SEQS="$2"
    PREFIX="$3"
    BLOCKS_OVERRIDE="$4"
    PLAN="$5"

    cleanup_server

    pkill -f "${ENV}/bin/vllm serve" \
        2>/dev/null || true

    pkill -f "vllm.*EngineCore" \
        2>/dev/null || true

    sleep 2

    LOG="${RESULT_DIR}/P1_M10_${MODE}_SERVER_${TS}.log"

    METRICS="${RESULT_DIR}/P1_M10_${MODE}_METRICS_${TS}.txt"

    ARGS=(
        serve
        "${MODEL_PATH}"

        --host "${HOST}"
        --port "${PORT}"

        --served-model-name "${MODEL_NAME}"

        --dtype float16

        --max-model-len 4096

        --gpu-memory-utilization 0.70

        --generation-config vllm

        --max-num-seqs "${MAX_SEQS}"
    )


    if [ "${PREFIX}" = "on" ]; then
        ARGS+=(
            --enable-prefix-caching
        )
    else
        ARGS+=(
            --no-enable-prefix-caching
        )
    fi


    if [ "${BLOCKS_OVERRIDE}" -gt 0 ]; then
        ARGS+=(
            --num-gpu-blocks-override
            "${BLOCKS_OVERRIDE}"
        )
    fi


    echo
    echo "${LINE}"
    echo "START SERVER MODE = ${MODE}"
    echo "${LINE}"

    echo "MAX_NUM_SEQS = ${MAX_SEQS}"
    echo "PREFIX_CACHING = ${PREFIX}"
    echo "GPU_BLOCKS_OVERRIDE = ${BLOCKS_OVERRIDE}"
    echo "PLAN = ${PLAN}"

    nvidia-smi \
        --query-gpu=name,memory.used,memory.free \
        --format=csv,noheader


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
    "${VLLM}" \
        "${ARGS[@]}" \
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

        cat "${LOG}" || true

        return 11
    fi


    SERVER_INFO="$(
        "${PYTHON}" - "${LOG}" <<'PY'
import re
import sys

text = open(
    sys.argv[1],
    "r",
    encoding="utf-8",
    errors="replace",
).read()

m1 = re.findall(
    r"GPU KV cache size:\s*"
    r"([\d,]+)\s*tokens",
    text,
)

m2 = re.findall(
    r"Maximum concurrency for "
    r"4,096 tokens per request:\s*"
    r"([\d.]+)x",
    text,
)

kv = (
    int(
        m1[-1].replace(",", "")
    )
    if m1
    else 0
)

conc = (
    float(m2[-1])
    if m2
    else 0.0
)

print(
    f"{kv}|{conc}"
)
PY
    )"


    SERVER_KV_TOKENS="$(
        echo "${SERVER_INFO}" \
        | cut -d'|' -f1
    )"


    SERVER_MAX_CONCURRENCY="$(
        echo "${SERVER_INFO}" \
        | cut -d'|' -f2
    )"


    echo "SERVER_KV_CACHE_TOKENS = ${SERVER_KV_TOKENS}"

    echo "SERVER_MAX_CONCURRENCY_4096 = ${SERVER_MAX_CONCURRENCY}"


    P1_M10_HOST="${HOST}" \
    P1_M10_PORT="${PORT}" \
    P1_M10_MODEL="${MODEL_NAME}" \
    P1_M10_MODEL_PATH="${MODEL_PATH}" \
    P1_M10_MODE="${MODE}" \
    P1_M10_PLAN="${PLAN}" \
    P1_M10_MAX_NUM_SEQS="${MAX_SEQS}" \
    P1_M10_PREFIX_CACHING="${PREFIX}" \
    P1_M10_GPU_BLOCKS_OVERRIDE="${BLOCKS_OVERRIDE}" \
    P1_M10_SERVER_KV_TOKENS="${SERVER_KV_TOKENS}" \
    P1_M10_SERVER_MAX_CONCURRENCY="${SERVER_MAX_CONCURRENCY}" \
    P1_M10_CSV="${CSV}" \
    "${PYTHON}" "${CLIENT}"


    curl \
        --noproxy '*' \
        -sS \
        "http://${HOST}:${PORT}/metrics" \
        > "${METRICS}" \
        || true


    echo
    echo "${MODE} SERVER EVIDENCE:"

    grep -Ei \
        'GPU KV cache size|Maximum concurrency|preempt|Prefix cache|Running:|Waiting:|KV cache usage' \
        "${LOG}" \
        | tail -n 150 \
        || true


    cleanup_server

    echo "MODE_${MODE}_COMPLETE = TRUE"
}


main() {
    trap cleanup_server EXIT
    set -e

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M10"
    echo "KV CACHE AND SCHEDULING"
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

    export PATH="${ENV}/bin:/usr/local/cuda-12.8/bin:${PATH}"

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
    "ENVIRONMENT = HEALTHY"
)
PY


    # ==============================================================================================
    # 2. VERIFY FLAGS
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "2. VERIFY REQUIRED VLLM FLAGS"
    echo "${LINE}"

    HELP="$(
        "${VLLM}" serve --help=all
    )"


    for FLAG in \
        "--max-num-seqs" \
        "--enable-prefix-caching" \
        "--no-enable-prefix-caching" \
        "--num-gpu-blocks-override"
    do

        if echo "${HELP}" \
            | grep -q -- "${FLAG}"
        then
            echo "${FLAG} = SUPPORTED"
        else
            echo "${FLAG} = NOT_SUPPORTED"
            return 2
        fi
    done


    # ==============================================================================================
    # 3. MODEL
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "3. LOCAL MODEL"
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


    MODEL_PATH="$(
        cat "${MODEL_PATH_FILE}"
    )"


    if [ -z "${MODEL_PATH}" ] \
        || [ ! -d "${MODEL_PATH}" ]
    then
        echo "ERROR = MODEL_PATH_INVALID"
        return 3
    fi


    echo "MODEL_PATH = ${MODEL_PATH}"


    # ==============================================================================================
    # 4. CLIENTS
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "4. CREATE EXPERIMENT CLIENTS"
    echo "${LINE}"

    create_client
    create_compare

    rm -f "${CSV}"

    echo "CLIENT = ${CLIENT}"
    echo "COMPARE = ${COMPARE}"


    # ==============================================================================================
    # 5. BASELINE
    #
    # Normal KV cache
    # 32 sequences
    # prefix caching ON
    # Runs all three experiments.
    # ==============================================================================================

    start_server \
        "baseline" \
        32 \
        "on" \
        0 \
        "baseline"


    # ==============================================================================================
    # 6. SCHEDULER-CONSTRAINED
    #
    # Only 4 active sequences.
    # Same workload as baseline scheduler experiment.
    # ==============================================================================================

    start_server \
        "sched4" \
        4 \
        "on" \
        0 \
        "scheduler"


    # ==============================================================================================
    # 7. TINY KV CACHE
    #
    # 512 blocks × 16 tokens/block ≈ 8192 token capacity
    # for this model/config.
    #
    # This flag is specifically intended by vLLM for
    # preemption testing.
    # ==============================================================================================

    start_server \
        "tinykv" \
        32 \
        "on" \
        512 \
        "kv_pressure"


    # ==============================================================================================
    # 8. PREFIX CACHE OFF
    # ==============================================================================================

    start_server \
        "nocache" \
        32 \
        "off" \
        0 \
        "prefix"


    # ==============================================================================================
    # 9. COMPLETE CSV
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "9. COMPLETE STRUCTURED RESULTS"
    echo "${LINE}"

    cat "${CSV}"


    # ==============================================================================================
    # 10. COMPARISON
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "10. MATCHED COMPARISON"
    echo "${LINE}"

    "${PYTHON}" \
        "${COMPARE}" \
        "${CSV}"


    # ==============================================================================================
    # 11. FINAL VERDICT
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "11. FINAL VERDICT"
    echo "${LINE}"


    if grep -q \
        'VERDICT = READY_FOR_P1_M11_KERNEL_LEVEL_PROFILING' \
        "${TXT}"
    then

        echo "SCHEDULER_CHARACTERIZATION = SUCCESS"
        echo "KV_CACHE_PRESSURE_CHARACTERIZATION = SUCCESS"
        echo "PREFIX_CACHE_CHARACTERIZATION = SUCCESS"
        echo "PREEMPTION_PRESSURE_TEST = SUCCESS"
        echo "VERDICT = READY_FOR_P1_M11_KERNEL_LEVEL_PROFILING"

    else

        echo "VERDICT = P1_M10_BEHAVIOR_REQUIRES_REVIEW"

        return 20
    fi


    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "CSV_RESULT = ${CSV}"
    echo "CLIENT = ${CLIENT}"
    echo "COMPARE = ${COMPARE}"

    echo
    echo "${LINE}"
    echo "END OF P1-M10"
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
