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
