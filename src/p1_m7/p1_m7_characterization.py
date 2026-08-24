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
