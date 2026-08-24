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

    f = p - lo

    return (
        xs[lo] * (1.0 - f)
        + xs[hi] * f
    )


def chat_tokens(text):
    """
    Robust chat token counting.

    Do not depend on the return shape of
    apply_chat_template(tokenize=True), because that differs
    across Transformers/tokenizer versions.

    Instead:

        messages
          -> rendered chat-template string
          -> ordinary tokenizer.encode()
          -> exact 1-D token-id count
    """

    messages = [
        {
            "role": "user",
            "content": text,
        }
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    ids = tokenizer.encode(
        rendered,
        add_special_tokens=False,
    )

    return len(ids)


def output_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def make_prompt(
    target_tokens,
    nonce,
    shared_prefix="",
):
    intro = (
        shared_prefix
        + f" Request identifier {nonce}. "
        + "Analyze the synthetic context below. "
    )

    filler = (
        "GPU inference scheduling KV cache batching "
        "memory latency throughput attention. "
    )

    suffix = (
        "Give one concise technical explanation."
    )

    lo = 0
    hi = max(
        32,
        target_tokens,
    )

    while chat_tokens(
        intro + filler * hi + suffix
    ) < target_tokens:
        hi *= 2

        if hi > 10000:
            raise RuntimeError(
                "Prompt construction runaway."
            )

    while lo < hi:

        mid = (
            lo + hi
        ) // 2

        text = (
            intro
            + filler * mid
            + suffix
        )

        n = chat_tokens(text)

        if n < target_tokens:
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
            intro
            + filler * reps
            + suffix
        )

        n = chat_tokens(text)

        candidates.append(
            (
                abs(n - target_tokens),
                n,
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
    max_completion_tokens,
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
            max_completion_tokens
        ),
        "stream": True,
    }

    start = time.perf_counter()

    first_content_time = None

    output_parts = []

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

            if response.status_code != 200:

                body = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

                print()
                print("=" * 120)
                print("HTTP REQUEST FAILURE")
                print("=" * 120)

                print(
                    "HTTP_STATUS =",
                    response.status_code,
                )

                print(
                    "REQUEST_PAYLOAD =",
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                    )[:4000],
                )

                print(
                    "RESPONSE_BODY =",
                    body,
                )

                print(
                    "LOCAL_PROMPT_TOKENS =",
                    chat_tokens(prompt),
                )

                raise RuntimeError(
                    f"vLLM API HTTP "
                    f"{response.status_code}"
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

                if content:

                    if first_content_time is None:
                        first_content_time = (
                            time.perf_counter()
                            - start
                        )

                    output_parts.append(
                        content
                    )

    total_s = (
        time.perf_counter()
        - start
    )

    if first_content_time is None:
        raise RuntimeError(
            "No output content observed."
        )

    full_output = "".join(
        output_parts
    )

    completion_tokens = max(
        1,
        output_tokens(
            full_output
        ),
    )

    prompt_tokens = chat_tokens(
        prompt
    )

    ttft_ms = (
        first_content_time
        * 1000.0
    )

    total_latency_ms = (
        total_s
        * 1000.0
    )

    if completion_tokens > 1:

        tpot_ms = (
            (
                total_latency_ms
                - ttft_ms
            )
            / (
                completion_tokens
                - 1
            )
        )

    else:
        tpot_ms = float("nan")

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": (
            total_latency_ms
        ),
        "tpot_ms": tpot_ms,
        "prompt_tokens": (
            prompt_tokens
        ),
        "completion_tokens": (
            completion_tokens
        ),
    }


def run_group(
    experiment,
    prompt_tokens,
    output_tokens_target,
    concurrency,
    request_count,
    prefix_mode,
):

    if prefix_mode == "shared":

        shared_prefix = (
            "Shared reusable context: "
            + (
                "LLM serving uses scheduling KV cache "
                "attention kernels GPU batching. "
                * 60
            )
        )

    else:
        shared_prefix = ""

    prompts = []

    for _ in range(
        request_count
    ):

        prompt, actual = make_prompt(
            target_tokens=prompt_tokens,
            nonce=uuid.uuid4().hex,
            shared_prefix=shared_prefix,
        )

        if (
            actual
            + output_tokens_target
            > 4000
        ):
            raise RuntimeError(
                "Constructed request too close "
                "to max_model_len."
            )

        prompts.append(prompt)


    # Shape warmup with unique request text.
    warm_prompt, warm_tokens = (
        make_prompt(
            target_tokens=prompt_tokens,
            nonce=uuid.uuid4().hex,
            shared_prefix=shared_prefix,
        )
    )

    print()
    print(
        "WARMUP_PROMPT_TARGET =",
        prompt_tokens,
    )

    print(
        "WARMUP_PROMPT_ACTUAL =",
        warm_tokens,
    )

    stream_request(
        warm_prompt,
        output_tokens_target,
    )


    wall_start = (
        time.perf_counter()
    )

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                stream_request,
                prompt,
                output_tokens_target,
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
        if x["tpot_ms"] == x["tpot_ms"]
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
            prompt_tokens
        ),
        "observed_prompt_tokens_median": (
            statistics.median(
                [
                    x["prompt_tokens"]
                    for x in results
                ]
            )
        ),
        "max_output_tokens": (
            output_tokens_target
        ),
        "concurrency": concurrency,
        "request_count": request_count,
        "prefix_mode": prefix_mode,
        "wall_s": wall_s,
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
            statistics.median(
                tpots
            )
            if tpots
            else float("nan")
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
    }


    records.append(row)

    print()
    print("-" * 120)

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


print(LINE)
print(
    "P1-M7 V3 "
    "VLLM SERVING CHARACTERIZATION"
)
print(LINE)


# =====================================================================
# 1. PROMPT LENGTH
# =====================================================================

print()
print(LINE)
print(
    "EXPERIMENT 1 — "
    "PROMPT LENGTH / PREFILL"
)
print(LINE)

for p in [
    128,
    512,
    2048,
]:

    run_group(
        experiment="prompt_length",
        prompt_tokens=p,
        output_tokens_target=32,
        concurrency=1,
        request_count=4,
        prefix_mode="unique",
    )


# =====================================================================
# 2. CONCURRENCY
# =====================================================================

print()
print(LINE)
print(
    "EXPERIMENT 2 — "
    "CONCURRENCY / CONTINUOUS BATCHING"
)
print(LINE)

for concurrency, count in [
    (1, 8),
    (4, 12),
    (8, 16),
    (16, 32),
]:

    run_group(
        experiment="concurrency",
        prompt_tokens=512,
        output_tokens_target=64,
        concurrency=concurrency,
        request_count=count,
        prefix_mode="unique",
    )


# =====================================================================
# 3. OUTPUT LENGTH
# =====================================================================

print()
print(LINE)
print(
    "EXPERIMENT 3 — "
    "OUTPUT LENGTH / DECODE"
)
print(LINE)

for out_len in [
    16,
    64,
    128,
]:

    run_group(
        experiment="output_length",
        prompt_tokens=512,
        output_tokens_target=out_len,
        concurrency=4,
        request_count=8,
        prefix_mode="unique",
    )


# =====================================================================
# 4. PREFIX CACHE
# =====================================================================

print()
print(LINE)
print(
    "EXPERIMENT 4 — "
    "PREFIX CACHE"
)
print(LINE)

run_group(
    experiment="prefix_cache",
    prompt_tokens=2048,
    output_tokens_target=32,
    concurrency=8,
    request_count=16,
    prefix_mode="unique",
)

run_group(
    experiment="prefix_cache",
    prompt_tokens=2048,
    output_tokens_target=32,
    concurrency=8,
    request_count=16,
    prefix_mode="shared",
)


# =====================================================================
# CSV
# =====================================================================

print()
print(LINE)
print("SAVE CSV")
print(LINE)

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fields = [
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
        fieldnames=fields,
    )

    writer.writeheader()
    writer.writerows(
        records
    )

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
The serving system is characterized along four axes:

1. Prompt length
   → primarily changes Prefill pressure.

2. Concurrency
   → exposes continuous batching and
     latency/throughput trade-offs.

3. Output length
   → increases Decode pressure.

4. Shared prefixes
   → expose prefix-cache reuse.

Primary metrics:

TTFT
    Time To First Token.

TPOT
    Time Per Output Token after TTFT.

Request throughput
    requests completed per second.

Output throughput
    generated tokens per second.

A valid serving benchmark must distinguish
throughput gains caused by batching from
gains caused by cache reuse.
""".strip()
)


print()
print(LINE)
print("FINAL VERDICT")
print(LINE)

print(
    "PROMPT_LENGTH_SWEEP = SUCCESS"
)

print(
    "CONCURRENCY_SWEEP = SUCCESS"
)

print(
    "OUTPUT_LENGTH_SWEEP = SUCCESS"
)

print(
    "PREFIX_CACHE_CHARACTERIZATION = SUCCESS"
)

print(
    "TTFT_CHARACTERIZATION = SUCCESS"
)

print(
    "TPOT_CHARACTERIZATION = SUCCESS"
)

print(
    "STRUCTURED_CSV_RESULT = SUCCESS"
)

print(
    "VERDICT = "
    "READY_FOR_P1_M8_OPTIMIZED_VS_EAGER_VLLM"
)
