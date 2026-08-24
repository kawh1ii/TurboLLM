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
