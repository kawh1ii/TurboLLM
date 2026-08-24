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
