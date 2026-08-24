import concurrent.futures
import json
import os
import time

import httpx


HOST = os.environ["P1_M11B_HOST"]
PORT = int(os.environ["P1_M11B_PORT"])
MODEL = os.environ["P1_M11B_MODEL"]

BASE = f"http://{HOST}:{PORT}"


def request_once(i, max_tokens=64):
    # Unique text near the beginning avoids turning the
    # experiment into a prefix-cache benchmark.
    content = (
        f"Unique profiling request {i}. "
        "Analyze GPU inference serving architecture. "
        + (
            "Discuss scheduler KV cache attention kernels "
            "memory bandwidth batching prefill decode latency "
            "throughput CUDA graph tensor computation. "
            * 25
        )
        + "Give a detailed technical answer."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }

    start = time.perf_counter()

    with httpx.Client(
        trust_env=False,
        timeout=300.0,
    ) as client:

        response = client.post(
            BASE + "/v1/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        obj = response.json()

    elapsed = time.perf_counter() - start

    usage = obj.get("usage", {})

    return {
        "request": i,
        "latency_ms": elapsed * 1000.0,
        "prompt_tokens": usage.get(
            "prompt_tokens",
            0,
        ),
        "output_tokens": usage.get(
            "completion_tokens",
            0,
        ),
    }


def run_batch(
    base_id,
    concurrency,
    count,
    output_tokens,
):
    start = time.perf_counter()

    rows = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                base_id + i,
                output_tokens,
            )
            for i in range(count)
        ]

        for future in (
            concurrent.futures.as_completed(
                futures
            )
        ):
            rows.append(
                future.result()
            )

    wall = time.perf_counter() - start

    output_total = sum(
        x["output_tokens"]
        for x in rows
    )

    prompt_total = sum(
        x["prompt_tokens"]
        for x in rows
    )

    print(
        "REQUEST_COUNT =",
        count,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "WALL_S =",
        f"{wall:.6f}",
    )

    print(
        "PROMPT_TOKENS =",
        prompt_total,
    )

    print(
        "OUTPUT_TOKENS =",
        output_total,
    )

    print(
        "OUTPUT_TOKENS_PER_S =",
        f"{output_total / wall:.3f}",
    )


print("=" * 120)
print("P1-M11B PRODUCTION VLLM WORKLOAD")
print("=" * 120)

# Moderate concurrency is enough to expose real
# batched attention / GEMM / normalization / sampling kernels.
run_batch(
    base_id=100000,
    concurrency=8,
    count=16,
    output_tokens=64,
)

print("PROFILE_WORKLOAD = SUCCESS")
