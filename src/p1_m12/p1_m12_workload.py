import concurrent.futures
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M12_HOST"]
PORT = int(os.environ["P1_M12_PORT"])
MODEL = os.environ["P1_M12_MODEL"]
MODEL_PATH = os.environ["P1_M12_MODEL_PATH"]

PHASE = os.environ["P1_M12_PHASE"]
META_PATH = Path(os.environ["P1_M12_META"])

BASE = f"http://{HOST}:{PORT}"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
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


def output_token_count(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_prompt(
    target_tokens,
    long_output,
):
    prefix = (
        f"Unique request {uuid.uuid4().hex}. "
        "Analyze this synthetic AI infrastructure workload. "
    )

    filler = (
        "GPU inference scheduling KV cache attention "
        "memory bandwidth continuous batching kernel "
        "prefill decode throughput latency tensor compute. "
    )

    if long_output:
        suffix = (
            "Give a long detailed technical explanation. "
            "Continue discussing different systems aspects "
            "until the output limit is reached."
        )
    else:
        suffix = (
            "Give one very concise technical explanation."
        )

    lo = 0
    hi = 32

    while chat_tokens(
        prefix + filler * hi + suffix
    ) < target_tokens:

        hi *= 2

        if hi > 16384:
            raise RuntimeError(
                "Prompt construction runaway."
            )

    while lo < hi:
        mid = (lo + hi) // 2

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

    elapsed = (
        time.perf_counter()
        - start
    )

    usage = obj.get(
        "usage",
        {},
    )

    content = (
        obj.get(
            "choices",
            [{}],
        )[0]
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
    )

    prompt_tokens = int(
        usage.get(
            "prompt_tokens",
            chat_tokens(prompt),
        )
    )

    completion_tokens = int(
        usage.get(
            "completion_tokens",
            output_token_count(content),
        )
    )

    return {
        "latency_ms": elapsed * 1000.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def run_workload():

    if PHASE == "prefill":

        target_prompt = 2048
        output_target = 16
        concurrency = 8
        request_count = 16
        long_output = False

    elif PHASE == "decode":

        target_prompt = 128
        output_target = 512
        concurrency = 8
        request_count = 8
        long_output = True

    else:
        raise RuntimeError(
            f"Unknown PHASE={PHASE}"
        )

    prompts = [
        build_prompt(
            target_prompt,
            long_output,
        )
        for _ in range(
            request_count
        )
    ]

    print(
        "PHASE =",
        PHASE,
    )

    print(
        "TARGET_PROMPT_TOKENS =",
        target_prompt,
    )

    print(
        "TARGET_OUTPUT_TOKENS =",
        output_target,
    )

    print(
        "CONCURRENCY =",
        concurrency,
    )

    print(
        "REQUEST_COUNT =",
        request_count,
    )

    start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                prompt,
                output_target,
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
        - start
    )

    prompt_total = sum(
        x["prompt_tokens"]
        for x in results
    )

    output_total = sum(
        x["output_tokens"]
        for x in results
    )

    latencies = [
        x["latency_ms"]
        for x in results
    ]

    meta = {
        "phase": PHASE,
        "target_prompt_tokens": (
            target_prompt
        ),
        "target_output_tokens": (
            output_target
        ),
        "concurrency": concurrency,
        "request_count": (
            request_count
        ),
        "wall_s": wall,
        "prompt_tokens_total": (
            prompt_total
        ),
        "output_tokens_total": (
            output_total
        ),
        "prompt_tokens_median": (
            statistics.median(
                [
                    x["prompt_tokens"]
                    for x in results
                ]
            )
        ),
        "output_tokens_median": (
            statistics.median(
                [
                    x["output_tokens"]
                    for x in results
                ]
            )
        ),
        "latency_median_ms": (
            statistics.median(
                latencies
            )
        ),
        "output_tokens_per_s_profiled": (
            output_total / wall
        ),
    }

    META_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    META_PATH.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    for k, v in meta.items():

        if isinstance(
            v,
            float,
        ):
            print(
                f"{k} = {v:.6f}"
            )
        else:
            print(
                f"{k} = {v}"
            )

    print(
        "PROFILE_WORKLOAD = SUCCESS"
    )


print("=" * 120)
print(
    "P1-M12 PHASE-SEPARATED "
    "PRODUCTION WORKLOAD"
)
print("=" * 120)

run_workload()
