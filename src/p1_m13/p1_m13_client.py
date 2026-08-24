import concurrent.futures
import csv
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer


HOST = os.environ["P1_M13_HOST"]
PORT = int(os.environ["P1_M13_PORT"])

MODEL = os.environ["P1_M13_MODEL"]
MODEL_PATH = os.environ["P1_M13_MODEL_PATH"]

MODE = os.environ["P1_M13_MODE"]
ACTUAL_MODE = os.environ["P1_M13_ACTUAL_MODE"]

PHASE = os.environ["P1_M13_PHASE"]

CSV_PATH = Path(
    os.environ["P1_M13_CSV"]
)

RAW_PATH = Path(
    os.environ["P1_M13_RAW"]
)

STARTUP_S = float(
    os.environ["P1_M13_STARTUP_S"]
)

READY_VRAM_MIB = float(
    os.environ["P1_M13_READY_VRAM_MIB"]
)

WARMUP_ONLY = (
    os.environ.get(
        "P1_M13_WARMUP_ONLY",
        "0",
    )
    == "1"
)

BASE_URL = f"http://{HOST}:{PORT}"

TRIALS = 3

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


def percentile(
    values,
    q,
):
    xs = sorted(values)

    if not xs:
        return float("nan")

    pos = (
        len(xs) - 1
    ) * q

    lo = int(pos)

    hi = min(
        lo + 1,
        len(xs) - 1,
    )

    frac = pos - lo

    return (
        xs[lo]
        * (1.0 - frac)
        + xs[hi]
        * frac
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

        if (
            chat_tokens(text)
            < target_tokens
        ):
            lo = mid + 1
        else:
            hi = mid

    candidates = []

    for repeats in [
        max(0, lo - 2),
        max(0, lo - 1),
        lo,
        lo + 1,
    ]:

        text = (
            prefix
            + filler * repeats
            + suffix
        )

        n = chat_tokens(text)

        candidates.append(
            (
                abs(
                    n
                    - target_tokens
                ),
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

    total_s = (
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
        completion_tokens(
            output_text
        ),
    )

    ttft_ms = (
        first
        * 1000.0
    )

    latency_ms = (
        total_s
        * 1000.0
    )

    if n_output > 1:

        tpot_ms = (
            latency_ms
            - ttft_ms
        ) / (
            n_output
            - 1
        )

    else:

        tpot_ms = float("nan")

    return {
        "ttft_ms": ttft_ms,

        "latency_ms": (
            latency_ms
        ),

        "tpot_ms": (
            tpot_ms
        ),

        "prompt_tokens": (
            chat_tokens(prompt)
        ),

        "output_tokens": (
            n_output
        ),
    }


def workload_config():

    if PHASE == "prefill":

        return {
            "target_prompt": 2048,
            "target_output": 16,
            "concurrency": 8,
            "request_count": 16,
            "long_output": False,
        }

    if PHASE == "decode":

        return {
            "target_prompt": 128,
            "target_output": 512,
            "concurrency": 8,
            "request_count": 8,
            "long_output": True,
        }

    raise RuntimeError(
        f"Unknown phase {PHASE}"
    )


def execute_trial(
    cfg,
):

    prompts = [
        build_prompt(
            cfg[
                "target_prompt"
            ],
            cfg[
                "long_output"
            ],
        )
        for _ in range(
            cfg[
                "request_count"
            ]
        )
    ]

    results = []

    wall_start = (
        time.perf_counter()
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg[
            "concurrency"
        ]
    ) as executor:

        futures = [
            executor.submit(
                request_once,
                prompt,
                cfg[
                    "target_output"
                ],
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

    wall_s = (
        time.perf_counter()
        - wall_start
    )

    ttfts = [
        x["ttft_ms"]
        for x in results
    ]

    tpots = [
        x["tpot_ms"]
        for x in results
        if not math.isnan(
            x["tpot_ms"]
        )
    ]

    latencies = [
        x["latency_ms"]
        for x in results
    ]

    prompt_total = sum(
        x["prompt_tokens"]
        for x in results
    )

    output_total = sum(
        x["output_tokens"]
        for x in results
    )

    return {
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

        "median_tpot_ms": (
            statistics.median(
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
            cfg[
                "request_count"
            ]
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

        "total_tokens_per_s": (
            (
                prompt_total
                + output_total
            )
            / wall_s
        ),

        "prompt_tokens_total": (
            prompt_total
        ),

        "output_tokens_total": (
            output_total
        ),
    }


cfg = workload_config()

print("=" * 120)
print("P1-M13 CUDA GRAPH ABLATION CLIENT")
print("=" * 120)

print(
    "MODE =",
    MODE,
)

print(
    "ACTUAL_MODE =",
    ACTUAL_MODE,
)

print(
    "PHASE =",
    PHASE,
)

print(
    "TARGET_PROMPT_TOKENS =",
    cfg[
        "target_prompt"
    ],
)

print(
    "TARGET_OUTPUT_TOKENS =",
    cfg[
        "target_output"
    ],
)

print(
    "CONCURRENCY =",
    cfg[
        "concurrency"
    ],
)

print(
    "REQUEST_COUNT =",
    cfg[
        "request_count"
    ],
)


if WARMUP_ONLY:

    trial = execute_trial(
        cfg
    )

    print(
        "WARMUP_WALL_S =",
        f"{trial['wall_s']:.6f}",
    )

    print(
        "WARMUP = SUCCESS"
    )

    raise SystemExit(0)


trials = []


for i in range(
    TRIALS
):

    result = execute_trial(
        cfg
    )

    result[
        "trial"
    ] = i + 1

    trials.append(
        result
    )

    print()
    print(
        f"TRIAL_{i + 1}"
    )

    for key, value in (
        result.items()
    ):

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


def median_trial_metric(
    name,
):

    return statistics.median(
        [
            x[name]
            for x in trials
        ]
    )


row = {
    "mode": MODE,

    "actual_cudagraph_mode": (
        ACTUAL_MODE
    ),

    "phase": PHASE,

    "startup_s": (
        STARTUP_S
    ),

    "ready_vram_mib": (
        READY_VRAM_MIB
    ),

    "target_prompt_tokens": (
        cfg[
            "target_prompt"
        ]
    ),

    "target_output_tokens": (
        cfg[
            "target_output"
        ]
    ),

    "concurrency": (
        cfg[
            "concurrency"
        ]
    ),

    "request_count": (
        cfg[
            "request_count"
        ]
    ),

    "trial_count": (
        TRIALS
    ),

    "median_ttft_ms": (
        median_trial_metric(
            "median_ttft_ms"
        )
    ),

    "p95_ttft_ms": (
        median_trial_metric(
            "p95_ttft_ms"
        )
    ),

    "median_tpot_ms": (
        median_trial_metric(
            "median_tpot_ms"
        )
    ),

    "median_latency_ms": (
        median_trial_metric(
            "median_latency_ms"
        )
    ),

    "p95_latency_ms": (
        median_trial_metric(
            "p95_latency_ms"
        )
    ),

    "request_throughput_rps": (
        median_trial_metric(
            "request_throughput_rps"
        )
    ),

    "prompt_tokens_per_s": (
        median_trial_metric(
            "prompt_tokens_per_s"
        )
    ),

    "output_tokens_per_s": (
        median_trial_metric(
            "output_tokens_per_s"
        )
    ),

    "total_tokens_per_s": (
        median_trial_metric(
            "total_tokens_per_s"
        )
    ),
}


print()
print("=" * 120)
print("AGGREGATED RESULT")
print("=" * 120)


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


RAW_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_PATH.write_text(
    json.dumps(
        {
            "aggregate": row,
            "trials": trials,
        },
        indent=2,
    ),
    encoding="utf-8",
)


fields = list(
    row.keys()
)


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

    writer.writerow(
        row
    )


print(
    "RAW_RESULT =",
    RAW_PATH,
)

print(
    "CSV_RESULT =",
    CSV_PATH,
)

print(
    "PHASE_RESULT = SUCCESS"
)
