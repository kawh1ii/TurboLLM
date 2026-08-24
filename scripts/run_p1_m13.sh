#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"
VLLM="${ENV}/bin/vllm"

HF_HOME="${BASE}/caches/huggingface"

SRC_DIR="${PROJECT}/src/p1_m13"
RESULT_DIR="${PROJECT}/results/p1_m13"
ARTIFACT_DIR="${BASE}/artifacts"

HOST="127.0.0.1"
PORT="8000"
MODEL_NAME="qwen25-05b"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M13_CUDAGRAPH_ABLATION_${TS}.txt"

CSV="${RESULT_DIR}/P1_M13_CUDAGRAPH_ABLATION_${TS}.csv"

CLIENT="${SRC_DIR}/p1_m13_client.py"
COMPARE="${SRC_DIR}/p1_m13_compare.py"

SERVER_PID=""

LINE="========================================================================================================================"


cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] \
        && kill -0 "${SERVER_PID}" 2>/dev/null
    then
        echo "STOPPING_SERVER_PID = ${SERVER_PID}"

        kill -TERM -- "-${SERVER_PID}" \
            2>/dev/null \
            || kill -TERM "${SERVER_PID}" \
            2>/dev/null \
            || true

        for _ in $(seq 1 30); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL -- "-${SERVER_PID}" \
                2>/dev/null \
                || kill -KILL "${SERVER_PID}" \
                2>/dev/null \
                || true
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
PY
}


create_compare() {

cat > "${COMPARE}" <<'PY'
import csv
import sys
from collections import defaultdict


path = sys.argv[1]


with open(
    path,
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


data = defaultdict(dict)


for row in rows:

    data[
        row["phase"]
    ][
        row["mode"]
    ] = row


required_modes = [
    "none",
    "piecewise",
    "full_and_piecewise",
]


for phase in [
    "prefill",
    "decode",
]:

    for mode in required_modes:

        if mode not in data[
            phase
        ]:

            raise RuntimeError(
                f"Missing result: "
                f"{phase}/{mode}"
            )


def num(
    row,
    key,
):

    return float(
        row[key]
    )


print("=" * 120)
print(
    "P1-M13 CUDA GRAPH ABLATION COMPARISON"
)
print("=" * 120)


for phase in [
    "prefill",
    "decode",
]:

    print()
    print("=" * 120)
    print(
        f"PHASE = {phase.upper()}"
    )
    print("=" * 120)

    baseline = data[
        phase
    ][
        "none"
    ]

    piecewise = data[
        phase
    ][
        "piecewise"
    ]

    full = data[
        phase
    ][
        "full_and_piecewise"
    ]


    for mode in required_modes:

        row = data[
            phase
        ][
            mode
        ]

        print()
        print(
            f"MODE = {mode}"
        )

        print(
            "MEDIAN_TTFT_MS =",
            f"{num(row, 'median_ttft_ms'):.3f}",
        )

        print(
            "MEDIAN_TPOT_MS =",
            f"{num(row, 'median_tpot_ms'):.3f}",
        )

        print(
            "MEDIAN_LATENCY_MS =",
            f"{num(row, 'median_latency_ms'):.3f}",
        )

        print(
            "OUTPUT_TOKENS_S =",
            f"{num(row, 'output_tokens_per_s'):.3f}",
        )

        print(
            "REQUESTS_S =",
            f"{num(row, 'request_throughput_rps'):.3f}",
        )

        print(
            "READY_VRAM_MIB =",
            f"{num(row, 'ready_vram_mib'):.1f}",
        )

        print(
            "STARTUP_S =",
            f"{num(row, 'startup_s'):.3f}",
        )


    none_out = num(
        baseline,
        "output_tokens_per_s",
    )

    piece_out = num(
        piecewise,
        "output_tokens_per_s",
    )

    full_out = num(
        full,
        "output_tokens_per_s",
    )


    none_tpot = num(
        baseline,
        "median_tpot_ms",
    )

    piece_tpot = num(
        piecewise,
        "median_tpot_ms",
    )

    full_tpot = num(
        full,
        "median_tpot_ms",
    )


    print()
    print("-" * 120)

    print(
        "PIECEWISE_OUTPUT_SPEEDUP_VS_NONE =",
        f"{piece_out / none_out:.3f}x",
    )

    print(
        "FULL_OUTPUT_SPEEDUP_VS_NONE =",
        f"{full_out / none_out:.3f}x",
    )

    print(
        "FULL_OUTPUT_SPEEDUP_VS_PIECEWISE =",
        f"{full_out / piece_out:.3f}x",
    )

    print(
        "PIECEWISE_TPOT_SPEEDUP_VS_NONE =",
        f"{none_tpot / piece_tpot:.3f}x",
    )

    print(
        "FULL_TPOT_SPEEDUP_VS_NONE =",
        f"{none_tpot / full_tpot:.3f}x",
    )

    print(
        "FULL_TPOT_SPEEDUP_VS_PIECEWISE =",
        f"{piece_tpot / full_tpot:.3f}x",
    )


    best_mode = max(
        required_modes,
        key=lambda mode: num(
            data[
                phase
            ][
                mode
            ],
            "output_tokens_per_s",
        ),
    )

    print(
        "BEST_OUTPUT_THROUGHPUT_MODE =",
        best_mode,
    )


print()
print("=" * 120)
print("TARGETED DECODE ANALYSIS")
print("=" * 120)


decode_none = data[
    "decode"
][
    "none"
]

decode_piece = data[
    "decode"
][
    "piecewise"
]

decode_full = data[
    "decode"
][
    "full_and_piecewise"
]


decode_none_out = num(
    decode_none,
    "output_tokens_per_s",
)

decode_piece_out = num(
    decode_piece,
    "output_tokens_per_s",
)

decode_full_out = num(
    decode_full,
    "output_tokens_per_s",
)


print(
    "DECODE_FULL_GRAPH_INCREMENTAL_SPEEDUP_VS_PIECEWISE =",
    f"{decode_full_out / decode_piece_out:.3f}x",
)

print(
    "DECODE_FULL_GRAPH_SPEEDUP_VS_NO_GRAPH =",
    f"{decode_full_out / decode_none_out:.3f}x",
)


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print(
"""
This experiment keeps torch.compile enabled in every mode.

Only CUDA Graph execution strategy changes:

    NONE
        no CUDA Graph execution

    PIECEWISE
        piecewise CUDA Graph execution

    FULL_AND_PIECEWISE
        full CUDA Graph execution for Decode
        and piecewise execution for Prefill/mixed batches

Therefore this is narrower than the earlier:

    eager vs optimized

comparison.

P1-M8 changed both compilation and CUDA Graph execution.

P1-M13 isolates the CUDA Graph contribution while
preserving the compiled model path.

The winning mode is determined from measurements.
The experiment does NOT require FULL_AND_PIECEWISE
to win in order to be considered successful.
""".strip()
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)

print(
    "MATCHED_PREFILL_ABLATION = SUCCESS"
)

print(
    "MATCHED_DECODE_ABLATION = SUCCESS"
)

print(
    "TORCH_COMPILE_HELD_CONSTANT = TRUE"
)

print(
    "CUDA_GRAPH_MODE_ISOLATED = TRUE"
)

print(
    "THREE_MODE_COMPARISON = SUCCESS"
)

print(
    "VERDICT = "
    "READY_FOR_P1_M14_OPTIMIZATION_DECISION"
)
PY
}


run_mode() {

    MODE="$1"

    cleanup_server

    pkill -f "${ENV}/bin/vllm serve" \
        2>/dev/null || true

    pkill -f "vllm.*EngineCore" \
        2>/dev/null || true

    sleep 2

    LOG="${RESULT_DIR}/P1_M13_${MODE}_SERVER_${TS}.log"

    MODE_UPPER="$(
        echo "${MODE}" \
        | tr '[:lower:]' '[:upper:]'
    )"

    echo
    echo "${LINE}"
    echo "CUDA GRAPH MODE = ${MODE}"
    echo "${LINE}"

    echo "TORCH_COMPILE_MODE = 3"
    echo "REQUESTED_CUDAGRAPH_MODE = ${MODE}"

    rm -f "${LOG}"

    START_NS="$(
        date +%s%N
    )"


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
        --max-num-seqs 32 \
        --no-enable-prefix-caching \
        "-cc.mode=3" \
        "-cc.cudagraph_mode=${MODE}" \
        > "${LOG}" 2>&1 &


    SERVER_PID=$!

    echo "SERVER_PID = ${SERVER_PID}"
    echo "SERVER_LOG = ${LOG}"


    READY="FALSE"


    for SECOND in $(seq 1 900); do

        if ! kill -0 "${SERVER_PID}" \
            2>/dev/null
        then

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
            READY_SECOND="${SECOND}"

            break
        fi


        if [ $((SECOND % 60)) -eq 0 ]; then

            echo "WAITING_SECOND = ${SECOND}"

            tail -n 50 \
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


    END_NS="$(
        date +%s%N
    )"


    STARTUP_S="$(
        "${PYTHON}" - \
            "${START_NS}" \
            "${END_NS}" \
        <<'PY'
import sys

start = int(
    sys.argv[1]
)

end = int(
    sys.argv[2]
)

print(
    (end - start)
    / 1e9
)
PY
    )"


    READY_VRAM="$(
        nvidia-smi \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d ' '
    )"


    ACTUAL_MODE="$(
        "${PYTHON}" - \
            "${LOG}" \
        <<'PY'
import re
import sys

text = open(
    sys.argv[1],
    "r",
    encoding="utf-8",
    errors="replace",
).read()

matches = re.findall(
    r"cudagraph_mode['\"]?"
    r"\s*:\s*"
    r"<CUDAGraphMode\.([A-Z_]+)",
    text,
)

if not matches:
    matches = re.findall(
        r"cudagraph_mode="
        r"(?:CUDAGraphMode\.)?"
        r"([A-Z_]+)",
        text,
    )

print(
    matches[-1].lower()
    if matches
    else "unknown"
)
PY
    )"


    echo "SERVER_HEALTH = READY"
    echo "SECONDS_WAITED = ${READY_SECOND}"
    echo "STARTUP_S = ${STARTUP_S}"
    echo "READY_VRAM_MIB = ${READY_VRAM}"
    echo "ACTUAL_CUDAGRAPH_MODE = ${ACTUAL_MODE}"


    if [ "${ACTUAL_MODE}" != "${MODE}" ]; then

        echo "ERROR = CUDAGRAPH_MODE_MISMATCH"

        echo
        echo "COMPILATION CONFIG EVIDENCE:"

        grep -Ei \
            'compilation_config|cudagraph|cuda graph|compile' \
            "${LOG}" \
            | tail -n 150 \
            || true

        return 12
    fi


    # ==============================================================================================
    # WARM BOTH WORKLOAD SHAPES
    # ==============================================================================================

    for PHASE in \
        prefill \
        decode
    do

        echo
        echo "WARMUP_PHASE = ${PHASE}"

        P1_M13_HOST="${HOST}" \
        P1_M13_PORT="${PORT}" \
        P1_M13_MODEL="${MODEL_NAME}" \
        P1_M13_MODEL_PATH="${MODEL_PATH}" \
        P1_M13_MODE="${MODE}" \
        P1_M13_ACTUAL_MODE="${ACTUAL_MODE}" \
        P1_M13_PHASE="${PHASE}" \
        P1_M13_CSV="${CSV}" \
        P1_M13_RAW="${RESULT_DIR}/unused.json" \
        P1_M13_STARTUP_S="${STARTUP_S}" \
        P1_M13_READY_VRAM_MIB="${READY_VRAM}" \
        P1_M13_WARMUP_ONLY="1" \
        "${PYTHON}" \
            "${CLIENT}"
    done


    # ==============================================================================================
    # MEASURE PREFILL + DECODE
    # ==============================================================================================

    for PHASE in \
        prefill \
        decode
    do

        echo
        echo "${LINE}"
        echo "MEASURE MODE=${MODE} PHASE=${PHASE}"
        echo "${LINE}"

                RAW="${RESULT_DIR}/P1_M13_${MODE}_${PHASE}_${TS}.json"


        P1_M13_HOST="${HOST}" \
        P1_M13_PORT="${PORT}" \
        P1_M13_MODEL="${MODEL_NAME}" \
        P1_M13_MODEL_PATH="${MODEL_PATH}" \
        P1_M13_MODE="${MODE}" \
        P1_M13_ACTUAL_MODE="${ACTUAL_MODE}" \
        P1_M13_PHASE="${PHASE}" \
        P1_M13_CSV="${CSV}" \
        P1_M13_RAW="${RAW}" \
        P1_M13_STARTUP_S="${STARTUP_S}" \
        P1_M13_READY_VRAM_MIB="${READY_VRAM}" \
        P1_M13_WARMUP_ONLY="0" \
        "${PYTHON}" \
            "${CLIENT}"
    done


    echo
    echo "SERVER OPTIMIZATION EVIDENCE:"

    grep -Ei \
        'cudagraph|cuda graph|compile|capture|AOT' \
        "${LOG}" \
        | tail -n 180 \
        || true


    cleanup_server

    echo "MODE_${MODE^^}_COMPLETE = TRUE"
}


main() {

    trap cleanup_server EXIT
    set -e

    export CUDA_HOME="/usr/local/cuda-12.8"

    export PATH="${CUDA_HOME}/bin:${ENV}/bin:${PATH}"

    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"


    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M13"
    echo "TARGETED CUDA GRAPH OPTIMIZATION ABLATION"
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
    # 2. VERIFY COMPILATION CONFIG
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "2. VERIFY COMPILATION CONFIG SUPPORT"
    echo "${LINE}"

    HELP="$(
        NO_COLOR=1 \
        "${VLLM}" serve --help=all \
        2>&1
    )"


    if echo "${HELP}" \
        | grep -q -- \
        "--compilation-config"
    then

        echo "COMPILATION_CONFIG = SUPPORTED"

    else

        echo "COMPILATION_CONFIG = NOT_SUPPORTED"

        return 2
    fi


    echo
    echo "TARGET CONFIGURATION:"

    echo "TORCH_COMPILE_MODE = 3"

    echo "CUDA_GRAPH_MODES = none,piecewise,full_and_piecewise"


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
    # 4. CREATE CLIENT / COMPARATOR
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "4. CREATE BENCHMARK CLIENT"
    echo "${LINE}"

    create_client
    create_compare

    rm -f "${CSV}"

    echo "CLIENT = ${CLIENT}"
    echo "COMPARE = ${COMPARE}"


    # ==============================================================================================
    # 5. NO CUDA GRAPH
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "5. TORCH.COMPILE + CUDA GRAPH NONE"
    echo "${LINE}"

    run_mode \
        "none"


    # ==============================================================================================
    # 6. PIECEWISE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "6. TORCH.COMPILE + PIECEWISE CUDA GRAPH"
    echo "${LINE}"

    run_mode \
        "piecewise"


    # ==============================================================================================
    # 7. FULL + PIECEWISE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "7. TORCH.COMPILE + FULL_AND_PIECEWISE CUDA GRAPH"
    echo "${LINE}"

    run_mode \
        "full_and_piecewise"


    # ==============================================================================================
    # 8. RESULTS
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "8. STRUCTURED RESULTS"
    echo "${LINE}"

    cat "${CSV}"


    # ==============================================================================================
    # 9. COMPARISON
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "9. MATCHED CUDA GRAPH COMPARISON"
    echo "${LINE}"

    "${PYTHON}" \
        "${COMPARE}" \
        "${CSV}"


    # ==============================================================================================
    # 10. KNOWLEDGE
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "P1-M13 KNOWLEDGE CHECKPOINT"
    echo "${LINE}"

    echo "
P1-M13 IS AN ABLATION, NOT A PROFILER RUN.

P1-M12 identified repeated production execution
as a major Decode characteristic.

P1-M13 therefore isolates CUDA Graph strategy while
holding torch.compile enabled.

CONTROL:

    compilation mode = VLLM_COMPILE / mode 3

VARIABLE:

    cudagraph_mode

        NONE
        PIECEWISE
        FULL_AND_PIECEWISE

WORKLOADS:

    Prefill-heavy
        ~2048 prompt tokens
        16 output tokens
        concurrency 8

    Decode-heavy
        ~128 prompt tokens
        512 output tokens
        concurrency 8


WHY THIS MATTERS

CUDA Graphs reduce repeated CPU-side launch overhead by
capturing GPU execution into reusable graphs.

They do not make the mathematical GEMM itself magically
faster.

The benefit should therefore depend on workload shape,
launch frequency, graph compatibility and batch structure.

The correct result is whatever the matched measurement
shows. No mode is required to win in advance.
"


    # ==============================================================================================
    # 11. FINAL
    # ==============================================================================================

    echo
    echo "${LINE}"
    echo "FINAL VERDICT"
    echo "${LINE}"

    ROW_COUNT="$(
        tail -n +2 "${CSV}" \
        | wc -l
    )"

    echo "RESULT_ROW_COUNT = ${ROW_COUNT}"

    if [ "${ROW_COUNT}" -ne 6 ]; then

        echo "VERDICT = P1_M13_INCOMPLETE"

        return 20
    fi


    echo "NO_GRAPH_BASELINE = SUCCESS"
    echo "PIECEWISE_GRAPH = SUCCESS"
    echo "FULL_AND_PIECEWISE_GRAPH = SUCCESS"
    echo "PREFILL_MATCHED_COMPARISON = SUCCESS"
    echo "DECODE_MATCHED_COMPARISON = SUCCESS"
    echo "TORCH_COMPILE_HELD_CONSTANT = TRUE"
    echo "PREFIX_CACHE_HELD_OFF = TRUE"
    echo "VERDICT = READY_FOR_P1_M14_OPTIMIZATION_DECISION"


    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "CSV_RESULT = ${CSV}"
    echo "CLIENT = ${CLIENT}"
    echo "COMPARE = ${COMPARE}"


    echo
    echo "${LINE}"
    echo "END OF P1-M13"
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
