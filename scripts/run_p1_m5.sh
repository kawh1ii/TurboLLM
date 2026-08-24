#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV_DIR="${BASE}/envs/project1"

SRC_DIR="${PROJECT}/src/p1_m5"
RESULT_DIR="${PROJECT}/results/p1_m5"
ARTIFACT_DIR="${BASE}/artifacts"

HF_HOME="${BASE}/caches/huggingface"

MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}" \
    "${HF_HOME}" \
    "${BASE}/envs"

TXT="${ARTIFACT_DIR}/P1_M5_TRANSFORMER_LLM_INFERENCE_${TS}.txt"
CSV="${RESULT_DIR}/P1_M5_LLM_INFERENCE_${TS}.csv"
PY="${SRC_DIR}/transformer_llm_inference.py"

# Keep an untouched copy of terminal stdout.
exec 3>&1

# Everything produced by this run is saved in TXT.
exec > >(tee "${TXT}") 2>&1

finish() {
    STATUS=$?

    trap - EXIT

    echo
    echo "========================================================================================================================"
    echo "SCRIPT COMPLETION"
    echo "========================================================================================================================"
    echo "EXIT_CODE = ${STATUS}"

    if [ "${STATUS}" -eq 0 ]; then
        echo "SCRIPT_STATUS = SUCCESS"
    else
        echo "SCRIPT_STATUS = FAILED"
    fi

    # Restore terminal stdout/stderr.
    exec 1>&3 2>&3

    # FINAL TERMINAL LINE:
    # absolute TXT path only.
    printf '%s\n' "${TXT}"

    exec 3>&-

    exit "${STATUS}"
}

trap finish EXIT

set -e

LINE="========================================================================================================================"

echo "${LINE}"
echo "AI INFRA PROJECT 1 - P1-M5"
echo "TRANSFORMER / LLM INFERENCE / PREFILL / DECODE / KV CACHE"
echo "${LINE}"

echo
echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "MODEL_ID = ${MODEL_ID}"
echo "HF_HOME = ${HF_HOME}"

# -------------------------------------------------------------------------------------------------
# 1. Proxy
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "1. NETWORK / PROXY"
echo "${LINE}"

if [ -f /root/autodl-tmp/mihomo/proxy_env.sh ]; then
    source /root/autodl-tmp/mihomo/proxy_env.sh
fi

echo "HTTPS_PROXY = ${HTTPS_PROXY:-NOT_SET}"

GH_CODE="$(
    curl \
        -L \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        --connect-timeout 10 \
        --max-time 30 \
        https://github.com \
    || true
)"

HF_CODE="$(
    curl \
        -L \
        -sS \
        -o /dev/null \
        -w '%{http_code}' \
        --connect-timeout 10 \
        --max-time 30 \
        https://huggingface.co \
    || true
)"

echo "GITHUB_HTTP = ${GH_CODE}"
echo "HUGGINGFACE_HTTP = ${HF_CODE}"

# -------------------------------------------------------------------------------------------------
# 2. Project 1 isolated environment
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "2. PROJECT 1 ISOLATED PYTHON ENVIRONMENT"
echo "${LINE}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then

    echo "PROJECT1_ENV = NOT_FOUND"
    echo "ACTION = CLONE_WORKING_BASE_ENVIRONMENT"

    source /root/miniconda3/etc/profile.d/conda.sh

    conda create \
        -y \
        --prefix "${ENV_DIR}" \
        --clone /root/miniconda3

    echo "PROJECT1_ENV_CREATE = SUCCESS"

else

    echo "PROJECT1_ENV = EXISTING"

fi

PYTHON="${ENV_DIR}/bin/python"

echo "PROJECT1_PYTHON = ${PYTHON}"

"${PYTHON}" --version

"${PYTHON}" - <<'PY'
import torch

print("TORCH =", torch.__version__)
print("TORCH_CUDA =", torch.version.cuda)
print("CUDA_AVAILABLE =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
PY

# -------------------------------------------------------------------------------------------------
# 3. Install only the packages needed for this milestone
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "3. TRANSFORMERS DEPENDENCIES"
echo "${LINE}"

NEED_INSTALL="$(
"${PYTHON}" - <<'PY'
try:
    import transformers
    import accelerate
    import safetensors
    print("FALSE")
except Exception:
    print("TRUE")
PY
)"

echo "DEPENDENCY_INSTALL_REQUIRED = ${NEED_INSTALL}"

if [ "${NEED_INSTALL}" = "TRUE" ]; then

    "${PYTHON}" -m pip install \
        --upgrade \
        "transformers>=4.48,<5" \
        "accelerate>=1,<2" \
        "safetensors>=0.4" \
        sentencepiece

fi

echo
echo "INSTALLED_VERSIONS:"

"${PYTHON}" - <<'PY'
import torch
import transformers
import accelerate
import safetensors

print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
print("safetensors =", safetensors.__version__)
PY

# Freeze the exact Project 1 environment.
"${PYTHON}" -m pip freeze \
    > "${PROJECT}/docs/project1_environment_pip_freeze.txt"

# -------------------------------------------------------------------------------------------------
# 4. Write benchmark
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "4. CREATE P1-M5 BENCHMARK"
echo "${LINE}"

cat > "${PY}" <<'PY'
import csv
import gc
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


LINE = "=" * 120

MODEL_ID = os.environ["P1_M5_MODEL"]
CSV_PATH = Path(os.environ["P1_M5_CSV"])

DEVICE = "cuda"
DTYPE = torch.float16

records = []


def add(
    experiment,
    variant,
    seq_len,
    value,
    unit,
    notes="",
):
    records.append(
        {
            "experiment": experiment,
            "variant": variant,
            "seq_len": seq_len,
            "value": value,
            "unit": unit,
            "notes": notes,
        }
    )


def cuda_sync():
    torch.cuda.synchronize()


def allocated_mib():
    return torch.cuda.memory_allocated() / 1024**2


def reserved_mib():
    return torch.cuda.memory_reserved() / 1024**2


def median_prefill_ms(
    model,
    vocab_size,
    seq_len,
    warmup=3,
    repeats=10,
):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            out = model(
                input_ids=input_ids,
                use_cache=True,
            )
            del out

        cuda_sync()

        times = []

        for _ in range(repeats):
            start = torch.cuda.Event(
                enable_timing=True
            )
            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            out = model(
                input_ids=input_ids,
                use_cache=True,
            )

            end.record()
            end.synchronize()

            times.append(
                start.elapsed_time(end)
            )

            del out

    del input_ids
    gc.collect()

    return statistics.median(times)


def kv_cache_measurement(
    model,
    config,
    vocab_size,
    seq_len,
):
    torch.cuda.empty_cache()
    gc.collect()
    cuda_sync()

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
        dtype=torch.long,
        device=DEVICE,
    )

    before = torch.cuda.memory_allocated()

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            use_cache=True,
        )

    cuda_sync()

    past = out.past_key_values

    # Remove logits/output object while keeping KV cache.
    del out
    gc.collect()

    cuda_sync()

    after = torch.cuda.memory_allocated()

    observed_delta = max(
        0,
        after - before,
    )

    num_layers = int(
        getattr(
            config,
            "num_hidden_layers",
        )
    )

    num_attention_heads = int(
        getattr(
            config,
            "num_attention_heads",
        )
    )

    num_kv_heads = int(
        getattr(
            config,
            "num_key_value_heads",
            num_attention_heads,
        )
    )

    hidden_size = int(
        getattr(
            config,
            "hidden_size",
        )
    )

    head_dim = int(
        getattr(
            config,
            "head_dim",
            hidden_size // num_attention_heads,
        )
    )

    bytes_per_value = 2  # FP16

    theoretical_bytes = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * bytes_per_value
    )

    result = {
        "observed_mib": (
            observed_delta / 1024**2
        ),
        "theoretical_mib": (
            theoretical_bytes / 1024**2
        ),
        "layers": num_layers,
        "attention_heads": (
            num_attention_heads
        ),
        "kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }

    del past
    del input_ids

    gc.collect()
    torch.cuda.empty_cache()
    cuda_sync()

    return result


def cached_decode_benchmark(
    model,
    vocab_size,
    prompt_len,
    steps=16,
):
    prompt = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, prompt_len),
        dtype=torch.long,
        device=DEVICE,
    )

    next_token = torch.ones(
        (1, 1),
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.inference_mode():
        initial = model(
            input_ids=prompt,
            use_cache=True,
        )

    cuda_sync()

    past = initial.past_key_values

    del initial

    times = []

    with torch.inference_mode():
        for _ in range(steps):

            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )

            end.record()
            end.synchronize()

            times.append(
                start.elapsed_time(end)
            )

            past = out.past_key_values

            del out

    median_ms = statistics.median(
        times
    )

    mean_ms = statistics.mean(
        times
    )

    del past
    del prompt
    del next_token

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "times": times,
    }


def no_cache_decode_benchmark(
    model,
    vocab_size,
    prompt_len,
    steps=16,
):
    sequence = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, prompt_len),
        dtype=torch.long,
        device=DEVICE,
    )

    token = torch.ones(
        (1, 1),
        dtype=torch.long,
        device=DEVICE,
    )

    times = []

    with torch.inference_mode():

        for _ in range(steps):

            sequence = torch.cat(
                [sequence, token],
                dim=1,
            )

            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            out = model(
                input_ids=sequence,
                use_cache=False,
            )

            end.record()
            end.synchronize()

            times.append(
                start.elapsed_time(end)
            )

            del out

    median_ms = statistics.median(
        times
    )

    mean_ms = statistics.mean(
        times
    )

    del sequence
    del token

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "times": times,
    }


print(LINE)
print("AI INFRA PROJECT 1")
print("P1-M5 — TRANSFORMER / LLM INFERENCE")
print(LINE)

print()
print("ENVIRONMENT")
print("-" * 120)

print(
    "Python =",
    platform.python_version(),
)

print(
    "PyTorch =",
    torch.__version__,
)

print(
    "Transformers =",
    transformers.__version__,
)

print(
    "PyTorch CUDA =",
    torch.version.cuda,
)

print(
    "CUDA available =",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

print(
    "GPU =",
    torch.cuda.get_device_name(0),
)

props = torch.cuda.get_device_properties(0)

print(
    "VRAM GiB =",
    round(
        props.total_memory / 1024**3,
        3,
    ),
)

print(
    "Compute Capability =",
    f"{props.major}.{props.minor}",
)


print()
print(LINE)
print("1. LOAD TOKENIZER AND MODEL")
print(LINE)

print(
    "MODEL_ID =",
    MODEL_ID,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
)

load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

model.eval()
model.to(DEVICE)

cuda_sync()

load_seconds = (
    time.perf_counter()
    - load_start
)

config = model.config

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

parameter_storage_gib = (
    sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    / 1024**3
)

print(
    "MODEL_LOAD_SECONDS =",
    round(
        load_seconds,
        3,
    ),
)

print(
    "PARAMETER_COUNT =",
    parameter_count,
)

print(
    "PARAMETER_COUNT_BILLION =",
    round(
        parameter_count / 1e9,
        4,
    ),
)

print(
    "TRAINABLE_PARAMETER_COUNT =",
    trainable_count,
)

print(
    "MODEL_PARAMETER_STORAGE_GIB =",
    round(
        parameter_storage_gib,
        4,
    ),
)

print(
    "MODEL_DTYPE =",
    str(
        next(
            model.parameters()
        ).dtype
    ),
)

print(
    "MODEL_DEVICE =",
    str(
        next(
            model.parameters()
        ).device
    ),
)

print(
    "GPU_ALLOCATED_AFTER_LOAD_MIB =",
    round(
        allocated_mib(),
        3,
    ),
)

print(
    "GPU_RESERVED_AFTER_LOAD_MIB =",
    round(
        reserved_mib(),
        3,
    ),
)

print()
print("MODEL CONFIG")

for name in [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "max_position_embeddings",
]:
    print(
        f"{name} =",
        getattr(
            config,
            name,
            "N/A",
        ),
    )


print()
print(LINE)
print("2. REAL TEXT GENERATION SANITY TEST")
print(LINE)

prompt_text = (
    "In one short sentence, "
    "explain what a GPU is."
)

inputs = tokenizer(
    prompt_text,
    return_tensors="pt",
).to(DEVICE)

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=24,
        do_sample=False,
        use_cache=True,
    )

generated_text = tokenizer.decode(
    generated[0],
    skip_special_tokens=True,
)

print(
    "PROMPT =",
    prompt_text,
)

print(
    "GENERATED_TEXT =",
    generated_text.replace(
        "\n",
        " ",
    ),
)

print(
    "TEXT_GENERATION = SUCCESS"
)


print()
print(LINE)
print("3. PREFILL BENCHMARK")
print(LINE)

vocab_size = int(
    config.vocab_size
)

prefill_lengths = [
    128,
    512,
    1024,
    2048,
]

print(
    "seq_len"
    "\t"
    "median_ms"
    "\t"
    "tokens_per_second"
)

for seq_len in prefill_lengths:

    ms = median_prefill_ms(
        model,
        vocab_size,
        seq_len,
    )

    tokens_per_second = (
        seq_len
        / (
            ms / 1000.0
        )
    )

    print(
        f"{seq_len}"
        f"\t"
        f"{ms:.6f}"
        f"\t"
        f"{tokens_per_second:.2f}"
    )

    add(
        "prefill",
        "use_cache_true",
        seq_len,
        ms,
        "ms",
        (
            f"tokens_per_second="
            f"{tokens_per_second:.3f}"
        ),
    )


print()
print(LINE)
print("4. KV CACHE MEMORY")
print(LINE)

KV_SEQ_LEN = 2048

kv = kv_cache_measurement(
    model,
    config,
    vocab_size,
    KV_SEQ_LEN,
)

print(
    "KV_SEQUENCE_LENGTH =",
    KV_SEQ_LEN,
)

print(
    "NUM_LAYERS =",
    kv["layers"],
)

print(
    "NUM_ATTENTION_HEADS =",
    kv["attention_heads"],
)

print(
    "NUM_KV_HEADS =",
    kv["kv_heads"],
)

print(
    "HEAD_DIM =",
    kv["head_dim"],
)

print(
    "THEORETICAL_KV_CACHE_MIB =",
    round(
        kv["theoretical_mib"],
        3,
    ),
)

print(
    "OBSERVED_ALLOCATED_DELTA_MIB =",
    round(
        kv["observed_mib"],
        3,
    ),
)

add(
    "kv_cache",
    "theoretical",
    KV_SEQ_LEN,
    kv["theoretical_mib"],
    "MiB",
)

add(
    "kv_cache",
    "observed_allocated_delta",
    KV_SEQ_LEN,
    kv["observed_mib"],
    "MiB",
)


print()
print(LINE)
print("5. DECODE WITH KV CACHE")
print(LINE)

DECODE_PROMPT = 1024
DECODE_STEPS = 16

cached = cached_decode_benchmark(
    model,
    vocab_size,
    DECODE_PROMPT,
    DECODE_STEPS,
)

print(
    "PROMPT_LENGTH =",
    DECODE_PROMPT,
)

print(
    "DECODE_STEPS =",
    DECODE_STEPS,
)

print(
    "CACHED_DECODE_MEDIAN_MS_PER_TOKEN =",
    round(
        cached["median_ms"],
        6,
    ),
)

print(
    "CACHED_DECODE_MEAN_MS_PER_TOKEN =",
    round(
        cached["mean_ms"],
        6,
    ),
)

print(
    "CACHED_DECODE_TOKENS_PER_SECOND =",
    round(
        1000.0
        / cached["median_ms"],
        3,
    ),
)

print(
    "CACHED_DECODE_STEP_TIMES_MS =",
    ",".join(
        f"{x:.6f}"
        for x in cached["times"]
    ),
)

add(
    "decode",
    "with_kv_cache",
    DECODE_PROMPT,
    cached["median_ms"],
    "ms/token",
)


print()
print(LINE)
print("6. DECODE WITHOUT KV CACHE")
print(LINE)

nocache = no_cache_decode_benchmark(
    model,
    vocab_size,
    DECODE_PROMPT,
    DECODE_STEPS,
)

print(
    "NO_CACHE_DECODE_MEDIAN_MS_PER_TOKEN =",
    round(
        nocache["median_ms"],
        6,
    ),
)

print(
    "NO_CACHE_DECODE_MEAN_MS_PER_TOKEN =",
    round(
        nocache["mean_ms"],
        6,
    ),
)

print(
    "NO_CACHE_DECODE_STEP_TIMES_MS =",
    ",".join(
        f"{x:.6f}"
        for x in nocache["times"]
    ),
)

speedup = (
    nocache["median_ms"]
    / cached["median_ms"]
)

print(
    "KV_CACHE_DECODE_SPEEDUP =",
    round(
        speedup,
        3,
    ),
    "x",
)

add(
    "decode",
    "without_kv_cache",
    DECODE_PROMPT,
    nocache["median_ms"],
    "ms/token",
)

add(
    "decode",
    "kv_cache_speedup",
    DECODE_PROMPT,
    speedup,
    "x",
)


print()
print(LINE)
print("7. FINAL GPU MEMORY")
print(LINE)

cuda_sync()

print(
    "GPU_ALLOCATED_MIB =",
    round(
        allocated_mib(),
        3,
    ),
)

print(
    "GPU_RESERVED_MIB =",
    round(
        reserved_mib(),
        3,
    ),
)

print(
    "GPU_MAX_ALLOCATED_MIB =",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**2,
        3,
    ),
)


print()
print(LINE)
print("8. SAVE STRUCTURED RESULTS")
print(LINE)

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "experiment",
            "variant",
            "seq_len",
            "value",
            "unit",
            "notes",
        ],
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
print("P1-M5 KNOWLEDGE CHECKPOINT")
print(LINE)

print(
r"""
LLM INFERENCE PIPELINE

Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Transformer Layers
 ↓
Logits
 ↓
Next Token


PREFILL

Suppose the prompt contains:

    1024 tokens

During Prefill, the model processes the prompt tokens
and creates attention state for the sequence.

Characteristics:

    many prompt tokens available in parallel
    substantial matrix multiplication
    high GPU parallelism

Longer prompt:

    more work
    larger attention state
    larger KV Cache


DECODE

After Prefill, generation becomes iterative:

    token N
       ↓
    token N+1
       ↓
    token N+2
       ↓
    ...

Each step generates roughly one new token per sequence.

Decode therefore has much less parallel work
per individual request than Prefill.


KV CACHE

Without KV Cache:

    every new decoding step would repeatedly recompute
    K and V for previous tokens.

With KV Cache:

    previous K/V tensors remain available
    and only the new token's K/V must be computed.

Conceptually:

    Prefill
       ↓
    build K/V for prompt
       ↓
    Decode step 1
       ↓
    append new K/V
       ↓
    Decode step 2
       ↓
    append new K/V


KV CACHE MEMORY

For batch size 1, approximate KV storage is:

    2
    × number of layers
    × KV heads
    × head dimension
    × sequence length
    × bytes per value

The leading 2 means:

    K
    +
    V


GQA / MQA

A model may have:

    many attention query heads

but fewer:

    KV heads

This reduces KV Cache size.

Therefore:

    num_attention_heads
    and
    num_key_value_heads

are not necessarily equal.


WHY PREFILL AND DECODE BEHAVE DIFFERENTLY

From P1-M4:

    low arithmetic intensity
        → memory-side sensitive

    high arithmetic intensity
        → compute-side sensitive

Large Prefill GEMMs generally expose more parallel computation.

Single-request Decode repeatedly accesses model weights
and KV state while doing relatively little work for each step.

Therefore Decode is often substantially more
memory-bandwidth-sensitive.


WHY THIS LEADS TO LLM SERVING

One single Decode request may not use a large GPU efficiently.

The system therefore wants to combine work from:

    Request A
    Request B
    Request C
    Request D
    ...

This leads directly to:

    batching
    continuous batching
    request scheduling
    KV Cache management
    PagedAttention

which is exactly the problem solved by systems such as:

    vLLM
    SGLang
""".strip()
)


print()
print(LINE)
print("FINAL VERDICT")
print(LINE)

print(
    "MODEL_LOAD = SUCCESS"
)

print(
    "REAL_TEXT_GENERATION = SUCCESS"
)

print(
    "PREFILL_BENCHMARK = SUCCESS"
)

print(
    "KV_CACHE_MEMORY_ANALYSIS = SUCCESS"
)

print(
    "CACHED_DECODE_BENCHMARK = SUCCESS"
)

print(
    "NO_CACHE_DECODE_CONTROL = SUCCESS"
)

print(
    "STRUCTURED_CSV_RESULT = SUCCESS"
)

print(
    "VERDICT = READY_FOR_P1_M6_VLLM_SERVING_BASELINE"
)
PY

echo "PYTHON_BENCHMARK_CREATED = ${PY}"

# -------------------------------------------------------------------------------------------------
# 5. Run
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "5. RUN TRANSFORMER / LLM INFERENCE EXPERIMENT"
echo "${LINE}"

export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

export TOKENIZERS_PARALLELISM=false

P1_M5_MODEL="${MODEL_ID}" \
P1_M5_CSV="${CSV}" \
"${PYTHON}" "${PY}"

# -------------------------------------------------------------------------------------------------
# 6. Include structured output in TXT
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "6. CSV CONTENT"
echo "${LINE}"

cat "${CSV}"

# -------------------------------------------------------------------------------------------------
# 7. Environment snapshot
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "7. PROJECT 1 ENVIRONMENT SNAPSHOT"
echo "${LINE}"

echo "PROJECT1_ENV = ${ENV_DIR}"
echo "PYTHON = ${PYTHON}"
echo

"${PYTHON}" - <<'PY'
import torch
import transformers
import accelerate
import safetensors

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
print("safetensors =", safetensors.__version__)
PY

echo
echo "PIP_FREEZE = ${PROJECT}/docs/project1_environment_pip_freeze.txt"

# -------------------------------------------------------------------------------------------------
# 8. Artifact summary
# -------------------------------------------------------------------------------------------------

echo
echo "${LINE}"
echo "8. ARTIFACTS"
echo "${LINE}"

echo "BENCHMARK_SOURCE = ${PY}"
echo "CSV_RESULT = ${CSV}"
echo "ENVIRONMENT_FREEZE = ${PROJECT}/docs/project1_environment_pip_freeze.txt"
echo "HF_CACHE = ${HF_HOME}"
echo "TXT_RESULT = ${TXT}"

echo
echo "${LINE}"
echo "END OF P1-M5 RESULT"
echo "${LINE}"
