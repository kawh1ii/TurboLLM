import csv
import sys


path = sys.argv[1]

with open(
    path,
    "r",
    encoding="utf-8",
) as f:

    rows = list(
        csv.DictReader(f)
    )


def get(
    mode,
    experiment,
):
    for row in rows:
        if (
            row["mode"] == mode
            and
            row["experiment"]
            == experiment
        ):
            return row

    raise RuntimeError(
        f"Missing row "
        f"{mode}/{experiment}"
    )


def f(
    row,
    key,
):
    return float(
        row[key]
    )


baseline_sched = get(
    "baseline",
    "scheduler_pressure",
)

sched4 = get(
    "sched4",
    "scheduler_pressure",
)


baseline_kv = get(
    "baseline",
    "kv_pressure",
)

tinykv = get(
    "tinykv",
    "kv_pressure",
)


cache_on = get(
    "baseline",
    "prefix_cache",
)

cache_off = get(
    "nocache",
    "prefix_cache",
)


print("=" * 120)
print(
    "P1-M10 KV CACHE / SCHEDULING COMPARISON"
)
print("=" * 120)


# ==================================================================================================
# SCHEDULER
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT A — SCHEDULER ADMISSION"
)
print("=" * 120)

print(
    "BASELINE_MAX_NUM_SEQS =",
    baseline_sched[
        "max_num_seqs"
    ],
)

print(
    "CONSTRAINED_MAX_NUM_SEQS =",
    sched4[
        "max_num_seqs"
    ],
)

print(
    "BASELINE_PEAK_RUNNING =",
    f(
        baseline_sched,
        "peak_running_requests",
    ),
)

print(
    "SCHED4_PEAK_RUNNING =",
    f(
        sched4,
        "peak_running_requests",
    ),
)

print(
    "BASELINE_PEAK_WAITING =",
    f(
        baseline_sched,
        "peak_waiting_requests",
    ),
)

print(
    "SCHED4_PEAK_WAITING =",
    f(
        sched4,
        "peak_waiting_requests",
    ),
)

print(
    "BASELINE_TTFT_MS =",
    f(
        baseline_sched,
        "median_ttft_ms",
    ),
)

print(
    "SCHED4_TTFT_MS =",
    f(
        sched4,
        "median_ttft_ms",
    ),
)

print(
    "BASELINE_OUTPUT_TOKENS_S =",
    f(
        baseline_sched,
        "output_tokens_per_s",
    ),
)

print(
    "SCHED4_OUTPUT_TOKENS_S =",
    f(
        sched4,
        "output_tokens_per_s",
    ),
)


scheduler_effect = (
    f(
        sched4,
        "peak_waiting_requests",
    )
    >
    f(
        baseline_sched,
        "peak_waiting_requests",
    )
)


print(
    "SCHEDULER_WAITING_INCREASE_OBSERVED =",
    scheduler_effect,
)


# ==================================================================================================
# KV PRESSURE
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT B — KV CACHE PRESSURE"
)
print("=" * 120)

print(
    "BASELINE_KV_CACHE_TOKENS =",
    baseline_kv[
        "server_kv_cache_tokens"
    ],
)

print(
    "TINYKV_KV_CACHE_TOKENS =",
    tinykv[
        "server_kv_cache_tokens"
    ],
)

print(
    "BASELINE_MAX_CONCURRENCY_4096 =",
    baseline_kv[
        "server_max_concurrency_4096"
    ],
)

print(
    "TINYKV_MAX_CONCURRENCY_4096 =",
    tinykv[
        "server_max_concurrency_4096"
    ],
)

print(
    "BASELINE_PREEMPTIONS =",
    f(
        baseline_kv,
        "num_preemptions",
    ),
)

print(
    "TINYKV_PREEMPTIONS =",
    f(
        tinykv,
        "num_preemptions",
    ),
)

print(
    "BASELINE_PEAK_KV_USAGE_PCT =",
    f(
        baseline_kv,
        "peak_kv_cache_usage_pct",
    ),
)

print(
    "TINYKV_PEAK_KV_USAGE_PCT =",
    f(
        tinykv,
        "peak_kv_cache_usage_pct",
    ),
)

print(
    "BASELINE_TTFT_MS =",
    f(
        baseline_kv,
        "median_ttft_ms",
    ),
)

print(
    "TINYKV_TTFT_MS =",
    f(
        tinykv,
        "median_ttft_ms",
    ),
)

print(
    "BASELINE_OUTPUT_TOKENS_S =",
    f(
        baseline_kv,
        "output_tokens_per_s",
    ),
)

print(
    "TINYKV_OUTPUT_TOKENS_S =",
    f(
        tinykv,
        "output_tokens_per_s",
    ),
)


kv_capacity_reduced = (
    f(
        tinykv,
        "server_kv_cache_tokens",
    )
    <
    f(
        baseline_kv,
        "server_kv_cache_tokens",
    )
)


kv_pressure_observed = (
    f(
        tinykv,
        "num_preemptions",
    )
    > 0
    or
    f(
        tinykv,
        "peak_waiting_requests",
    )
    >
    f(
        baseline_kv,
        "peak_waiting_requests",
    )
)


print(
    "KV_CAPACITY_REDUCTION_OBSERVED =",
    kv_capacity_reduced,
)

print(
    "KV_PRESSURE_OR_PREEMPTION_OBSERVED =",
    kv_pressure_observed,
)


# ==================================================================================================
# PREFIX CACHE
# ==================================================================================================

print()
print("=" * 120)
print(
    "EXPERIMENT C — PREFIX CACHE"
)
print("=" * 120)

print(
    "CACHE_ON_QUERIES =",
    f(
        cache_on,
        "prefix_cache_queries_tokens",
    ),
)

print(
    "CACHE_ON_HITS =",
    f(
        cache_on,
        "prefix_cache_hits_tokens",
    ),
)

print(
    "CACHE_ON_HIT_RATE =",
    f(
        cache_on,
        "prefix_cache_hit_rate",
    ),
)

print(
    "CACHE_OFF_QUERIES =",
    f(
        cache_off,
        "prefix_cache_queries_tokens",
    ),
)

print(
    "CACHE_OFF_HITS =",
    f(
        cache_off,
        "prefix_cache_hits_tokens",
    ),
)

print(
    "CACHE_ON_PREFILL_KV_COMPUTED =",
    f(
        cache_on,
        "prefill_kv_computed_tokens",
    ),
)

print(
    "CACHE_OFF_PREFILL_KV_COMPUTED =",
    f(
        cache_off,
        "prefill_kv_computed_tokens",
    ),
)

print(
    "CACHE_ON_TTFT_MS =",
    f(
        cache_on,
        "median_ttft_ms",
    ),
)

print(
    "CACHE_OFF_TTFT_MS =",
    f(
        cache_off,
        "median_ttft_ms",
    ),
)

print(
    "CACHE_ON_PROMPT_TOKENS_S =",
    f(
        cache_on,
        "prompt_tokens_per_s",
    ),
)

print(
    "CACHE_OFF_PROMPT_TOKENS_S =",
    f(
        cache_off,
        "prompt_tokens_per_s",
    ),
)


prefix_hits_observed = (
    f(
        cache_on,
        "prefix_cache_hits_tokens",
    )
    > 0
)


prefix_compute_reduced = (
    f(
        cache_on,
        "prefill_kv_computed_tokens",
    )
    <
    f(
        cache_off,
        "prefill_kv_computed_tokens",
    )
)


print(
    "PREFIX_CACHE_HITS_OBSERVED =",
    prefix_hits_observed,
)

print(
    "PREFIX_CACHE_REDUCED_COMPUTE =",
    prefix_compute_reduced,
)


# ==================================================================================================
# KNOWLEDGE
# ==================================================================================================

print()
print("=" * 120)
print(
    "P1-M10 KNOWLEDGE CHECKPOINT"
)
print("=" * 120)

print(
"""
KV CACHE CAPACITY

The KV cache stores attention K/V state for active
and reusable token prefixes.

Reducing KV capacity does not change model weights,
but it changes how many sequence tokens can coexist.


SCHEDULER ADMISSION

max_num_seqs limits the number of active sequences
the scheduler may process in one iteration.

If incoming concurrency exceeds this limit:

    running requests are capped
    waiting requests increase

This creates an explicit throughput / queueing trade-off.


KV PRESSURE AND PREEMPTION

When active sequence state cannot fit comfortably in
the KV cache, the serving engine must delay work,
evict/recompute state, or preempt sequences.

Therefore KV capacity is not merely a memory statistic.

It directly participates in scheduling.


PREFIX CACHE

Prefix caching reuses already-computed KV blocks
for matching prompt prefixes.

A correct experiment must:

    create a genuinely shared prefix
    warm that prefix first
    measure cache hits
    measure newly-computed KV tokens

rather than infer caching from latency alone.


SYSTEMS VIEW

Scheduler + KV cache form one coupled resource manager:

      request admission
            ↓
      active sequences
            ↓
      KV block allocation
            ↓
      batching / execution
            ↓
      latency + throughput
""".strip()
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)


print(
    "SCHEDULER_EFFECT =",
    scheduler_effect,
)

print(
    "KV_CAPACITY_REDUCTION =",
    kv_capacity_reduced,
)

print(
    "KV_PRESSURE_EFFECT =",
    kv_pressure_observed,
)

print(
    "PREFIX_HITS =",
    prefix_hits_observed,
)

print(
    "PREFIX_COMPUTE_REDUCTION =",
    prefix_compute_reduced,
)


if (
    scheduler_effect
    and
    kv_capacity_reduced
    and
    kv_pressure_observed
    and
    prefix_hits_observed
    and
    prefix_compute_reduced
):

    print(
        "VERDICT = "
        "READY_FOR_P1_M11_KERNEL_LEVEL_PROFILING"
    )

else:

    print(
        "VERDICT = "
        "P1_M10_BEHAVIOR_REQUIRES_REVIEW"
    )
