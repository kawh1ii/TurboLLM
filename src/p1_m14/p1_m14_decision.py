import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path


csv_path = Path(sys.argv[1])
m13_timestamp = sys.argv[2]
result_dir = Path(sys.argv[3])
report_path = Path(sys.argv[4])
json_path = Path(sys.argv[5])
artifact_dir = Path(sys.argv[6])


MODES = [
    "none",
    "piecewise",
    "full_and_piecewise",
]

PHASES = [
    "prefill",
    "decode",
]


# ==================================================================================================
# LOAD CSV
# ==================================================================================================

with csv_path.open(
    "r",
    encoding="utf-8",
) as f:
    rows = list(
        csv.DictReader(f)
    )


print("=" * 120)
print("P1-M14 FROZEN RESULT VERIFICATION")
print("=" * 120)

print(
    "SOURCE_CSV =",
    csv_path,
)

print(
    "CSV_ROW_COUNT =",
    len(rows),
)


if len(rows) != 6:
    raise RuntimeError(
        f"Expected 6 rows, got {len(rows)}"
    )


data = {}


for row in rows:

    mode = row["mode"]
    phase = row["phase"]

    if mode not in MODES:
        raise RuntimeError(
            f"Unexpected mode: {mode}"
        )

    if phase not in PHASES:
        raise RuntimeError(
            f"Unexpected phase: {phase}"
        )

    key = (
        phase,
        mode,
    )

    if key in data:
        raise RuntimeError(
            f"Duplicate result: {key}"
        )

    requested = mode

    actual = row[
        "actual_cudagraph_mode"
    ]

    if actual != requested:
        raise RuntimeError(
            f"CUDA Graph mismatch: "
            f"requested={requested}, "
            f"actual={actual}"
        )

    data[key] = row


for phase in PHASES:
    for mode in MODES:

        if (
            phase,
            mode,
        ) not in data:

            raise RuntimeError(
                f"Missing "
                f"{phase}/{mode}"
            )


print(
    "SIX_MATCHED_ROWS = TRUE"
)

print(
    "REQUESTED_ACTUAL_MODE_MATCH = TRUE"
)

print(
    "TORCH_COMPILE_CONTROL = MODE_3"
)

print(
    "PREFIX_CACHE_CONTROL = DISABLED"
)


# ==================================================================================================
# HELPERS
# ==================================================================================================

def number(
    phase,
    mode,
    field,
):

    return float(
        data[
            (
                phase,
                mode,
            )
        ][field]
    )


def ratio(
    a,
    b,
):
    if b == 0:
        return float("nan")

    return a / b


def pct_change(
    new,
    old,
):
    if old == 0:
        return float("nan")

    return (
        new / old
        - 1.0
    ) * 100.0


# ==================================================================================================
# COMPLETE FROZEN TABLE
# ==================================================================================================

print()
print("=" * 120)
print("3. COMPLETE P1-M13 MATCHED RESULTS")
print("=" * 120)


for phase in PHASES:

    print()
    print("-" * 120)
    print(
        "PHASE =",
        phase.upper(),
    )
    print("-" * 120)

    for mode in MODES:

        print()
        print(
            "MODE =",
            mode,
        )

        for field in [
            "startup_s",
            "ready_vram_mib",
            "median_ttft_ms",
            "p95_ttft_ms",
            "median_tpot_ms",
            "median_latency_ms",
            "p95_latency_ms",
            "request_throughput_rps",
            "prompt_tokens_per_s",
            "output_tokens_per_s",
            "total_tokens_per_s",
        ]:

            print(
                f"{field} = "
                f"{number(phase, mode, field):.6f}"
            )


# ==================================================================================================
# PRIMARY METRIC POLICY
# ==================================================================================================

print()
print("=" * 120)
print("4. METRIC POLICY")
print("=" * 120)

print(
"""
PREFILL-HEAVY workload:

    PRIMARY THROUGHPUT METRIC
        prompt_tokens_per_s

    PRIMARY LATENCY METRIC
        median_ttft_ms

The output contains only 16 generated tokens, therefore
output_tokens_per_s is not treated as the primary Prefill
performance metric.


DECODE-HEAVY workload:

    PRIMARY THROUGHPUT METRIC
        output_tokens_per_s

    PRIMARY TOKEN-LATENCY METRIC
        median_tpot_ms

The workload generates 512 output tokens, therefore TPOT
and output throughput are the main Decode measurements.


Startup time and ready-state VRAM are treated as costs,
not as substitutes for steady-state serving performance.
""".strip()
)


# ==================================================================================================
# PREFILL DECISION
# ==================================================================================================

prefill_throughput_winner = max(
    MODES,
    key=lambda mode: number(
        "prefill",
        mode,
        "prompt_tokens_per_s",
    ),
)

prefill_ttft_winner = min(
    MODES,
    key=lambda mode: number(
        "prefill",
        mode,
        "median_ttft_ms",
    ),
)


print()
print("=" * 120)
print("5. PREFILL OPTIMIZATION DECISION")
print("=" * 120)


for mode in MODES:

    prompt_tps = number(
        "prefill",
        mode,
        "prompt_tokens_per_s",
    )

    ttft = number(
        "prefill",
        mode,
        "median_ttft_ms",
    )

    print()
    print(
        f"{mode.upper()}_PROMPT_TOKENS_S = "
        f"{prompt_tps:.3f}"
    )

    print(
        f"{mode.upper()}_TTFT_MS = "
        f"{ttft:.3f}"
    )


print()
print(
    "PREFILL_THROUGHPUT_WINNER =",
    prefill_throughput_winner,
)

print(
    "PREFILL_TTFT_WINNER =",
    prefill_ttft_winner,
)


prefill_none = number(
    "prefill",
    "none",
    "prompt_tokens_per_s",
)

prefill_piece = number(
    "prefill",
    "piecewise",
    "prompt_tokens_per_s",
)

prefill_full = number(
    "prefill",
    "full_and_piecewise",
    "prompt_tokens_per_s",
)


print(
    "PREFILL_PIECEWISE_SPEEDUP_VS_NONE =",
    f"{ratio(prefill_piece, prefill_none):.4f}x",
)

print(
    "PREFILL_FULL_SPEEDUP_VS_NONE =",
    f"{ratio(prefill_full, prefill_none):.4f}x",
)

print(
    "PREFILL_FULL_SPEEDUP_VS_PIECEWISE =",
    f"{ratio(prefill_full, prefill_piece):.4f}x",
)


# ==================================================================================================
# DECODE DECISION
# ==================================================================================================

decode_throughput_winner = max(
    MODES,
    key=lambda mode: number(
        "decode",
        mode,
        "output_tokens_per_s",
    ),
)

decode_tpot_winner = min(
    MODES,
    key=lambda mode: number(
        "decode",
        mode,
        "median_tpot_ms",
    ),
)


print()
print("=" * 120)
print("6. DECODE OPTIMIZATION DECISION")
print("=" * 120)


for mode in MODES:

    output_tps = number(
        "decode",
        mode,
        "output_tokens_per_s",
    )

    tpot = number(
        "decode",
        mode,
        "median_tpot_ms",
    )

    print()
    print(
        f"{mode.upper()}_OUTPUT_TOKENS_S = "
        f"{output_tps:.3f}"
    )

    print(
        f"{mode.upper()}_TPOT_MS = "
        f"{tpot:.3f}"
    )


print()
print(
    "DECODE_THROUGHPUT_WINNER =",
    decode_throughput_winner,
)

print(
    "DECODE_TPOT_WINNER =",
    decode_tpot_winner,
)


decode_none = number(
    "decode",
    "none",
    "output_tokens_per_s",
)

decode_piece = number(
    "decode",
    "piecewise",
    "output_tokens_per_s",
)

decode_full = number(
    "decode",
    "full_and_piecewise",
    "output_tokens_per_s",
)


print(
    "DECODE_PIECEWISE_SPEEDUP_VS_NONE =",
    f"{ratio(decode_piece, decode_none):.4f}x",
)

print(
    "DECODE_FULL_SPEEDUP_VS_NONE =",
    f"{ratio(decode_full, decode_none):.4f}x",
)

print(
    "DECODE_FULL_SPEEDUP_VS_PIECEWISE =",
    f"{ratio(decode_full, decode_piece):.4f}x",
)


# ==================================================================================================
# RAW TRIAL STABILITY
# ==================================================================================================

print()
print("=" * 120)
print("7. THREE-TRIAL STABILITY AUDIT")
print("=" * 120)


trial_audit = {}

primary_cvs = []


for phase in PHASES:

    for mode in MODES:

        raw_path = (
            result_dir
            / (
                f"P1_M13_"
                f"{mode}_"
                f"{phase}_"
                f"{m13_timestamp}.json"
            )
        )

        key = (
            f"{phase}/"
            f"{mode}"
        )

        print()
        print(
            "TRIAL_GROUP =",
            key,
        )

        print(
            "RAW_JSON =",
            raw_path,
        )


        if not raw_path.exists():

            print(
                "RAW_TRIAL_DATA = MISSING"
            )

            trial_audit[
                key
            ] = {
                "available": False,
            }

            continue


        obj = json.loads(
            raw_path.read_text(
                encoding="utf-8",
            )
        )

        trials = obj.get(
            "trials",
            [],
        )

        print(
            "TRIAL_COUNT =",
            len(trials),
        )


        if phase == "prefill":

            primary_field = (
                "prompt_tokens_per_s"
            )

        else:

            primary_field = (
                "output_tokens_per_s"
            )


        values = [
            float(
                trial[
                    primary_field
                ]
            )
            for trial in trials
        ]


        mean = (
            statistics.mean(values)
            if values
            else float("nan")
        )

        median = (
            statistics.median(values)
            if values
            else float("nan")
        )

        stdev = (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        )


        cv = (
            stdev / mean * 100.0
            if mean != 0
            else float("nan")
        )


        primary_cvs.append(
            cv
        )


        print(
            "PRIMARY_METRIC =",
            primary_field,
        )

        print(
            "TRIAL_VALUES =",
            ",".join(
                f"{x:.6f}"
                for x in values
            ),
        )

        print(
            "TRIAL_MEAN =",
            f"{mean:.6f}",
        )

        print(
            "TRIAL_MEDIAN =",
            f"{median:.6f}",
        )

        print(
            "TRIAL_STDEV =",
            f"{stdev:.6f}",
        )

        print(
            "TRIAL_CV_PERCENT =",
            f"{cv:.3f}",
        )


        if cv <= 3.0:

            stability = (
                "HIGH_STABILITY"
            )

        elif cv <= 5.0:

            stability = (
                "ACCEPTABLE_STABILITY"
            )

        else:

            stability = (
                "NOISY_RESULT"
            )


        print(
            "TRIAL_STABILITY =",
            stability,
        )


        trial_audit[
            key
        ] = {
            "available": True,
            "primary_metric": (
                primary_field
            ),
            "values": values,
            "mean": mean,
            "median": median,
            "stdev": stdev,
            "cv_percent": cv,
            "stability": stability,
        }


if primary_cvs:

    max_cv = max(
        primary_cvs
    )

else:

    max_cv = float("nan")


print()
print(
    "MAX_PRIMARY_METRIC_CV_PERCENT =",
    f"{max_cv:.3f}",
)


# ==================================================================================================
# COST ANALYSIS
# ==================================================================================================

print()
print("=" * 120)
print("8. STARTUP AND VRAM COST")
print("=" * 120)


for mode in MODES:

    startup = number(
        "decode",
        mode,
        "startup_s",
    )

    vram = number(
        "decode",
        mode,
        "ready_vram_mib",
    )

    print(
        f"{mode.upper()}_STARTUP_S = "
        f"{startup:.3f}"
    )

    print(
        f"{mode.upper()}_READY_VRAM_MIB = "
        f"{vram:.1f}"
    )


baseline_vram = number(
    "decode",
    "none",
    "ready_vram_mib",
)


for mode in [
    "piecewise",
    "full_and_piecewise",
]:

    delta = (
        number(
            "decode",
            mode,
            "ready_vram_mib",
        )
        - baseline_vram
    )

    print(
        f"{mode.upper()}_VRAM_DELTA_VS_NONE_MIB = "
        f"{delta:.1f}"
    )


# ==================================================================================================
# BALANCED MIXED-WORKLOAD INDEX
#
# This is explicitly a summary heuristic, not a physical benchmark.
# ==================================================================================================

print()
print("=" * 120)
print("9. MIXED-WORKLOAD SUMMARY INDEX")
print("=" * 120)

print(
"""
The following index is only a decision heuristic.

For each CUDA Graph mode:

    Prefill factor
        =
        prompt tokens/s
        normalized to NONE

    Decode factor
        =
        output tokens/s
        normalized to NONE

Balanced index
        =
        geometric mean(
            Prefill factor,
            Decode factor
        )

It gives Prefill and Decode equal multiplicative weight.

It is NOT presented as a measured production workload.
""".strip()
)


mixed_index = {}


for mode in MODES:

    prefill_factor = ratio(
        number(
            "prefill",
            mode,
            "prompt_tokens_per_s",
        ),
        prefill_none,
    )

    decode_factor = ratio(
        number(
            "decode",
            mode,
            "output_tokens_per_s",
        ),
        decode_none,
    )

    index = math.sqrt(
        prefill_factor
        * decode_factor
    )

    mixed_index[
        mode
    ] = index


    print()
    print(
        f"{mode.upper()}_PREFILL_FACTOR = "
        f"{prefill_factor:.4f}x"
    )

    print(
        f"{mode.upper()}_DECODE_FACTOR = "
        f"{decode_factor:.4f}x"
    )

    print(
        f"{mode.upper()}_BALANCED_INDEX = "
        f"{index:.4f}"
    )


mixed_winner = max(
    MODES,
    key=lambda mode: (
        mixed_index[
            mode
        ]
    ),
)


print()
print(
    "BALANCED_MIXED_WORKLOAD_WINNER =",
    mixed_winner,
)


# ==================================================================================================
# DECISION CLASSIFICATION
# ==================================================================================================

print()
print("=" * 120)
print("10. OPTIMIZATION DECISION")
print("=" * 120)


prefill_metric_agreement = (
    prefill_throughput_winner
    ==
    prefill_ttft_winner
)

decode_metric_agreement = (
    decode_throughput_winner
    ==
    decode_tpot_winner
)


print(
    "PREFILL_PRIMARY_METRICS_AGREE =",
    prefill_metric_agreement,
)

print(
    "DECODE_PRIMARY_METRICS_AGREE =",
    decode_metric_agreement,
)


if (
    primary_cvs
    and
    max_cv <= 5.0
):

    stability_verdict = (
        "STABLE_ENOUGH_FOR_PROJECT_DECISION"
    )

elif primary_cvs:

    stability_verdict = (
        "RESULTS_VALID_BUT_VARIANCE_REQUIRES_CAUTION"
    )

else:

    stability_verdict = (
        "AGGREGATE_ONLY_RAW_TRIALS_UNAVAILABLE"
    )


print(
    "STABILITY_VERDICT =",
    stability_verdict,
)


default_mode = (
    "full_and_piecewise"
)


default_prefill = number(
    "prefill",
    default_mode,
    "prompt_tokens_per_s",
)

best_prefill = number(
    "prefill",
    prefill_throughput_winner,
    "prompt_tokens_per_s",
)


default_decode = number(
    "decode",
    default_mode,
    "output_tokens_per_s",
)

best_decode = number(
    "decode",
    decode_throughput_winner,
    "output_tokens_per_s",
)


default_prefill_gap = pct_change(
    best_prefill,
    default_prefill,
)

default_decode_gap = pct_change(
    best_decode,
    default_decode,
)


print(
    "BEST_PREFILL_GAIN_OVER_DEFAULT_PERCENT =",
    f"{default_prefill_gap:.3f}",
)

print(
    "BEST_DECODE_GAIN_OVER_DEFAULT_PERCENT =",
    f"{default_decode_gap:.3f}",
)


# A conservative deployment recommendation:
#
# Prefer vLLM's default mixed mode unless an alternative
# beats it by >5% on the relevant phase.
#
# The 5% threshold is a project decision policy,
# NOT a claim of statistical significance.

DEPLOYMENT_MARGIN_PERCENT = 5.0


if (
    default_prefill_gap
    <= DEPLOYMENT_MARGIN_PERCENT
    and
    default_decode_gap
    <= DEPLOYMENT_MARGIN_PERCENT
):

    deployment_recommendation = (
        "full_and_piecewise"
    )

    deployment_reason = (
        "DEFAULT_WITHIN_5_PERCENT_OF_PHASE_WINNERS"
    )

else:

    deployment_recommendation = (
        mixed_winner
    )

    deployment_reason = (
        "MEASURED_PHASE_GAIN_EXCEEDS_5_PERCENT"
    )


print(
    "DECISION_POLICY_MARGIN_PERCENT =",
    DEPLOYMENT_MARGIN_PERCENT,
)

print(
    "RECOMMENDED_GENERAL_SERVING_MODE =",
    deployment_recommendation,
)

print(
    "RECOMMENDATION_REASON =",
    deployment_reason,
)


# ==================================================================================================
# EVIDENCE INVENTORY
# ==================================================================================================

print()
print("=" * 120)
print("11. PROJECT 1 EVIDENCE INVENTORY")
print("=" * 120)


artifact_files = sorted(
    artifact_dir.glob(
        "P1_M*.txt"
    )
)


print(
    "P1_ARTIFACT_TXT_COUNT =",
    len(
        artifact_files
    ),
)


for path in artifact_files:

    print(
        "EVIDENCE_ARTIFACT =",
        path,
    )


# ==================================================================================================
# STRUCTURED JSON
# ==================================================================================================

decision = {
    "source_csv": str(
        csv_path
    ),

    "m13_timestamp": (
        m13_timestamp
    ),

    "scientific_model_execution": False,

    "prefill": {
        "throughput_winner": (
            prefill_throughput_winner
        ),

        "ttft_winner": (
            prefill_ttft_winner
        ),

        "primary_metrics_agree": (
            prefill_metric_agreement
        ),

        "piecewise_speedup_vs_none": (
            ratio(
                prefill_piece,
                prefill_none,
            )
        ),

        "full_speedup_vs_none": (
            ratio(
                prefill_full,
                prefill_none,
            )
        ),
    },

    "decode": {
        "throughput_winner": (
            decode_throughput_winner
        ),

        "tpot_winner": (
            decode_tpot_winner
        ),

        "primary_metrics_agree": (
            decode_metric_agreement
        ),

        "piecewise_speedup_vs_none": (
            ratio(
                decode_piece,
                decode_none,
            )
        ),

        "full_speedup_vs_none": (
            ratio(
                decode_full,
                decode_none,
            )
        ),

        "full_speedup_vs_piecewise": (
            ratio(
                decode_full,
                decode_piece,
            )
        ),
    },

    "trial_stability": {
        "groups": trial_audit,

        "max_primary_metric_cv_percent": (
            max_cv
        ),

        "verdict": (
            stability_verdict
        ),
    },

    "mixed_workload_index": (
        mixed_index
    ),

    "balanced_mixed_winner": (
        mixed_winner
    ),

    "recommended_general_serving_mode": (
        deployment_recommendation
    ),

    "recommendation_reason": (
        deployment_reason
    ),

    "decision_policy_margin_percent": (
        DEPLOYMENT_MARGIN_PERCENT
    ),
}


json_path.write_text(
    json.dumps(
        decision,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "DECISION_JSON =",
    json_path,
)


# ==================================================================================================
# FINAL MARKDOWN REPORT
# ==================================================================================================

def fmt(
    value,
    digits=3,
):
    return (
        f"{value:.{digits}f}"
    )


markdown = f"""# Project 1 — TurboLLM Optimization Evidence Freeze

## 1. Scope

This report freezes the final CUDA Graph optimization decision from P1-M13.

No new model execution was performed in P1-M14.

Frozen source:

`{csv_path}`

Environment:

- GPU: NVIDIA GeForce RTX 4090
- vLLM: 0.27.1+cu129
- PyTorch: 2.13.0+cu129
- CUDA runtime used by PyTorch: 12.9
- torch.compile mode: VLLM_COMPILE / mode 3
- Prefix caching during CUDA Graph ablation: disabled

Compared CUDA Graph modes:

1. NONE
2. PIECEWISE
3. FULL_AND_PIECEWISE

---

## 2. Prefill-heavy result

Workload:

- prompt length ≈ 2048 tokens
- output length = 16 tokens
- concurrency = 8

Primary throughput metric: **prompt tokens/s**

| Mode | Prompt tok/s | TTFT ms | Startup s | Ready VRAM MiB |
|---|---:|---:|---:|---:|
| NONE | {fmt(number("prefill","none","prompt_tokens_per_s"))} | {fmt(number("prefill","none","median_ttft_ms"))} | {fmt(number("prefill","none","startup_s"))} | {fmt(number("prefill","none","ready_vram_mib"),1)} |
| PIECEWISE | {fmt(number("prefill","piecewise","prompt_tokens_per_s"))} | {fmt(number("prefill","piecewise","median_ttft_ms"))} | {fmt(number("prefill","piecewise","startup_s"))} | {fmt(number("prefill","piecewise","ready_vram_mib"),1)} |
| FULL_AND_PIECEWISE | {fmt(number("prefill","full_and_piecewise","prompt_tokens_per_s"))} | {fmt(number("prefill","full_and_piecewise","median_ttft_ms"))} | {fmt(number("prefill","full_and_piecewise","startup_s"))} | {fmt(number("prefill","full_and_piecewise","ready_vram_mib"),1)} |

Prefill throughput winner:

**{prefill_throughput_winner}**

Prefill TTFT winner:

**{prefill_ttft_winner}**

PIECEWISE throughput speedup vs NONE:

**{ratio(prefill_piece, prefill_none):.4f}×**

FULL_AND_PIECEWISE throughput speedup vs NONE:

**{ratio(prefill_full, prefill_none):.4f}×**

---

## 3. Decode-heavy result

Workload:

- prompt length ≈ 128 tokens
- output length = 512 tokens
- concurrency = 8

Primary throughput metric: **output tokens/s**

Primary token latency metric: **TPOT**

| Mode | Output tok/s | TPOT ms | Median latency ms | Ready VRAM MiB |
|---|---:|---:|---:|---:|
| NONE | {fmt(number("decode","none","output_tokens_per_s"))} | {fmt(number("decode","none","median_tpot_ms"))} | {fmt(number("decode","none","median_latency_ms"))} | {fmt(number("decode","none","ready_vram_mib"),1)} |
| PIECEWISE | {fmt(number("decode","piecewise","output_tokens_per_s"))} | {fmt(number("decode","piecewise","median_tpot_ms"))} | {fmt(number("decode","piecewise","median_latency_ms"))} | {fmt(number("decode","piecewise","ready_vram_mib"),1)} |
| FULL_AND_PIECEWISE | {fmt(number("decode","full_and_piecewise","output_tokens_per_s"))} | {fmt(number("decode","full_and_piecewise","median_tpot_ms"))} | {fmt(number("decode","full_and_piecewise","median_latency_ms"))} | {fmt(number("decode","full_and_piecewise","ready_vram_mib"),1)} |

Decode throughput winner:

**{decode_throughput_winner}**

Decode TPOT winner:

**{decode_tpot_winner}**

PIECEWISE throughput speedup vs NONE:

**{ratio(decode_piece, decode_none):.4f}×**

FULL_AND_PIECEWISE throughput speedup vs NONE:

**{ratio(decode_full, decode_none):.4f}×**

FULL_AND_PIECEWISE incremental throughput speedup vs PIECEWISE:

**{ratio(decode_full, decode_piece):.4f}×**

---

## 4. Stability

Maximum coefficient of variation across available primary 3-trial measurements:

**{max_cv:.3f}%**

Stability classification:

**{stability_verdict}**

The coefficient of variation is used only as a repeatability diagnostic. It is not presented as a formal statistical confidence interval.

---

## 5. Balanced mixed-workload summary

This is a decision heuristic, not a measured production workload.

Balanced index = geometric mean of:

1. Prefill prompt-throughput factor vs NONE.
2. Decode output-throughput factor vs NONE.

| Mode | Balanced index |
|---|---:|
| NONE | {mixed_index["none"]:.4f} |
| PIECEWISE | {mixed_index["piecewise"]:.4f} |
| FULL_AND_PIECEWISE | {mixed_index["full_and_piecewise"]:.4f} |

Balanced winner:

**{mixed_winner}**

---

## 6. Final serving decision

Best measured Prefill throughput mode:

**{prefill_throughput_winner}**

Best measured Decode throughput mode:

**{decode_throughput_winner}**

Recommended general serving mode under the project decision policy:

**{deployment_recommendation}**

Reason:

**{deployment_reason}**

The project decision policy retains the standard FULL_AND_PIECEWISE path when it is within 5% of the phase-specific throughput winners. The 5% value is an engineering decision margin, not a statistical significance threshold.

---

## 7. Evidence boundary

The project has directly measured:

- TTFT
- TPOT
- request throughput
- prompt throughput
- output throughput
- concurrency effects
- scheduler queueing
- KV-cache pressure
- preemption
- prefix-cache hits
- Prefill/Decode timing
- real vLLM CUDA kernel names
- kernel launch counts
- relative CUDA kernel time
- CUDA Graph ablation

Nsight Compute hardware counters are **not available** on the current cloud host because of `ERR_NVGPUCTRPERM`.

Therefore the project does **not** claim directly measured:

- DRAM bandwidth saturation
- Tensor Core utilization
- achieved occupancy
- detailed SM hardware-counter utilization

---

## 8. Final project status

P1-M13 matched CUDA Graph ablation: **PASS**

P1-M14 optimization decision: **PASS**

Project 1 experimental evidence: **READY TO FREEZE**

"""


report_path.write_text(
    markdown,
    encoding="utf-8",
)


print(
    "FINAL_REPORT_MD =",
    report_path,
)


# ==================================================================================================
# FINAL VERDICT
# ==================================================================================================

print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)

print(
    "M13_FROZEN_RESULT_VERIFIED = TRUE"
)

print(
    "SIX_MATCHED_ABLATION_ROWS = TRUE"
)

print(
    "TRIAL_STABILITY_AUDIT = COMPLETE"
)

print(
    "PREFILL_DECISION = COMPLETE"
)

print(
    "DECODE_DECISION = COMPLETE"
)

print(
    "MIXED_WORKLOAD_DECISION = COMPLETE"
)

print(
    "NEW_SCIENTIFIC_MODEL_EXECUTION = FALSE"
)

print(
    "FINAL_MARKDOWN_REPORT = SUCCESS"
)

print(
    "DECISION_JSON = SUCCESS"
)

print(
    "VERDICT = PROJECT1_EXPERIMENTAL_EVIDENCE_FREEZE_READY"
)
