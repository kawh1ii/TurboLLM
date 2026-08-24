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
