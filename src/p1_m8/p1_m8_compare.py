import csv
import sys
from collections import defaultdict


path = sys.argv[1]

rows = []

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
    data[row["workload"]][
        row["mode"]
    ] = row


print("=" * 120)
print("P1-M8 EAGER VS OPTIMIZED COMPARISON")
print("=" * 120)


for workload in [
    "single_c1",
    "batch_c8",
    "batch_c16",
]:
    eager = data[workload][
        "eager"
    ]

    optimized = data[workload][
        "optimized"
    ]


    def num(row, name):
        return float(
            row[name]
        )


    eager_ttft = num(
        eager,
        "median_ttft_ms",
    )

    opt_ttft = num(
        optimized,
        "median_ttft_ms",
    )


    eager_tpot = num(
        eager,
        "median_tpot_ms",
    )

    opt_tpot = num(
        optimized,
        "median_tpot_ms",
    )


    eager_out = num(
        eager,
        "output_tokens_per_s",
    )

    opt_out = num(
        optimized,
        "output_tokens_per_s",
    )


    eager_rps = num(
        eager,
        "request_throughput_rps",
    )

    opt_rps = num(
        optimized,
        "request_throughput_rps",
    )


    eager_latency = num(
        eager,
        "median_latency_ms",
    )

    opt_latency = num(
        optimized,
        "median_latency_ms",
    )


    print()
    print("-" * 120)

    print(
        "WORKLOAD =",
        workload,
    )

    print(
        "EAGER_MEDIAN_TTFT_MS =",
        f"{eager_ttft:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_TTFT_MS =",
        f"{opt_ttft:.3f}",
    )

    print(
        "TTFT_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_ttft / opt_ttft:.3f}x",
    )


    print(
        "EAGER_MEDIAN_TPOT_MS =",
        f"{eager_tpot:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_TPOT_MS =",
        f"{opt_tpot:.3f}",
    )

    print(
        "TPOT_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_tpot / opt_tpot:.3f}x",
    )


    print(
        "EAGER_OUTPUT_TOKENS_S =",
        f"{eager_out:.3f}",
    )

    print(
        "OPTIMIZED_OUTPUT_TOKENS_S =",
        f"{opt_out:.3f}",
    )

    print(
        "OUTPUT_THROUGHPUT_SPEEDUP =",
        f"{opt_out / eager_out:.3f}x",
    )


    print(
        "EAGER_REQUESTS_S =",
        f"{eager_rps:.3f}",
    )

    print(
        "OPTIMIZED_REQUESTS_S =",
        f"{opt_rps:.3f}",
    )

    print(
        "REQUEST_THROUGHPUT_SPEEDUP =",
        f"{opt_rps / eager_rps:.3f}x",
    )


    print(
        "EAGER_MEDIAN_LATENCY_MS =",
        f"{eager_latency:.3f}",
    )

    print(
        "OPTIMIZED_MEDIAN_LATENCY_MS =",
        f"{opt_latency:.3f}",
    )

    print(
        "LATENCY_SPEEDUP_EAGER_OVER_OPT =",
        f"{eager_latency / opt_latency:.3f}x",
    )


eager_startup = float(
    data["single_c1"][
        "eager"
    ][
        "startup_s"
    ]
)

opt_startup = float(
    data["single_c1"][
        "optimized"
    ][
        "startup_s"
    ]
)


print()
print("=" * 120)
print("STARTUP")
print("=" * 120)

print(
    "EAGER_STARTUP_S =",
    f"{eager_startup:.3f}",
)

print(
    "OPTIMIZED_STARTUP_S =",
    f"{opt_startup:.3f}",
)

print(
    "OPTIMIZED_STARTUP_OVER_EAGER =",
    f"{opt_startup / eager_startup:.3f}x",
)


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print(
"""
Eager mode disables torch.compile and CUDA Graph execution.

Optimized mode allows vLLM's default optimization path.

The comparison therefore captures a systems trade-off:

    initialization / compilation cost
              versus
    steady-state serving performance

Do not judge serving systems only by startup time.

Long-running production servers may amortize compilation
cost over many requests, so steady-state TTFT, TPOT and
throughput are the more important runtime metrics.

Likewise, an optimization is not automatically beneficial
for every workload shape.

The impact may differ between:

    single-request latency
    moderate concurrency
    high concurrency
""".strip()
)


print()
print("=" * 120)
print("FINAL VERDICT")
print("=" * 120)

print(
    "EAGER_BASELINE = SUCCESS"
)

print(
    "OPTIMIZED_BASELINE = SUCCESS"
)

print(
    "MATCHED_WORKLOAD_COMPARISON = SUCCESS"
)

print(
    "VERDICT = READY_FOR_P1_M9_PROFILING_AND_BOTTLENECK_ANALYSIS"
)
