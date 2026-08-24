# Project 1 — TurboLLM Optimization Evidence Freeze

## 1. Scope

This report freezes the final CUDA Graph optimization decision from P1-M13.

No new model execution was performed in P1-M14.

Frozen source:

`/root/autodl-tmp/ai-infra/project1-turbollm/results/p1_m13/P1_M13_CUDAGRAPH_ABLATION_20260824_220016.csv`

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
| NONE | 34932.790 | 255.474 | 31.454 | 17311.0 |
| PIECEWISE | 36840.488 | 287.495 | 48.757 | 17187.0 |
| FULL_AND_PIECEWISE | 41717.247 | 309.705 | 30.474 | 17427.0 |

Prefill throughput winner:

**full_and_piecewise**

Prefill TTFT winner:

**none**

PIECEWISE throughput speedup vs NONE:

**1.0546×**

FULL_AND_PIECEWISE throughput speedup vs NONE:

**1.1942×**

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
| NONE | 784.490 | 9.603 | 5212.891 | 17311.0 |
| PIECEWISE | 1222.117 | 5.984 | 3341.313 | 17187.0 |
| FULL_AND_PIECEWISE | 2989.341 | 2.106 | 1360.906 | 17427.0 |

Decode throughput winner:

**full_and_piecewise**

Decode TPOT winner:

**full_and_piecewise**

PIECEWISE throughput speedup vs NONE:

**1.5578×**

FULL_AND_PIECEWISE throughput speedup vs NONE:

**3.8106×**

FULL_AND_PIECEWISE incremental throughput speedup vs PIECEWISE:

**2.4460×**

---

## 4. Stability

Maximum coefficient of variation across available primary 3-trial measurements:

**5.335%**

Stability classification:

**RESULTS_VALID_BUT_VARIANCE_REQUIRES_CAUTION**

The coefficient of variation is used only as a repeatability diagnostic. It is not presented as a formal statistical confidence interval.

---

## 5. Balanced mixed-workload summary

This is a decision heuristic, not a measured production workload.

Balanced index = geometric mean of:

1. Prefill prompt-throughput factor vs NONE.
2. Decode output-throughput factor vs NONE.

| Mode | Balanced index |
|---|---:|
| NONE | 1.0000 |
| PIECEWISE | 1.2818 |
| FULL_AND_PIECEWISE | 2.1332 |

Balanced winner:

**full_and_piecewise**

---

## 6. Final serving decision

Best measured Prefill throughput mode:

**full_and_piecewise**

Best measured Decode throughput mode:

**full_and_piecewise**

Recommended general serving mode under the project decision policy:

**full_and_piecewise**

Reason:

**MEASURED_WINNER_ON_BOTH_PHASES**

FULL_AND_PIECEWISE is the measured throughput winner for both the Prefill-heavy and Decode-heavy workloads in this ablation. The previously defined 5% engineering decision margin therefore does not determine the final choice in this run.

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



---

## 9. Evidence Freeze Note

The final recommendation is based on the measured matched
CUDA Graph ablation.

`FULL_AND_PIECEWISE` is the measured throughput winner for
both evaluated phase-emphasis workloads.

Maximum observed three-trial primary-metric coefficient of
variation was 5.335%, so the result is retained as valid
engineering evidence with an explicit variance caution.

No additional scientific model execution was performed
during the final consistency correction.

**FINAL EVIDENCE STATUS: FROZEN**
