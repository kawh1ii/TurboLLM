# TurboLLM Experiment Map

This repository is organized as a systems-performance investigation rather than a collection of unrelated benchmarks.

## Phase A — GPU Fundamentals

| Milestone | Topic |
|---|---|
| P1-M1 | PyTorch GPU execution, transfer, async execution |
| P1-M2 | CUDA execution model |
| P1-M3 | Coalescing, shared memory, warp divergence |
| P1-M4 | GPU performance / arithmetic-intensity model |
| P1-M5 | Transformer Prefill, Decode and KV Cache |

## Phase B — Production LLM Serving

| Milestone | Topic |
|---|---|
| P1-M6 | vLLM serving baseline |
| P1-M7 | TTFT / TPOT / serving characterization |
| P1-M8 | Eager vs optimized execution |
| P1-M9 | Bottleneck profiling |

## Phase C — Scheduler and Memory System

| Milestone | Topic |
|---|---|
| P1-M10 | Scheduler, KV-cache pressure, prefix caching |

## Phase D — Production GPU Profiling

| Milestone | Topic |
|---|---|
| P1-M11 | Nsight Compute capability boundary |
| P1-M11B | PyTorch/Kineto production CUDA trace |
| P1-M12 | Prefill-heavy vs Decode-heavy kernel attribution |

## Phase E — Targeted Optimization

| Milestone | Topic |
|---|---|
| P1-M13 | Matched CUDA Graph ablation |
| P1-M14 | Optimization decision and evidence freeze |

## Final optimization result

Decode-heavy workload:

- NONE: 784.490 output tok/s
- PIECEWISE: 1222.117 output tok/s
- FULL_AND_PIECEWISE: 2989.341 output tok/s

FULL_AND_PIECEWISE vs NONE:

**3.8106x throughput**

TPOT:

**9.603 ms -> 2.106 ms**

## Reading order

For a quick project review:

1. README.md
2. docs/PROJECT_SUMMARY.md
3. docs/EXPERIMENT_MAP.md
4. docs/P1_FINAL_OPTIMIZATION_REPORT_*.md
5. docs/INTERVIEW_GUIDE.md

For implementation details:

1. src/
2. scripts/
3. results/
