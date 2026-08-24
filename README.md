# TurboLLM

LLM Serving Performance Engineering with vLLM, PyTorch and CUDA.

## Overview

TurboLLM is an AI Infra project that studies LLM inference performance from CUDA execution fundamentals through production serving, scheduler and KV-cache behavior, real CUDA kernel tracing, and targeted CUDA Graph optimization.

## Technical coverage

1. GPU execution fundamentals
2. CPU/GPU compute comparison
3. FP16 execution
4. Host-to-Device and Device-to-Host transfer
5. Pinned memory
6. CUDA asynchronous execution
7. PyTorch CUDA allocator
8. Global-memory coalescing
9. Shared-memory tiling
10. Warp divergence
11. Transformer Prefill and Decode
12. KV Cache
13. vLLM OpenAI-compatible serving
14. TTFT and TPOT
15. Continuous batching
16. Scheduler queueing
17. Prefix caching
18. KV-cache pressure
19. Request preemption
20. Production CUDA kernel tracing
21. Prefill-heavy vs Decode-heavy kernel attribution
22. CUDA Graph matched ablation

## Final Decode-heavy CUDA Graph result

Workload:

- Prompt length: approximately 128 tokens
- Output length: 512 tokens
- Concurrency: 8
- torch.compile: enabled for every mode
- Prefix caching: disabled

| CUDA Graph mode | Output throughput | TPOT |
|---|---:|---:|
| NONE | 784.490 tok/s | 9.603 ms |
| PIECEWISE | 1222.117 tok/s | 5.984 ms |
| FULL_AND_PIECEWISE | 2989.341 tok/s | 2.106 ms |

Measured Decode improvements:

- PIECEWISE vs NONE: **1.5578x**
- FULL_AND_PIECEWISE vs NONE: **3.8106x**
- FULL_AND_PIECEWISE vs PIECEWISE: **2.4460x**
- TPOT: **9.603 ms -> 2.106 ms**

## Prefill-heavy CUDA Graph result

Primary throughput metric: prompt tokens/s.

| CUDA Graph mode | Prompt throughput | TTFT |
|---|---:|---:|
| NONE | 34932.790 tok/s | 255.474 ms |
| PIECEWISE | 36840.488 tok/s | 287.495 ms |
| FULL_AND_PIECEWISE | 41717.247 tok/s | 309.705 ms |

FULL_AND_PIECEWISE Prefill throughput factor vs NONE:

**1.1942x**

Measured Prefill throughput winner:

**full_and_piecewise**

Measured Decode throughput winner:

**full_and_piecewise**

Recommended general serving mode:

**full_and_piecewise**

Recommendation reason:

**MEASURED_WINNER_ON_BOTH_PHASES**

## Performance-analysis workflow

Application metrics:
TTFT / TPOT / latency / tokens per second

Then:

Serving-system metrics:
scheduler queue / Prefill / Decode / KV Cache

Then:

GPU telemetry:
GPU utilization / VRAM / memory activity / power

Then:

Production CUDA trace:
kernel names / launch counts / duration

Then:

Targeted optimization:
matched CUDA Graph ablation

## Key experiments

### Scheduler

Constraining max_num_seqs increased waiting requests and queueing pressure and degraded tail latency.

### KV Cache

Artificial KV-cache reduction caused cache saturation, sequence preemption and throughput degradation.

### Prefix Cache

Controlled shared-prefix workloads directly measured prefix-cache hits and reduced newly computed KV tokens.

### Production profiling

PyTorch/Kineto was used to capture real vLLM CUDA kernel identities, launch counts and execution time.

### CUDA Graph optimization

The final experiment held torch.compile constant and changed only the CUDA Graph execution mode.

## Evidence boundary

Nsight Compute was installed and CUDA execution worked correctly.

The cloud host returned:

ERR_NVGPUCTRPERM

Therefore the project does not claim directly measured:

- DRAM bandwidth saturation
- Tensor Core utilization
- achieved occupancy
- detailed SM hardware-counter utilization

Those metrics require privileged NVIDIA GPU performance-counter access.

## Environment

- NVIDIA GeForce RTX 4090
- vLLM 0.27.1+cu129
- PyTorch 2.13.0+cu129
- CUDA Toolkit 12.8
- Qwen2.5-0.5B-Instruct

## Status

PROJECT1_SCIENTIFIC_EXPERIMENTS = COMPLETE

PROJECT1_EVIDENCE_FREEZE = COMPLETE

PROJECT1_STATUS = PORTFOLIO_READY
