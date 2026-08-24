# TurboLLM Project Summary

## Project goal

Build an evidence-driven LLM serving performance-engineering project rather than only reporting one tokens-per-second result.

The project investigates the complete path from GPU execution to production LLM serving.

## Stage 1 - GPU fundamentals

Controlled experiments covered:

- CPU vs GPU matrix multiplication
- FP16 execution
- PCIe transfer cost
- pinned-memory transfer
- asynchronous CUDA kernel execution
- CUDA allocator behavior
- global-memory coalescing
- shared-memory tiling
- shared-memory bank conflict avoidance
- warp divergence

## Stage 2 - LLM serving

A real vLLM serving environment was built using Qwen2.5-0.5B-Instruct.

Serving metrics included:

- TTFT
- TPOT
- latency
- request throughput
- prompt throughput
- output throughput

## Stage 3 - Scheduler behavior

Reducing max_num_seqs constrained active sequence admission.

Observed effects included:

- fewer running requests
- increased waiting requests
- increased queue time
- worse tail latency

## Stage 4 - KV Cache

Artificial KV-cache reduction produced:

- KV-cache saturation
- sequence preemption
- serving throughput degradation

This demonstrated that KV allocation and scheduling are coupled.

## Stage 5 - Prefix caching

A controlled shared-prefix workload directly measured:

- prefix-cache queries
- prefix-cache hits
- newly computed KV tokens

This avoided inferring cache effectiveness only from latency.

## Stage 6 - Production GPU profiling

Nsight Compute hardware counters were blocked by the cloud host with ERR_NVGPUCTRPERM.

The project therefore used PyTorch/Kineto to trace actual vLLM execution.

Measured trace evidence included:

- CUDA kernel names
- CUDA kernel launch counts
- CUDA kernel execution duration
- relative CUDA-time composition

Prefill-heavy and Decode-heavy traffic were profiled separately.

## Stage 7 - CUDA Graph matched ablation

torch.compile remained enabled for every configuration.

Only CUDA Graph mode changed:

1. NONE
2. PIECEWISE
3. FULL_AND_PIECEWISE

Decode-heavy output throughput:

- NONE: 784.490 tok/s
- PIECEWISE: 1222.117 tok/s
- FULL_AND_PIECEWISE: 2989.341 tok/s

FULL_AND_PIECEWISE achieved:

- 3.8106x throughput vs NONE
- 2.4460x throughput vs PIECEWISE

TPOT improved from:

9.603 ms -> 2.106 ms

Prefill throughput winner:

full_and_piecewise

Decode throughput winner:

full_and_piecewise

Recommended serving mode:

full_and_piecewise

Maximum primary-metric three-trial coefficient of variation:

5.335%

The result is retained as valid engineering evidence with explicit variance caution.

## Final systems workflow

measure
-> characterize
-> stress
-> profile
-> attribute
-> optimize
-> matched ablation
-> evidence freeze
