# TurboLLM Interview Guide

## 30-second introduction

I built an LLM serving performance-engineering project around vLLM, PyTorch and CUDA.

Instead of only measuring tokens per second, I decomposed serving performance into TTFT, TPOT, scheduler queueing, KV-cache behavior, Prefill/Decode phases and real CUDA kernel execution.

The final experiment held torch.compile constant and changed only CUDA Graph strategy.

On a Decode-heavy workload, FULL_AND_PIECEWISE increased throughput from approximately 784 to 2989 tokens/s, a 3.81x improvement, while TPOT fell from 9.60 ms to 2.11 ms.

## Why not only benchmark tokens per second?

Because a throughput change may come from:

- batching
- scheduler behavior
- cache reuse
- KV-cache pressure
- Prefill behavior
- Decode behavior
- GPU launch overhead

So the project used controlled workload matrices.

## What is TTFT?

Time To First Token.

It captures:

request arrival
-> queueing
-> scheduling
-> Prefill
-> first generated token

## What is TPOT?

Time Per Output Token.

It characterizes steady-state Decode performance after the first token.

## What did the scheduler experiment show?

Reducing max_num_seqs capped active sequences.

This increased waiting requests, queue time and tail latency.

## What did the KV-cache experiment show?

Reducing KV-cache capacity caused:

- cache saturation
- preemption
- lower serving throughput

KV allocation and request scheduling are therefore coupled.

## What did prefix caching show?

A common prefix was explicitly warmed.

The experiment directly measured:

- prefix-cache queries
- cache hits
- newly computed KV tokens

instead of inferring caching only from latency.

## Why CUDA Graph?

Decode consists of many repeated GPU execution iterations.

Without CUDA Graph, the CPU repeatedly launches GPU kernels.

CUDA Graph captures reusable GPU execution and reduces repeated launch overhead.

Matched Decode result:

NONE:
784.490 tok/s, TPOT 9.603 ms

PIECEWISE:
1222.117 tok/s, TPOT 5.984 ms

FULL_AND_PIECEWISE:
2989.341 tok/s, TPOT 2.106 ms

## Why did you not claim measured DRAM saturation or Tensor Core utilization?

Nsight Compute returned ERR_NVGPUCTRPERM.

The cloud host did not expose privileged NVIDIA hardware performance counters.

Therefore I deliberately did not claim measured:

- DRAM saturation
- Tensor Core utilization
- occupancy
- detailed SM utilization

PyTorch/Kineto was used for production CUDA kernel timing instead.

## Core engineering lesson

workload
-> metric
-> bottleneck evidence
-> subsystem
-> optimization
-> matched ablation
