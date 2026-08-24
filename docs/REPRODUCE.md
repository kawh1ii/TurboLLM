# TurboLLM Reproduction Guide

## Project structure

project1-turbollm/

README.md
scripts/
src/
results/
docs/

## Experimental progression

P1-M1
GPU execution fundamentals

P1-M3
CUDA memory access, shared memory and warp behavior

P1-M5
Transformer inference, Prefill, Decode and KV Cache

P1-M6
vLLM serving baseline

P1-M7
Serving characterization

P1-M8
Eager vs optimized serving

P1-M9
Profiling and bottleneck analysis

P1-M10
Scheduler, KV cache and prefix cache

P1-M11
Nsight Compute capability test

P1-M11B
Production PyTorch/Kineto trace

P1-M12
Prefill-heavy and Decode-heavy kernel attribution

P1-M13
Matched CUDA Graph ablation

P1-M14
Optimization decision and evidence freeze

## Important directories

Execution scripts:
scripts/

Source code:
src/

Structured results:
results/

Documentation:
docs/

Frozen TXT artifacts:
/root/autodl-tmp/ai-infra/artifacts/

## Hardware-counter limitation

The original AutoDL host returned:

ERR_NVGPUCTRPERM

Reproducing Nsight Compute hardware-counter measurements requires a GPU host that exposes NVIDIA performance counters.
