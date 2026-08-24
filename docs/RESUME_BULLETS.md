# TurboLLM Resume Bullets

## Recommended four-bullet version

TurboLLM - LLM Inference Serving Performance Engineering

- Built an end-to-end LLM serving performance stack using vLLM, PyTorch and CUDA, covering TTFT/TPOT benchmarking, continuous batching, scheduler queueing, KV-cache pressure, prefix caching and production CUDA kernel tracing.
- Designed a matched CUDA Graph ablation while holding torch.compile constant; increased Decode-heavy serving throughput from 784 to 2989 tokens/s (3.81x) and reduced TPOT from 9.60 ms to 2.11 ms using FULL_AND_PIECEWISE.
- Built controlled scheduler and KV-cache stress experiments demonstrating request queueing, KV saturation and sequence preemption under constrained serving resources.
- Profiled production Prefill-heavy and Decode-heavy vLLM execution with PyTorch/Kineto, attributing CUDA kernel identities, launch counts and relative execution time.

## Recommended two-bullet version

- Built a vLLM/CUDA LLM serving performance-engineering system covering TTFT/TPOT, continuous batching, scheduler/KV-cache pressure, prefix caching and production CUDA kernel tracing.
- Isolated CUDA Graph execution with matched workloads and improved Decode-heavy throughput by 3.81x (784 -> 2989 tok/s), reducing TPOT from 9.60 -> 2.11 ms.

## One-line version

Built a vLLM/CUDA inference-performance project covering batching, scheduler/KV-cache analysis and production GPU tracing, achieving 3.81x Decode throughput improvement through a matched CUDA Graph ablation.
