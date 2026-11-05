#!/usr/bin/env python3
"""Explicit launch command shared by baseline and policy; no hidden defaults."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--mode", choices=["native", "trace", "budget"], default="trace")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--max-model-len", type=int, default=4096)
p.add_argument("--max-num-seqs", type=int, default=32)
p.add_argument("--max-num-batched-tokens", type=int, default=2048)
p.add_argument("--kv-blocks", type=int)
p.add_argument("--block-size", type=int, default=16)
p.add_argument("--config")
p.add_argument("--trace")
p.add_argument("--eager", action="store_true")
p.add_argument("--batch-invariant", action="store_true")
p.add_argument("--no-prefix-cache", action="store_true")
p.add_argument("--watermark", type=float, default=0.0)
p.add_argument("--long-prefill-threshold", type=int, default=0)
p.add_argument("--adaptive-threshold", action="store_true")
p.add_argument("--max-active-seqs", type=int)
p.add_argument("--host", default="127.0.0.1")
a = p.parse_args()
env = os.environ.copy()
env["TOKENIZERS_PARALLELISM"] = "false"
env["OMP_NUM_THREADS"] = "4"
env["VLLM_BATCH_INVARIANT"] = "1" if a.batch_invariant else "0"
env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
env.pop("TURBOLLM_CONFIG", None)
env.pop("TURBOLLM_TRACE", None)
if a.config:
    env["TURBOLLM_CONFIG"] = str(Path(a.config).resolve())
if a.trace:
    env["TURBOLLM_TRACE"] = str(Path(a.trace).resolve())
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
       "--host", a.host, "--port", str(a.port), "--model", a.model,
       "--served-model-name", "turbollm", "--dtype", "bfloat16",
       "--attention-backend", "FLASH_ATTN",
       "--max-model-len", str(a.max_model_len), "--max-num-seqs", str(a.max_num_seqs),
       "--max-num-batched-tokens", str(a.max_num_batched_tokens),
       "--gpu-memory-utilization", "0.85", "--block-size", str(a.block_size),
       "--no-async-scheduling",
       "--watermark", str(a.watermark),
       "--long-prefill-token-threshold", str(a.long_prefill_threshold)]
if a.mode != "native":
    cls = "BudgetScheduler" if a.mode == "budget" else "TraceScheduler"
    cmd += ["--scheduler-cls", "turbollm.vllm_scheduler." + cls]
if a.eager:
    cmd += ["--enforce-eager"]
if a.no_prefix_cache:
    cmd += ["--no-enable-prefix-caching"]
if a.kv_blocks:
    cmd += ["--num-gpu-blocks-override", str(a.kv_blocks)]
if a.max_active_seqs:
    cmd += ["--max-num-active-seqs", str(a.max_active_seqs)]
if a.adaptive_threshold:
    cmd += ["--long-prefill-token-threshold-adaptive"]
print(json.dumps({"command": cmd, "mode": a.mode}, indent=2), flush=True)
# Replace the wrapper so the experiment manager waits for the actual API server
# and its shutdown, rather than just observing a dead wrapper with live children.
os.execvpe(cmd[0], cmd, env)
