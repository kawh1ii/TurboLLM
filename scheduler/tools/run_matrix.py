#!/usr/bin/env python3
"""Run bounded paired experiments, managing only server PIDs created here.

Run calibration separately, then evaluate on other seeds. Reports retain every
failure. Timing files and scheduler traces are separated for later replay.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def wait_ready(proc,port,timeout=600):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited; inspect server log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as r:
                if r.status==200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError("server startup timed out")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()


def run(a):
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    modes=a.modes.split(",")
    config=json.loads(Path(a.config).read_text()) if a.config else {}
    # Independent held-out seeds; identical workload files are regenerated from
    # the same seed for each mode, with predetermined open-loop arrival offsets.
    seeds=[int(s) for s in a.seeds.split(",")]
    for mode_index,mode in enumerate(modes):
        port=a.port+mode_index
        prefix=out/mode
        trace=str(prefix)+".trace.jsonl"
        if Path(trace).exists():
            raise FileExistsError("Refusing to append to previous experiment: "+trace)
        cfg=dict(config)
        if mode=="no_reserve": cfg["enable_reservations"]=False
        if mode=="no_time": cfg["enable_time_budget"]=False
        if mode.startswith("budget-h"):
            cfg["decode_horizon"]=int(mode.removeprefix("budget-h"))
        cfgpath=out/(mode+".config.json")
        cfgpath.write_text(json.dumps(cfg,indent=2))
        policy=mode in ("budget","no_reserve","no_time") or mode.startswith("budget-h")
        cmd=[sys.executable,str(ROOT/"tools/serve.py"),"--model",a.model,
             "--mode","budget" if policy else "trace","--port",str(port),
             "--config",str(cfgpath),
             "--max-model-len",str(a.max_model_len),
             "--max-num-seqs",str(a.max_num_seqs),
             "--max-num-batched-tokens",str(a.token_budget)]
        if not a.no_trace: cmd += ["--trace",trace]
        if a.eager: cmd += ["--eager"]
        if a.batch_invariant: cmd += ["--batch-invariant"]
        if a.kv_blocks: cmd += ["--kv-blocks",str(a.kv_blocks)]
        if mode=="tuned" or mode.startswith("tuned-"):
            threshold,watermark = mode.split("-")[1:] if mode.startswith("tuned-") else ("256","0.05")
            cmd += ["--watermark",watermark,"--long-prefill-threshold",threshold,"--adaptive-threshold"]
        log=open(str(prefix)+".server.log","w")
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            wait_ready(proc,port)
            print("READY",mode,flush=True)
            if a.correctness:
                check=[sys.executable,str(ROOT/"tools/correctness.py"),"--base",f"http://127.0.0.1:{port}",
                       "--output",str(prefix)+".correctness.json"]
                reference=out/"native.correctness.json"
                if mode!="native" and reference.exists():
                    check += ["--reference",str(reference)]
                subprocess.run(check,check=True)
            for seed in seeds:
                dest=out/f"{mode}-{a.kind}-s{seed}.json"
                subprocess.run([sys.executable,str(ROOT/"tools/benchmark.py"),
                                "--base",f"http://127.0.0.1:{port}",
                                "--output",str(dest),"--kind",a.kind,"--seed",str(seed),
                                "--count",str(a.count),"--rate",str(a.rate),
                                "--long-prompt",str(a.long_prompt)],check=True)
                time.sleep(1)
        finally:
            stop(proc)
            log.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--config")
    p.add_argument("--modes",default="native,tuned,budget")
    p.add_argument("--seeds",default="43,44,45")
    p.add_argument("--kind",default="mixed",choices=["mixed","burst","prefix","low","cancel"])
    p.add_argument("--count",type=int,default=48)
    p.add_argument("--rate",type=float,default=8)
    p.add_argument("--long-prompt",type=int,default=1024)
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--kv-blocks",type=int)
    p.add_argument("--max-model-len",type=int,default=4096)
    p.add_argument("--max-num-seqs",type=int,default=32)
    p.add_argument("--token-budget",type=int,default=2048)
    p.add_argument("--eager",action="store_true")
    p.add_argument("--batch-invariant",action="store_true")
    p.add_argument("--correctness",action="store_true")
    p.add_argument("--no-trace",action="store_true")
    run(p.parse_args())
