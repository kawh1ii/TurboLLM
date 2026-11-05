#!/usr/bin/env python3
"""Compare against the actual unwrapped, restored upstream scheduler.

Run ONLY after all other experiment servers have stopped. Native/tuned runs
restore the exact original source; budget runs apply the audited hooks. No
trace writer or baseline tracing subclass is used in this comparison.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_matrix import wait_ready,stop

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    dest=Path(a.output).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(a.selection).read_text())
    threshold,watermark=selection["selected_native"].split("-")[1:]
    protocol={"baseline":"exact original vLLM scheduler; no tracing subclass",
              "policy":"BudgetScheduler with trace writing disabled",
              "seeds":a.seeds,"kv_blocks":a.kv_blocks,"count":a.count,"rate":a.rate,
              "config":json.loads(Path(a.config).read_text()),
              "native_tuning_selection":selection["selected_native"]}
    (dest/"protocol.json").write_text(json.dumps(protocol,indent=2))
    try:
        for i,mode in enumerate(a.modes.split(",")):
            if any(dest.glob(mode+"-mixed-s*.json")):
                raise FileExistsError("Refusing to overwrite completed or partial runs")
            patch=[sys.executable,str(ROOT/"tools/patch_vllm.py")]
            if mode!="budget":patch += ["--restore"]
            subprocess.run(patch,check=True)
            state=json.loads(subprocess.check_output(
                [sys.executable,str(ROOT/"tools/patch_vllm.py"),"--check"],text=True))
            assert state["patched"] == (mode=="budget")
            (dest/(mode+".source-state.json")).write_text(json.dumps(state,indent=2))
            port=a.port+i
            cmd=[sys.executable,str(ROOT/"tools/serve.py"),"--model",a.model,
                 "--mode","budget" if mode=="budget" else "native",
                 "--config",a.config,"--port",str(port),"--max-model-len","2048",
                 "--max-num-seqs","32","--max-num-batched-tokens","2048"]
            if a.kv_blocks:cmd += ["--kv-blocks",str(a.kv_blocks)]
            if mode=="tuned":
                cmd += ["--long-prefill-threshold",threshold,"--watermark",watermark,"--adaptive-threshold"]
            log=(dest/(mode+".server.log")).open("w")
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                wait_ready(proc,port)
                print("PRODUCTION_READY",mode,flush=True)
                for seed in a.seeds.split(","):
                    subprocess.run([sys.executable,str(ROOT/"tools/benchmark.py"),
                                    "--base",f"http://127.0.0.1:{port}","--kind","mixed",
                                    "--seed",seed,"--count",str(a.count),"--rate",str(a.rate),
                                    "--output",str(dest/f"{mode}-mixed-s{seed}.json")],check=True)
            finally:
                stop(proc)
                log.close()
    finally:
        subprocess.run([sys.executable,str(ROOT/"tools/patch_vllm.py")],check=True)
    for f in dest.glob("*-mixed-s*.json"):
        if f.name.endswith((".workload.json",".summary.json")):continue
        d=json.loads(f.read_text())
        assert all(r["status"]=="completed" for r in d["requests"]),str(f)
    print("PRODUCTION_COMPARISON_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--selection",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--kv-blocks",type=int)
    p.add_argument("--seeds",default="43,44,45")
    p.add_argument("--count",type=int,default=48)
    p.add_argument("--rate",type=float,default=8)
    p.add_argument("--modes",default="native,tuned,budget")
    p.add_argument("--port",type=int,default=8100)
    run(p.parse_args())
