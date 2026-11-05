#!/usr/bin/env python3
"""Reproducible staged validation. Tuning seeds and evaluation seeds are disjoint.

All stages are sequential on the same GPU. A failed stage is never marked done.
Resume requires matching execution-source hashes; old evidence is not silently
reused after a code change. Large model downloads are handled separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/"tools"


def execution_hash():
    h=hashlib.sha256()
    paths=[p for p in (ROOT/"src").rglob("*.py") if not p.name.startswith("._")]
    paths += [TOOLS/(s+".py") for s in
              ["serve","benchmark","correctness","fit_cost","run_matrix"]]
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def call(script,*args):
    command=[sys.executable,str(TOOLS/script),*map(str,args)]
    print("RUN",json.dumps(command),flush=True)
    subprocess.run(command,check=True)


def verify(directory,cancel=False):
    for f in directory.glob("*-s*.json"):
        if f.name.endswith((".summary.json",".workload.json")):
            continue
        d=json.loads(f.read_text())
        for row in d["requests"]:
            allowed=("completed","cancelled") if cancel else ("completed",)
            if row["status"] not in allowed:
                raise RuntimeError(f"request failure in {f}: {row['id']}: {row['status']}")
    traces=list(directory.glob("*.trace.jsonl"))
    if traces:
        call("check_trace.py",*traces)


def main(a):
    base=Path(a.output).resolve()
    base.mkdir(parents=True,exist_ok=True)
    source_hash=execution_hash()
    def stage(name,fn,cancel=False):
        dest=base/name
        done=dest/".done.json"
        if done.exists():
            recorded=json.loads(done.read_text())["execution_sha256"]
            if recorded!=source_hash:
                audit_file=base/"source-hash-audit.json"
                audit=json.loads(audit_file.read_text()) if audit_file.exists() else {}
                if audit.get("portable_sha256")!=source_hash or audit.get("legacy_sha256")!=recorded:
                    raise RuntimeError("Execution code changed; use a new output directory")
            print("RESUME",name,flush=True)
            return
        if dest.exists() and any(dest.iterdir()):
            raise RuntimeError("Incomplete stage exists; inspect/archive before retrying: "+str(dest))
        dest.mkdir(exist_ok=True)
        started=time.time()
        fn(dest)
        verify(dest,cancel=cancel)
        done.write_text(json.dumps({"execution_sha256":source_hash,"completed_unix":time.time(),
                                    "wall_seconds":time.time()-started},indent=2))
        print("STAGE_COMPLETE",name,flush=True)

    def matrix(dest,model=a.model,config=None,modes="native,tuned,budget",kind="mixed",
               seeds=a.eval_seeds,count=48,rate=8,kv=None,extra=()):
        cmd=["--model",model,"--output",dest,"--modes",modes,"--kind",kind,
             "--seeds",seeds,"--count",count,"--rate",rate,"--max-model-len",2048]
        if config:cmd += ["--config",config]
        if kv:cmd += ["--kv-blocks",kv]
        call("run_matrix.py",*cmd,*extra)

    stage("calibration",lambda d:matrix(d,modes="native",seeds=str(a.calibration_seed),count=64))
    fitted=base/"calibrated.json"
    if not fitted.exists():
        call("fit_cost.py",base/"calibration/native.trace.jsonl","--output",fitted)
    tune_modes="tuned-128-0,tuned-256-0.05,tuned-512-0.1,budget-h16,budget-h64,budget-h128"
    stage("tuning",lambda d:matrix(d,config=fitted,modes=tune_modes,seeds=str(a.tuning_seed),count=32,kv=256))
    candidates={}
    from analyze import summarize
    for f in (base/"tuning").glob(f"*-s{a.tuning_seed}.json"):
        data=json.loads(f.read_text())
        name=f.name.split(f"-mixed-s{a.tuning_seed}")[0]
        metric=summarize(data)
        candidates[name]={"goodput":metric["goodput_requests_s"],
                          "output_throughput":metric["output_tokens_s"]}
    key=lambda name:(candidates[name]["goodput"],candidates[name]["output_throughput"])
    tuned=max([m for m in candidates if m.startswith("tuned-")],key=key)
    policy=max([m for m in candidates if m.startswith("budget-h")],key=key)
    frozen=base/"frozen-policy.json"
    config=json.loads(fitted.read_text())
    config["decode_horizon"]=int(policy.removeprefix("budget-h"))
    frozen.write_text(json.dumps(config,indent=2))
    (base/"selection.json").write_text(json.dumps({"seed":a.tuning_seed,"objective":["goodput","output_throughput"],
        "candidates":candidates,"selected_native":tuned,"selected_policy":policy,
        "warning":"Finite candidate search, not a globally optimal native baseline."},indent=2))
    modes="native,"+tuned+",budget"
    stage("mixed",lambda d:matrix(d,config=frozen,modes=modes))
    stage("pressure",lambda d:matrix(d,config=frozen,modes=modes,kv=256))
    stage("ablation",lambda d:matrix(d,config=frozen,modes="no_reserve,no_time",kv=256))
    stage("burst",lambda d:matrix(d,config=frozen,modes="native,budget",kind="burst",kv=256))
    stage("prefix",lambda d:matrix(d,config=frozen,modes="native,budget",kind="prefix",count=32))
    stage("long8k",lambda d:matrix(d,config=frozen,modes="native,budget",count=24,rate=4,
                                  extra=("--long-prompt",8192,"--max-model-len",10240)))
    stage("low",lambda d:matrix(d,config=frozen,modes="native,budget",kind="low",count=16,rate=2))
    stage("cancel",lambda d:matrix(d,config=frozen,modes="budget",kind="cancel",seeds="91",count=16),cancel=True)
    stage("trace-overhead",lambda d:matrix(d,config=frozen,modes=modes,kv=256,seeds=str(a.audit_seed)))
    stage("no-trace",lambda d:matrix(d,config=frozen,modes=modes,kv=256,seeds=str(a.audit_seed),extra=("--no-trace",)))
    if a.model7b:
        stage("7b-calibration",lambda d:matrix(d,model=a.model7b,modes="native",seeds=str(a.calibration_seed),count=48,rate=4))
        fitted7=base/"calibrated7b.json"
        if not fitted7.exists():
            call("fit_cost.py",base/"7b-calibration/native.trace.jsonl","--output",fitted7)
        stage("7b-pressure",lambda d:matrix(d,model=a.model7b,config=fitted7,
                                            modes="native,budget",count=32,rate=4,kv=256))
    call("report.py","--root",base)
    print("VALIDATION_COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--model7b")
    p.add_argument("--output",default="results/validation")
    p.add_argument("--calibration-seed",type=int,default=41)
    p.add_argument("--tuning-seed",type=int,default=42)
    p.add_argument("--eval-seeds",default="43,44,45")
    p.add_argument("--audit-seed",type=int,default=46)
    main(p.parse_args())
