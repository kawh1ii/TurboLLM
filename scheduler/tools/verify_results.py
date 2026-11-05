#!/usr/bin/env python3
"""Audit the delivered evidence without rerunning GPU inference."""
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
out=ROOT/"results/final"

subprocess.run([sys.executable,str(ROOT/"tools/verify_sources.py"),"--results",str(out)],
               check=True,stdout=subprocess.DEVNULL)
stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
test_result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
assert test_result.wasSuccessful(),stream.getvalue()

stages=list(out.glob("*/.done.json"))
assert len(stages)==14,len(stages)
statuses=Counter()
request_files=0
for f in out.rglob("*.json"):
    if f.name.endswith((".summary.json",".workload.json")):continue
    d=json.loads(f.read_text())
    if not isinstance(d,dict) or "elapsed_s" not in d or "requests" not in d:continue
    request_files+=1
    for row in d["requests"]:
        statuses[row["status"]]+=1
        if row["status"]=="cancelled":
            assert f.parent.name=="cancel",str(f)
        else:
            assert row["status"]=="completed",(str(f),row["id"],row["status"])
            assert len(row["token_ids"])==row["requested_output"]
            assert row["output_tokens"]==row["requested_output"]
assert request_files==87,request_files

traces=steps=policy_steps=0
for f in out.rglob("*.trace.jsonl"):
    records=[json.loads(s) for s in f.read_text().splitlines()]
    initial=next(r for r in records if r["event"]=="init")
    states=[r for r in records if "requests_after" in r]
    for r in records:
        if r["event"]!="step":continue
        steps+=1
        assert sum(r["scheduled"].values())<=initial["max_tokens"]
        if "quotas" in r:
            policy_steps+=1
            assert all(0<n<=r["quotas"].get(rid,0) for rid,n in r["scheduled"].items())
    assert states[-1]["requests_after"]==0,str(f)
    assert states[-1]["free_blocks_after"]==initial["free_blocks_initial"],str(f)
    assert not states[-1].get("ledger_after"),str(f)
    traces+=1

strict=ROOT/"results/correctness-final"
native=json.loads((strict/"native.correctness.json").read_text())["requests"]
budget=json.loads((strict/"budget.correctness.json").read_text())["requests"]
assert len(native)==len(budget)==17
assert all(x["id"]==y["id"] and x["token_ids"]==y["token_ids"] for x,y in zip(native,budget))
subprocess.run([sys.executable,str(ROOT/"tools/compare_requests.py"),
                str(strict/"native-mixed-s61.json"),str(strict/"budget-mixed-s61.json"),
                "--output",str(strict/"pressure-output-check.json")],
               check=True,stdout=subprocess.DEVNULL)
assert json.loads((strict/"pressure-output-check.json").read_text())["exact_matches"]==16

manifest=json.loads((ROOT/"patches/manifest.json").read_text())
states={}
for phase in ["production-mixed","production-pressure"]:
    states[phase]={}
    for mode in ["native","tuned","budget"]:
        state=json.loads((out/phase/(mode+".source-state.json")).read_text())
        assert state["patched"]==(mode=="budget")
        expected=manifest["patched_sha256" if mode=="budget" else "original_sha256"]
        assert state["sha256"]==expected
        states[phase][mode]=state

verification={"unit_tests":{"passed":True,"count":test_result.testsRun},
              "strict_output":{"special_cases":17,"concurrent_pressure":16,"exact_matches":33,
                               "mode":"batch invariant, BF16, eager, Qwen2.5-3B-Instruct"},
              "completed_stages":len(stages),"request_files":request_files,
              "request_records":sum(statuses.values()),"statuses":dict(statuses),
              "trace_files":traces,"checked_engine_steps":steps,
              "checked_policy_steps":policy_steps,"quota_and_drain_checks_passed":True,
              "source_state_checks":states,
              "scope":"Audits saved evidence and pure policy tests; does not rerun GPU inference."}
(out/"verification.json").write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2))
