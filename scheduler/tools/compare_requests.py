#!/usr/bin/env python3
"""Strictly compare replayed workloads and complete output token IDs by ID."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("reference")
p.add_argument("candidate")
p.add_argument("--output",required=True)
a=p.parse_args()
left,right=map(Path,(a.reference,a.candidate))
assert json.loads(Path(str(left)+".workload.json").read_text())==json.loads(Path(str(right)+".workload.json").read_text()),"workloads differ"
x={r["id"]:r for r in json.loads(left.read_text())["requests"]}
y={r["id"]:r for r in json.loads(right.read_text())["requests"]}
assert x.keys()==y.keys(),"request sets differ"
diffs=[]
for rid in x:
    if x[rid]["status"]!="completed" or y[rid]["status"]!="completed":
        raise RuntimeError("incomplete request: "+rid)
    u,v=x[rid]["token_ids"],y[rid]["token_ids"]
    if u!=v:
        k=next((i for i,(a,b) in enumerate(zip(u,v)) if a!=b),min(len(u),len(v)))
        diffs.append({"id":rid,"first_divergent_token":k,"reference_length":len(u),"candidate_length":len(v)})
result={"reference":str(left),"candidate":str(right),"requests":len(x),
        "exact_matches":len(x)-len(diffs),"differences":diffs}
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if diffs:raise SystemExit("output token mismatch")
