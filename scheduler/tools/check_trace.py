#!/usr/bin/env python3
"""Check actual allocations against plans and require fully drained resources."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("files",nargs="+")
a=p.parse_args()
for file in a.files:
    records=[json.loads(s) for s in Path(file).read_text().splitlines()]
    initial=next(r for r in records if r["event"]=="init")
    states=[]
    steps=0
    for r in records:
        if "requests_after" in r:
            states.append(r)
        if r["event"]!="step":
            continue
        steps+=1
        assert sum(r["scheduled"].values())<=initial["max_tokens"], (file,r["step"])
        if "quotas" in r:
            for rid,n in r["scheduled"].items():
                assert 0<n<=r["quotas"].get(rid,0), (file,r["step"],rid)
    assert states, "no execution states"
    last=states[-1]
    assert last["requests_after"]==0, (file,"requests not drained",last)
    assert last["free_blocks_after"]==initial["free_blocks_initial"], (file,"KV blocks not restored",last)
    assert not last.get("ledger_after"), (file,"custom reservations not drained",last)
    print(json.dumps({"file":file,"steps":steps,"allocations_within_budget":True,
                      "fully_drained":True,"free_blocks":last["free_blocks_after"]}))
