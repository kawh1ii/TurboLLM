#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def percentile(values, q):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    i = (len(values)-1)*q
    lo = int(i)
    return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)


def summarize(data, ttft_ms=1000, tpot_ms=40):
    rows, elapsed = data["requests"], data["elapsed_s"]
    complete = [r for r in rows if r["status"] == "completed"]
    passing = [r for r in complete if r.get("ttft_ms") is not None
               and r.get("tpot_ms") is not None
               and r["ttft_ms"] <= ttft_ms and r["tpot_ms"] <= tpot_ms]
    return {"statuses": dict(Counter(r["status"] for r in rows)),
            "ttft_target_ms": ttft_ms, "tpot_target_ms": tpot_ms,
            "elapsed_s": elapsed, "total_arrivals": len(rows),
            "completed": len(complete), "slo_pass": len(passing),
            "slo_pass_fraction_all_arrivals": len(passing)/len(rows) if rows else 0,
            "goodput_requests_s": len(passing)/elapsed,
            "output_tokens_s": sum(r.get("output_tokens",0) for r in complete)/elapsed,
            "ttft_p50_ms": percentile([r.get("ttft_ms") for r in complete],.5),
            "ttft_p95_ms": percentile([r.get("ttft_ms") for r in complete],.95),
            "tpot_p95_ms": percentile([r.get("tpot_ms") for r in complete],.95),
            "arrival_lag_p95_ms": percentile([r.get("arrival_lag_ms") for r in rows],.95)}


def trace_summary(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    steps = [r for r in rows if r["event"] == "step"]
    # A client abort may free the final request after the last model step.
    # The finish event, rather than the last step alone, owns the final state.
    states = [r for r in rows if "requests_after" in r]
    final = states[-1] if states else {}
    return {"steps": len(steps),
            "preemptions": sum(len(r.get("preempted",[])) for r in steps),
            "admission_denials": sum(r.get("admission_denied",0) for r in steps),
            "recomputed_tokens": sum(r.get("recomputed_tokens",0) for r in steps),
            "forced_steps": sum(r.get("forced_progress",False) for r in steps),
            "scheduler_p50_ms": percentile([r["scheduler_ms"] for r in steps],.5),
            "scheduler_p95_ms": percentile([r["scheduler_ms"] for r in steps],.95),
            "planner_p95_ms": percentile([r.get("planner_ms") for r in steps],.95),
            "free_blocks_at_end": final.get("free_blocks_after"),
            "requests_at_end": final.get("requests_after"),
            "ledger_at_end": final.get("ledger_after")}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("files",nargs="+")
    p.add_argument("--trace",action="store_true")
    a=p.parse_args()
    for f in a.files:
        result=trace_summary(f) if a.trace else summarize(json.loads(Path(f).read_text()))
        print(f, json.dumps(result,ensure_ascii=False,indent=2))
