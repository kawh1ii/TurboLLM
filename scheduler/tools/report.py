#!/usr/bin/env python3
"""Recompute tables from raw request files; no hand-entered performance values."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze import summarize,trace_summary

p=argparse.ArgumentParser()
p.add_argument("--root",default="results")
a=p.parse_args()
root=Path(a.root)
groups=defaultdict(list)
for f in sorted(root.rglob("*.json")):
    if any(x in f.name for x in [".summary.",".workload."]):
        continue
    try:
        d=json.loads(f.read_text())
    except (ValueError,UnicodeError):
        continue
    if "elapsed_s" not in d or "requests" not in d:
        continue
    # Mode names may contain hyphens, so split at the known workload suffix.
    mode=f.name.split("-"+d["kind"]+"-s")[0]
    groups[(str(f.parent.relative_to(root)),d["kind"],mode)].append((f,d,summarize(d)))

rows=[]
lines=["# 实验结果（从原始请求记录复算）","",
       "以下为各次独立运行的中位数。合成负载、有限模型与单卡结果不代表线上业务分布；三次运行不构成统计显著性证明。",
       "","|实验|负载|策略|次数|完成/到达|SLO通过/到达|goodput 请求/s|输出 token/s|TTFT P95 ms|TPOT P95 ms|",
       "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
def med(entries,key):
    values=[r[2][key] for r in entries if r[2][key] is not None]
    return statistics.median(values) if values else None
def fmt(x):
    return "—" if x is None else f"{x:.3f}"
for (experiment,kind,mode),entries in sorted(groups.items()):
    row={"experiment":experiment,"kind":kind,"mode":mode,"replicates":len(entries),
         "completed":sum(x[2]["completed"] for x in entries),
         "arrivals":sum(x[2]["total_arrivals"] for x in entries),
         "slo_pass":sum(x[2]["slo_pass"] for x in entries),
         **{key:med(entries,key) for key in ["goodput_requests_s","output_tokens_s","ttft_p95_ms","tpot_p95_ms"]},
         "files":[str(x[0].relative_to(root)) for x in entries]}
    rows.append(row)
    lines.append(f"|{experiment}|{kind}|{mode}|{len(entries)}|{row['completed']}/{row['arrivals']}|"
                 f"{row['slo_pass']}/{row['arrivals']}|{fmt(row['goodput_requests_s'])}|"
                 f"{fmt(row['output_tokens_s'])}|{fmt(row['ttft_p95_ms'])}|{fmt(row['tpot_p95_ms'])}|")
traces={str(f.relative_to(root)):trace_summary(f) for f in sorted(root.rglob("*.trace.jsonl"))}
(root/"summary.json").write_text(json.dumps({"experiments":rows,"traces":traces},indent=2))
lines += ["","## 机制计数","", "详见 summary.json 的 traces 字段：抢占、重计算、准入拒绝、调度开销与排空后的资源状态。",
          "","## 复算入口","", "`python tools/report.py --root results`。每个请求级 JSON 同目录保存输入工作负载和摘要；源码、配置和服务端日志可对应核查。"]
(root/"REPORT.md").write_text("\n".join(lines)+"\n")
manifest={}
for f in sorted(root.rglob("*")):
    if f.is_file() and f.name!="evidence-manifest.json":
        manifest[str(f.relative_to(root))]={"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
(root/"evidence-manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(rows,indent=2))
