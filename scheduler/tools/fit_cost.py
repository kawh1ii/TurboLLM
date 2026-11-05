#!/usr/bin/env python3
"""Fit nonnegative costs on alternate baseline steps; report held-out error."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

p=argparse.ArgumentParser()
p.add_argument("trace")
p.add_argument("--output",required=True)
p.add_argument("--skip",type=int,default=32)
a=p.parse_args()
rows=[json.loads(x) for x in Path(a.trace).read_text().splitlines()]
rows=[r for r in rows if r["event"]=="step" and r.get("features")][a.skip:]
rows=[r for r in rows if 0<r["execution_ms"]<500]
if len(rows)<40:
    raise SystemExit("At least 40 post-warmup samples are required")
x=np.array([r["features"] for r in rows],dtype=float)
y=np.array([r["execution_ms"] for r in rows])
coeff,_=nnls(x[::2],y[::2])
pred=x[1::2]@coeff
error=np.abs(pred-y[1::2])
config={"coefficients":coeff.tolist()}
Path(a.output).write_text(json.dumps(config,indent=2)+"\n")
report={"features":["intercept","decode_requests","decode_context_1k","prefill_tokens","prefill_attention_1m"],
        "train_samples":len(x[::2]),"held_out_samples":len(pred),
        "held_out_mae_ms":float(error.mean()),
        "held_out_p95_abs_error_ms":float(np.quantile(error,.95)),
        "held_out_median_relative_error":float(np.median(error/y[1::2])),
        "rank":int(np.linalg.matrix_rank(x[::2])),"coefficients":coeff.tolist(),
        "warning":"Alternate steps are a diagnostic holdout, not independent workload generalization."}
Path(a.output+".fit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
