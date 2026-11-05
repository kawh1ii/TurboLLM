#!/usr/bin/env python3
"""Record only relevant environment metadata; never dump environment secrets."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch
import vllm

root=Path(__file__).resolve().parents[1]
out=root/"results"
out.mkdir(exist_ok=True)
torch.manual_seed(0)
x=torch.randn((128,128),device="cuda",dtype=torch.bfloat16)
y=x@x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
env={"recorded_utc":datetime.now(timezone.utc).isoformat(),
     "python":sys.version,"platform":platform.platform(),
     "torch":torch.__version__,"torch_cuda_runtime":torch.version.cuda,
     "vllm":vllm.__version__,"gpu":torch.cuda.get_device_name(0),
     "vram_bytes":torch.cuda.get_device_properties(0).total_memory,
     "compute_capability":torch.cuda.get_device_capability(0),
     "cpu_affinity_count":len(os.sched_getaffinity(0)),
     "nvidia_smi":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version,memory.total,name","--format=csv,noheader"],text=True).strip(),
     "bf16_matmul_smoke":True,
     "source_manifest":json.loads((root/"patches/manifest.json").read_text())}
env["source_manifest"].pop("changes")
nvcc=Path("/usr/local/cuda/bin/nvcc")
env["system_nvcc"] = subprocess.check_output([str(nvcc),"--version"],text=True).strip() if nvcc.exists() else None
packages={d.metadata["Name"]:d.version for d in metadata.distributions()}
env["packages"]=packages
(out/"environment.json").write_text(json.dumps(env,indent=2))
(root/"requirements.lock").write_text("\n".join(
    f"{k}=={v}" for k,v in sorted(packages.items(),key=lambda p:p[0].lower())
    if k.lower()!="turbollm-budget")+"\n")
print(json.dumps({k:v for k,v in env.items() if k not in ("packages","source_manifest")},indent=2))
