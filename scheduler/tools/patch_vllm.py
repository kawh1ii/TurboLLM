#!/usr/bin/env python3
"""Apply only a known source hash. Idempotent, reversible, isolated-venv only."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "patches/manifest.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--restore", action="store_true")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
if sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to patch a base environment; use an isolated venv")
if importlib.metadata.version("vllm") != manifest["version"]:
    raise SystemExit("vLLM version does not match the audited manifest")
spec = importlib.util.find_spec("vllm")
target = Path(spec.origin).parent / "v1/core/sched/scheduler.py"
source = target.read_text()
digest = hashlib.sha256(source.encode()).hexdigest()
original, patched = manifest["original_sha256"], manifest["patched_sha256"]
if digest not in (original, patched):
    raise SystemExit("Unknown scheduler source hash: " + digest)
if args.check:
    print(json.dumps({"path": str(target), "patched": digest == patched,
                      "sha256": digest}))
    raise SystemExit(0)
desired = original if args.restore else patched
if digest == desired:
    print("Source already in requested state")
    raise SystemExit(0)
changes = reversed(manifest["changes"]) if args.restore else manifest["changes"]
for before, after in changes:
    if args.restore:
        before, after = after, before
    if source.count(before) != 1:
        raise SystemExit("Patch anchor mismatch")
    source = source.replace(before, after, 1)
assert hashlib.sha256(source.encode()).hexdigest() == desired
compile(source, str(target), "exec")
target.write_text(source)
print("Restored native source" if args.restore else "Installed three scheduling hooks")
