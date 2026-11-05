#!/usr/bin/env python3
"""Verify executable-source identity and the historical metadata-inclusive hash.

The earliest macOS tar transfer also carried three AppleDouble sidecars. They
were never imported, but the initial fingerprint glob included their .py names.
The archived bytes make that historical fingerprint independently reproducible;
new fingerprints exclude these non-code files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from full_validation import ROOT,execution_hash

p=argparse.ArgumentParser()
p.add_argument("--results",default="results/final")
a=p.parse_args()
results=Path(a.results)
paths=[p for p in (ROOT/"src").rglob("*.py") if not p.name.startswith("._")]
paths += [ROOT/"tools"/(s+".py") for s in
          ["serve","benchmark","correctness","fit_cost","run_matrix"]]
contents={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
code_hashes={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()}
sidecars={}
with zipfile.ZipFile(results/"legacy-fingerprint-sidecars.zip") as z:
    for name in z.namelist():
        assert name.startswith("src/") and Path(name).name.startswith("._")
        assert ".." not in Path(name).parts
        data=z.read(name)
        assert data.startswith(bytes.fromhex("00051607")),"not AppleDouble metadata"
        sidecars[name]=data
legacy=hashlib.sha256()
for name,data in sorted({**contents,**sidecars}.items()):
    legacy.update(name.encode());legacy.update(data)
legacy_hash=legacy.hexdigest()
stages=list(results.glob("*/.done.json"))
assert len(stages)==14,len(stages)
assert all(json.loads(f.read_text())["execution_sha256"]==legacy_hash for f in stages)
audit={"portable_sha256":execution_hash(),"legacy_sha256":legacy_hash,
       "verified_stages":len(stages),"executable_python_files":code_hashes,
       "excluded_non_executable_metadata":{k:{"bytes":len(v),"sha256":hashlib.sha256(v).hexdigest()}
                                           for k,v in sidecars.items()},
       "explanation":"All executable Python files match the tested code. The historical digest also included non-executable AppleDouble sidecars; their archived bytes reproduce it exactly."}
(results/"source-hash-audit.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
