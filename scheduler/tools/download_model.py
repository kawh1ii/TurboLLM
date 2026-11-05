#!/usr/bin/env python3
"""Download official Qwen mirror files with size and SHA-256 verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

p=argparse.ArgumentParser()
p.add_argument("--model",default="Qwen/Qwen2.5-3B-Instruct")
p.add_argument("--output",required=True)
a=p.parse_args()
root=Path(a.output).resolve()
root.mkdir(parents=True,exist_ok=True)
url="https://modelscope.cn/api/v1/models/"+a.model+"/repo/files?Revision=master&Recursive=true"
with urllib.request.urlopen(url,timeout=30) as r:
    meta=json.load(r)
files=[f for f in meta["Data"]["Files"] if f["Type"]=="blob"
       and (f["Path"].endswith((".json",".safetensors",".txt")) or f["Path"]=="LICENSE")]
(root/"download-manifest.json").write_text(json.dumps({"model":a.model,"files":files},indent=2))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*2**20),b""):
            h.update(b)
    return h.hexdigest()


def download(f):
    target=(root/f["Path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("unsafe model path")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==f["Size"] and sha(target)==f["Sha256"]:
        print("VERIFIED",f["Path"],flush=True)
        return
    partial=Path(str(target)+".part")
    u="https://modelscope.cn/models/"+a.model+"/resolve/master/"+urllib.parse.quote(f["Path"])
    for attempt in range(5):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={"Range":f"bytes={offset}-"} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(u,headers=headers),timeout=90) as r:
                append=offset>0 and r.status==206
                with partial.open("ab" if append else "wb") as out:
                    for b in iter(lambda:r.read(8*2**20),b""):
                        out.write(b)
            if partial.stat().st_size!=f["Size"] or sha(partial)!=f["Sha256"]:
                partial.unlink(missing_ok=True)
                raise ValueError("model integrity mismatch")
            partial.replace(target)
            print("DOWNLOADED",f["Path"],f["Size"],flush=True)
            return
        except Exception as e:
            print("RETRY",f["Path"],attempt,type(e).__name__,str(e)[:120],flush=True)
            if attempt==4:
                raise
            time.sleep(min(2**attempt,8))


with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(download,files))
print("MODEL_VERIFIED",a.model,flush=True)
