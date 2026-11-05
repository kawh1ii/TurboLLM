#!/usr/bin/env python3
"""Save / compare actual token IDs; exit nonzero on any discrepancy."""
import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def main(a):
    prompts = [[100+(i%200) for i in range(n)] for n in (16,17,31,32,127,128,255,256,511,512)]
    prompts += [[1000+(i%500) for i in range(1536)]]
    prompts += ["Explain why a KV cache saves work during autoregressive decoding.",
                "Write a short Python function that computes the greatest common divisor."]
    records=[]
    async with httpx.AsyncClient(timeout=120,trust_env=False) as client:
        async def one(i,prompt):
            response=await client.post(a.base+"/v1/completions",json={
                "model":"turbollm","prompt":prompt,"temperature":0,"seed":123,
                "max_tokens":32,"min_tokens":32,"ignore_eos":True,
                "return_token_ids":True})
            response.raise_for_status()
            data=response.json()
            choice=data["choices"][0]
            ids=choice.get("token_ids")
            if not ids or len(ids)!=32:
                raise RuntimeError("Missing/short output token IDs")
            return {"id":str(i),"response_id":data["id"],"token_ids":ids,"text":choice["text"]}
        for i,prompt in enumerate(prompts):
            records.append(await one(i,prompt))
        records.extend(await asyncio.gather(*(one("batch-"+str(i),prompt)
                                              for i,prompt in enumerate(prompts[-4:]))))
    result={"requests":records}
    if a.reference:
        reference=json.loads(Path(a.reference).read_text())["requests"]
        if len(reference)!=len(records):
            raise RuntimeError("reference size differs")
        matches=[x["token_ids"]==y["token_ids"] for x,y in zip(records,reference)]
        result.update({"exact_matches":sum(matches),"total":len(matches)})
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="requests"}))
    if a.reference and result["exact_matches"]!=result["total"]:
        raise SystemExit("Token mismatch: inspect output before claiming correctness")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="http://127.0.0.1:8000")
    p.add_argument("--output",required=True)
    p.add_argument("--reference")
    asyncio.run(main(p.parse_args()))
