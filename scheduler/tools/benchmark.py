#!/usr/bin/env python3
"""Open-loop SSE workload with replayable arrival times and per-request records.

Shape-controlled prompts are explicitly synthetic. No completion is dropped from
the report; rejected, cancelled and timed-out requests remain in the denominator.
"""
import argparse
import asyncio
import json
from pathlib import Path
import random
import time

import httpx


def make_workload(count, rate, seed, kind, long_prompt=1024):
    rng = random.Random(seed)
    items, arrival = [], 0.0
    shared = [rng.randrange(100, 20000) for _ in range(256)]
    for i in range(count):
        if i:
            arrival += rng.expovariate(rate) if rate > 0 else 0
        if kind == "burst":
            offset = (i // 16) * 1.5
        else:
            offset = arrival
        prompt_len = long_prompt if i % 4 == 0 else (128 if i % 4 == 1 else 384)
        output_len = 128 if i % 3 else 256
        if kind == "low":
            prompt_len, output_len = 128, 64
        prompt = [rng.randrange(100, 20000) for _ in range(prompt_len)]
        if kind == "prefix" and prompt_len >= 256:
            prompt[:256] = shared
        items.append({"id": str(i), "offset_s": offset, "prompt": prompt,
                      "output_tokens": output_len, "cancel": kind == "cancel" and i % 3 == 0})
    return items


async def request(client, base, item, origin, timeout):
    await asyncio.sleep(max(0, origin + item["offset_s"] - time.perf_counter()))
    start = time.perf_counter()
    result = {"id": item["id"], "offset_s": item["offset_s"],
              "prompt_tokens": len(item["prompt"]) if isinstance(item["prompt"], list) else None,
              "requested_output": item["output_tokens"], "status": "error",
              "arrival_lag_ms": (start-origin-item["offset_s"])*1000,
              "chunk_times_ms": [], "text": "", "usage": {}, "token_ids": []}
    body = {"model": "turbollm", "prompt": item["prompt"], "temperature": 0,
            "max_tokens": item["output_tokens"], "min_tokens": item["output_tokens"],
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "seed": 123}
    try:
        seen_done = False
        async with asyncio.timeout(timeout):
            async with client.stream("POST", base + "/v1/completions", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line == "data: [DONE]":
                        seen_done = True
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise RuntimeError(str(data["error"]))
                    if data.get("usage"):
                        result["usage"] = data["usage"]
                    for choice in data.get("choices", []):
                        text = choice.get("text", "")
                        ids = choice.get("token_ids") or []
                        # A network chunk may contain >1 token. Record actual
                        # IDs/counts instead of pretending each SSE event is one token.
                        if text or ids:
                            result["chunk_times_ms"].append((time.perf_counter()-start)*1000)
                            result["text"] += text
                            result["token_ids"].extend(ids)
                            if item.get("cancel") and len(result["chunk_times_ms"]) >= 4:
                                result["status"] = "cancelled"
                                return result
                if not seen_done or len(result["token_ids"]) != item["output_tokens"]:
                    raise RuntimeError("Incomplete SSE response or missing token IDs")
                result["status"] = "completed"
    except TimeoutError:
        result["status"] = "timeout"
    except Exception as e:
        result["error"] = type(e).__name__ + ": " + str(e)[:250]
    finally:
        result["latency_ms"] = (time.perf_counter()-start)*1000
    times = result["chunk_times_ms"]
    result["ttft_ms"] = times[0] if times else None
    n = result["usage"].get("completion_tokens", len(result["token_ids"]))
    result["output_tokens"] = n
    result["tpot_ms"] = (times[-1]-times[0])/(n-1) if len(times) > 1 and n > 1 else None
    result["max_chunk_gap_ms"] = max((b-a for a,b in zip(times,times[1:])), default=None)
    return result


async def main(a):
    items = json.loads(Path(a.workload).read_text()) if a.workload else make_workload(a.count, a.rate, a.seed, a.kind, a.long_prompt)
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(str(target)+".workload.json").write_text(json.dumps(items))
    limits = httpx.Limits(max_connections=max(64, len(items)), max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=None, limits=limits, trust_env=False) as client:
        # Warmup is explicit and is not part of arrivals or summary.
        warm = {"id": "warm", "offset_s": 0, "prompt": [100, 101]*64, "output_tokens": 16}
        w = await request(client, a.base, warm, time.perf_counter(), a.timeout)
        if w["status"] != "completed":
            raise RuntimeError("Warmup failed: " + str(w))
        origin = time.perf_counter()
        rows = await asyncio.gather(*(request(client,a.base,item,origin,a.timeout) for item in items))
        elapsed = time.perf_counter()-origin
    meta = {"seed": a.seed, "kind": a.kind, "rate": a.rate,
            "elapsed_s": elapsed, "count": len(items), "requests": rows}
    target.write_text(json.dumps(meta, indent=2))
    from analyze import summarize
    summary = summarize(meta, a.ttft_ms, a.tpot_ms)
    Path(str(target)+".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--output", required=True)
    p.add_argument("--workload")
    p.add_argument("--kind", choices=["mixed","burst","prefix","low","cancel"], default="mixed")
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--rate", type=float, default=8)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--long-prompt", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--ttft-ms", type=float, default=1000)
    p.add_argument("--tpot-ms", type=float, default=40)
    asyncio.run(main(p.parse_args()))
