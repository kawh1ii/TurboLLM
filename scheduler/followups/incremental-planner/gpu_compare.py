"""Paired vLLM tests for the baseline and incremental planners on one GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, nargs="+", default=[123, 124, 125])
    p.add_argument("--kv-blocks", type=int, default=256)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--port", type=int, default=8210)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument("--repeat-baseline", action="store_true",
                   help="Run the original implementation twice as a numerical control")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    tools = baseline / "tools"
    sys.path.insert(0, str(tools))
    from analyze import summarize, trace_summary
    from benchmark import make_workload
    from run_matrix import stop, wait_ready

    other = "repeat" if args.repeat_baseline else "incremental"
    roots = {"original": baseline, other: baseline if args.repeat_baseline else candidate}
    sources = {name: {str(f.relative_to(root)): digest(f)
                      for f in sorted((root / "src").rglob("*.py"))}
               for name, root in roots.items()}
    changed = {key for key in sources["original"].keys() | sources[other].keys()
               if sources["original"].get(key) != sources[other].get(key)}
    expected = set() if args.repeat_baseline else {"src/turbollm/policy.py"}
    if changed != expected:
        raise ValueError(f"Unexpected runtime source differences: {changed}")
    subprocess.run([sys.executable, str(tools / "patch_vllm.py")], check=True)
    patch_state = json.loads(subprocess.check_output(
        [sys.executable, str(tools / "patch_vllm.py"), "--check"], text=True))
    if not patch_state["patched"]:
        raise RuntimeError("Scheduler hooks were not applied")

    config = baseline / "results/final/frozen-policy.json"
    result = {
        "seeds": args.seed, "kv_blocks": args.kv_blocks, "eager": args.eager,
        "trace": args.trace, "batch_invariant": args.batch_invariant,
        "repeat_baseline": args.repeat_baseline,
        "vllm": importlib.metadata.version("vllm"), "torch": importlib.metadata.version("torch"),
        "model_manifest_sha256": digest(args.model / "download-manifest.json"),
        "source_hashes": sources, "upstream_patch_state": patch_state,
        "policy_config": json.loads(config.read_text()), "runs": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    result["output_comparisons"] = []
    for index, seed in enumerate(args.seed):
        kind = "burst" if args.kv_blocks else "mixed"
        workload = args.output / f"workload-s{seed}.json"
        workload.write_text(json.dumps(make_workload(48, 8, seed, kind)))
        order = list(roots) if index % 2 == 0 else list(reversed(roots))
        for variant in order:
            prefix = args.output / f"{variant}-s{seed}"
            trace = prefix.with_suffix(".trace.jsonl")
            responses = prefix.with_suffix(".json")
            command = [
                sys.executable, str(tools / "serve.py"), "--model", str(args.model),
                "--mode", "budget", "--config", str(config), "--port", str(args.port),
                "--max-model-len", "2048", "--max-num-seqs", "32",
                "--max-num-batched-tokens", "2048",
            ]
            if args.kv_blocks:
                command += ["--kv-blocks", str(args.kv_blocks)]
            if args.eager:
                command.append("--eager")
            if args.batch_invariant:
                command.append("--batch-invariant")
            if args.trace:
                command += ["--trace", str(trace)]
            env = dict(os.environ, PYTHONPATH=str(roots[variant] / "src"))
            print(f"RUN {variant} seed={seed}", flush=True)
            with prefix.with_suffix(".server.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    wait_ready(process, args.port, timeout=300)
                    subprocess.run([
                        sys.executable, str(tools / "benchmark.py"), "--base", f"http://127.0.0.1:{args.port}",
                        "--workload", str(workload), "--output", str(responses), "--kind", kind,
                        "--seed", str(seed),
                    ], check=True)
                finally:
                    stop(process)
            if Path(str(responses) + ".workload.json").read_bytes() != workload.read_bytes():
                raise ValueError("Replayed workload differs")
            metrics = summarize(json.loads(responses.read_text()))
            if metrics["completed"] != 48:
                raise RuntimeError(f"Incomplete workload: {metrics['statuses']}")
            row = {"variant": variant, "seed": seed, "workload_sha256": digest(workload),
                   "request_metrics": metrics}
            if args.trace:
                row["trace_metrics"] = trace_summary(trace)
                row["trace_verification"] = json.loads(subprocess.check_output([
                    sys.executable, str(tools / "check_trace.py"), str(trace)
                ], text=True))
            result["runs"].append(row)
            save()
        output_ids = []
        for variant in roots:
            rows = json.loads((args.output / f"{variant}-s{seed}.json").read_text())["requests"]
            output_ids.append({r["id"]: r["token_ids"] for r in rows})
        if output_ids[0].keys() != output_ids[1].keys():
            raise ValueError("Mismatched request identifiers")
        matches = sum(output_ids[0][rid] == output_ids[1][rid] for rid in output_ids[0])
        result["output_comparisons"].append({"seed": seed, "identical": matches, "total": 48})
        save()
        if args.batch_invariant and matches != 48:
            raise RuntimeError("Strict output ID comparison failed")
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
