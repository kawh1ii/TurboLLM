"""Compare a frozen TurboLLM policy with its patched copy on identical CPU inputs."""
import argparse
from dataclasses import astuple
import hashlib
import importlib.util
import json
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns


def load_policy(name, root):
    source = root / "src/turbollm/policy.py"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.read_bytes()).hexdigest()


def workload(rng, count, mode):
    rows = []
    for i in range(count):
        prompt = rng.choice([128, 384, 1024, 4096, 8192])
        phase = rng.choice(["prefill", "decode", "replay"]) if mode == "mixed" else "prefill"
        output = 0 if phase == "prefill" else rng.randrange(2, 97)
        total = prompt + output
        if phase == "decode":
            computed = total - 1
        elif phase == "replay":
            computed = rng.randrange(prompt, total - 1)
        else:
            computed = rng.randrange(prompt)
        age = rng.choice([0, 2000, 2001, 5000, rng.uniform(0, 6000)])
        idle = rng.uniform(0, 60 if mode == "mixed" else 20)
        rows.append((str(i), prompt, computed, total, output, 128, age, idle,
                     phase == "decode" or rng.choice([True, False]), 0))
    rng.shuffle(rows)
    return rows


def inputs(modules, rows):
    return [[module.RequestView(*row) for row in rows] for module in modules]


def planners(modules, config, scale):
    result = [module.Planner(module.PolicyConfig(**config)) for module in modules]
    for planner in result:
        planner.cost.scale = scale
    return result


def check(pair, data, budget):
    left, right = (planner.plan(rows, budget) for planner, rows in zip(pair, data))
    if astuple(left) != astuple(right) or list(left.quotas) != list(right.quotas):
        raise AssertionError(f"Different decision for budget {budget}: {left} vs {right}")


def quantiles(samples):
    return {
        "median_us": statistics.median(samples) / 1000,
        "p95_us": statistics.quantiles(samples, n=100, method="inclusive")[94] / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_sha = load_policy("baseline_policy", args.baseline)
    candidate, candidate_sha = load_policy("candidate_policy", args.candidate)
    modules = (baseline, candidate)
    frozen = json.loads((args.baseline / "results/final/frozen-policy.json").read_text())
    rng = random.Random(113)
    differential_cases = 10000
    for _ in range(differential_cases):
        rows = workload(rng, rng.randrange(0, 129), "mixed")
        cfg = {
            "coefficients": tuple(rng.uniform(0, limit) for limit in (16, .1, .4, .08, .5)),
            "min_chunk": rng.choice([1, 4, 16, 32]),
            "max_chunk": rng.choice([64, 128, 1024]),
            "itl_ms": rng.choice([5, 40, 150]),
            "enable_time_budget": rng.choice([True, True, False]),
        }
        check(planners(modules, cfg, rng.uniform(.25, 8)), inputs(modules, rows),
              rng.randrange(1, 8193))

    pair = planners(modules, frozen, 1.0)
    groups = []
    for mode in ("mixed", "prefill"):
        for count in (16, 64, 256, 512):
            cases = [inputs(modules, workload(rng, count, mode)) for _ in range(256)]
            for data in cases:
                check(pair, data, 4096)
            samples = [[], []]
            for repeat in range(3):
                for i, data in enumerate(cases):
                    order = (0, 1) if (i + repeat) % 2 == 0 else (1, 0)
                    for j in order:
                        start = perf_counter_ns()
                        pair[j].plan(data[j], 4096)
                        samples[j].append(perf_counter_ns() - start)
            groups.append({
                "mode": mode, "requests": count, "unique_cases": len(cases),
                "baseline": quantiles(samples[0]), "incremental": quantiles(samples[1]),
                "baseline_ns": samples[0], "incremental_ns": samples[1],
            })
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip() if sys.platform == "darwin" else platform.processor()
    result = {
        "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
        "baseline_sha256": baseline_sha, "candidate_sha256": candidate_sha,
        "seed": 113, "random_differential_cases": differential_cases,
        "timing_differential_cases": sum(group["unique_cases"] for group in groups),
        "matching_fields": ["quota values and insertion order", "request order", "predicted_ms", "forced_progress"],
        "timing_token_budget": 4096, "timing_cost_config": frozen, "groups": groups,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
