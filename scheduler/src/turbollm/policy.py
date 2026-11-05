"""Engine-independent policy. All clocks are milliseconds; lengths are tokens.

The planner never assumes the eventual output length is known. max_output is
only the user's hard cap, and the KV forecast uses a finite decode horizon.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class RequestView:
    request_id: str
    prompt: int
    computed: int
    total: int
    output: int
    max_output: int
    age_ms: float
    idle_ms: float
    running: bool
    held_blocks: int = 0

    @property
    def decode(self) -> bool:
        # A preempted request can have generated output AND still be replaying
        # it. Only the final pending token is ordinary decode; earlier history
        # must retain multi-token prefill/recompute treatment.
        return self.output > 0 and self.computed >= max(self.prompt, self.total - 1)

    @property
    def pending(self) -> int:
        return max(0, self.total - self.computed)


@dataclass(frozen=True)
class PolicyConfig:
    ttft_ms: float = 1000.0
    itl_ms: float = 40.0
    starvation_ms: float = 2000.0
    decode_horizon: int = 16
    min_chunk: int = 16
    max_chunk: int = 1024
    safety: float = 1.15
    enable_time_budget: bool = True
    enable_reservations: bool = True
    # Fixed intercept, decode batch, decode context/1k, prefill tokens,
    # and sum(prefill_tokens * (cached_context + tokens/2))/1e6.
    coefficients: tuple[float, ...] = (1.5, 0.05, 0.008, 0.015, 0.2)

    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.ttft_ms, self.itl_ms, self.starvation_ms)):
            raise ValueError("latency targets must be positive")
        if any(not isinstance(x, int) for x in
               (self.min_chunk, self.max_chunk, self.decode_horizon)):
            raise ValueError("token and horizon counts must be integers")
        if not 1 <= self.min_chunk <= self.max_chunk:
            raise ValueError("invalid chunk range")
        if self.decode_horizon < 0 or not math.isfinite(self.safety) or self.safety < 1:
            raise ValueError("invalid horizon or safety margin")
        if len(self.coefficients) != 5 or any(not math.isfinite(x) or x < 0 for x in self.coefficients):
            raise ValueError("cost model needs five nonnegative coefficients")
        object.__setattr__(self, "coefficients", tuple(self.coefficients))


def features(assignments: dict[str, int], requests: dict[str, RequestView]):
    decode_n = decode_context = prefill_n = prefill_attention = 0.0
    for rid, n in assignments.items():
        r = requests[rid]
        if r.decode:
            decode_n += 1
            decode_context += r.computed / 1024
        else:
            prefill_n += n
            prefill_attention += n * (r.computed + n / 2) / 1e6
    return (1.0, decode_n, decode_context, prefill_n, prefill_attention)


class StepCost:
    """Nonnegative model plus bounded online correction of execution wall time.

    This is NOT a GPU-kernel timer. IPC and model-runner overhead are included.
    """

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.scale = 1.0
        self.observations = 0

    def predict_features(self, x):
        return max(0.01, sum(a * b for a, b in zip(x, self.config.coefficients))) * self.scale

    def predict(self, assignments, requests):
        return self.predict_features(features(assignments, requests))

    def observe(self, x, elapsed_ms: float):
        base = sum(a * b for a, b in zip(x, self.config.coefficients))
        if base <= 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            return
        # Compilation / long descheduling outliers must not permanently inflate
        # the model. Offline fitting separately excludes warmup observations.
        ratio = min(8.0, max(0.25, elapsed_ms / base))
        self.scale = 0.95 * self.scale + 0.05 * ratio
        self.observations += 1


@dataclass
class Plan:
    quotas: dict[str, int] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    predicted_ms: float = 0.0
    forced_progress: bool = False


class Planner:
    def __init__(self, config: PolicyConfig, cost: StepCost | None = None):
        self.config = config
        self.cost = cost or StepCost(config)

    def urgency(self, r: RequestView):
        if r.decode:
            return (self.config.itl_ms - r.idle_ms) / self.config.itl_ms
        # Aging eventually outranks a decode even if TTFT is a loose target.
        return min((self.config.ttft_ms - r.age_ms) / self.config.ttft_ms,
                   1 - r.age_ms / self.config.starvation_ms)

    def plan(self, requests: list[RequestView], token_budget: int) -> Plan:
        if token_budget < 1:
            raise ValueError("token budget must be positive")
        candidates = sorted((r for r in requests if r.pending > 0),
                            key=lambda r: (self.urgency(r), -r.age_ms, r.request_id))
        by_id = {r.request_id: r for r in candidates}
        plan = Plan(order=[r.request_id for r in candidates])
        if not candidates:
            return plan
        # Every resident decode is offered one token first. At saturation the
        # oldest prefill gets a small progress grant; deadlines are soft targets.
        decodes = [r for r in candidates if r.decode and r.running]
        prefills = [r for r in candidates if not r.decode]
        oldest = max(prefills, key=lambda r: r.age_ms, default=None)
        forced = oldest is not None and oldest.age_ms >= self.config.starvation_ms
        grant = min(self.config.min_chunk, oldest.pending, token_budget) if forced else 0
        for r in decodes[:max(0, token_budget - grant)]:
            plan.quotas[r.request_id] = 1
        if forced and grant:
            plan.quotas[oldest.request_id] = grant
            plan.forced_progress = True
        remaining = token_budget - sum(plan.quotas.values())
        target_ms = self.config.itl_ms
        if decodes:
            # Do not collapse to a zero budget when an SLO is already missed.
            target_ms = max(1.0, min(self.config.itl_ms - r.idle_ms for r in decodes))
        for r in prefills:
            existing = plan.quotas.get(r.request_id, 0)
            limit = min(r.pending, self.config.max_chunk, existing + remaining)
            if limit <= existing:
                continue
            choices = {limit}
            n = self.config.min_chunk
            while n <= limit:
                choices.add(n)
                n *= 2
            best = existing
            for n in sorted(choices):
                if n <= existing:
                    continue
                trial = dict(plan.quotas, **{r.request_id: n})
                if (not self.config.enable_time_budget or
                        self.cost.predict(trial, by_id) * self.config.safety <= target_ms):
                    best = n
            if best > existing:
                plan.quotas[r.request_id] = best
                remaining -= best - existing
        if not plan.quotas:
            r = candidates[0]
            plan.quotas[r.request_id] = min(r.pending, self.config.min_chunk, token_budget)
            plan.forced_progress = True
        plan.predicted_ms = self.cost.predict(plan.quotas, by_id)
        assert 0 < sum(plan.quotas.values()) <= token_budget
        return plan


class ReservationLedger:
    """Future *unallocated* blocks, not a second owner of physical KV blocks.

    Native vLLM owns refcounts, prefix-cache sharing, allocation and free.
    The ledger reconciles from live requests and actual logical block tables.
    """

    def __init__(self, block_size: int, horizon: int, max_model_len: int):
        if block_size <= 0 or horizon < 0:
            raise ValueError("invalid reservation geometry")
        self.block_size = block_size
        self.horizon = horizon
        self.max_model_len = max_model_len
        self.obligations: dict[str, int] = {}

    def obligation(self, r: RequestView) -> int:
        future_output = min(max(0, r.max_output - r.output), self.horizon)
        # Replayed output is part of the prefill frontier after preemption.
        frontier = max(r.prompt, r.total - 1, r.computed) + future_output
        frontier = min(frontier, self.max_model_len)
        return max(0, math.ceil(frontier / self.block_size) - r.held_blocks)

    def reconcile(self, residents: list[RequestView]):
        self.obligations = {r.request_id: self.obligation(r) for r in residents}

    def reserved_for_others(self, request_id: str) -> int:
        return sum(n for rid, n in self.obligations.items() if rid != request_id)

    def release(self, request_id: str):
        self.obligations.pop(request_id, None)
