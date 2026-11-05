"""Version-locked adapter; native vLLM retains allocation and request ownership.

Three small audited hooks are installed by tools/patch_vllm.py. No startup
sitecustomize, process-global monkeypatch, or replacement attention kernel.
Only synchronous, single-GPU, single-KV-group, decoder-only text is supported.
"""

from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import time

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.kv_cache_interface import FullAttentionSpec

from turbollm.policy import (
    Planner, PolicyConfig, RequestView, ReservationLedger, StepCost, features,
)


class TraceScheduler(Scheduler):
    """Unchanged native scheduling, with identical timing instrumentation."""

    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("vllm") != "0.32.0":
            raise RuntimeError("Adapter is audited only against vLLM 0.32.0")
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("synchronous scheduling required for paired step timing")
        if self.parallel_config.world_size != 1 or self.connector is not None:
            raise ValueError("only single-GPU, local KV is supported")
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("only one full-attention KV group is supported")
        if not isinstance(self.kv_cache_config.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec):
            raise ValueError("only full-attention KV geometry has been audited")
        if self.vllm_config.speculative_config or self.is_encoder_decoder:
            raise ValueError("speculative / encoder-decoder models are unsupported")
        if self.vllm_config.model_config.is_multimodal_model:
            raise ValueError("multimodal models are unsupported")
        if self.scheduler_config.policy != "fcfs" or self.lora_config is not None:
            raise ValueError("adapter assumes FCFS requests without LoRA")
        self._last_progress = {}
        self._pending_samples = {}
        self._step_no = 0
        self._trace = None
        path = os.environ.get("TURBOLLM_TRACE")
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._trace = target.open("a", buffering=1)
            self._emit({"event": "init", "scheduler": type(self).__name__,
                        "vllm": "0.32.0", "pid": os.getpid(),
                        "kv_blocks": self.kv_cache_config.num_blocks,
                        "free_blocks_initial": self.kv_cache_manager.block_pool.get_num_free_blocks(),
                        "block_size": self.block_size,
                        "max_tokens": self.max_num_scheduled_tokens})

    def _emit(self, record):
        if self._trace:
            self._trace.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _view(self, request, running, now=None):
        now = time.monotonic() if now is None else now
        blocks = self.kv_cache_manager.get_blocks(request.request_id).blocks
        held = len(blocks[0]) if blocks else 0
        return RequestView(request.request_id, request.num_prompt_tokens,
                           request.num_computed_tokens, request.num_tokens,
                           request.num_output_tokens, request.max_tokens,
                           max(0, (time.time() - request.arrival_time) * 1000),
                           max(0, (now - self._last_progress.get(request.request_id, now)) * 1000),
                           running, held)

    def schedule(self, throttle_prefills=False):
        started = time.perf_counter()
        views = {rid: self._view(r, r in self.running)
                 for rid, r in self.requests.items()}
        output = super().schedule(throttle_prefills=throttle_prefills)
        self._step_no += 1
        assigned = dict(output.num_scheduled_tokens)
        # Native prefix lookup occurs inside schedule(). After its optimistic
        # advance, subtract this step's allocation to recover the actual cached
        # context. This avoids a second lookup that would distort cache metrics.
        for rid, n in assigned.items():
            if rid in self.requests:
                views[rid] = replace(views[rid], computed=max(
                    0, self.requests[rid].num_computed_tokens - n))
        x = features(assigned, views) if assigned else None
        self._pending_samples[id(output)] = {
            "event": "step", "step": self._step_no, "features": x,
            "scheduled": assigned, "scheduler_ms": (time.perf_counter()-started)*1000,
            "free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks(),
            "waiting": len(self.waiting), "running": len(self.running),
            "preempted": list(output.preempted_req_ids or []),
            "recomputed_tokens": sum(min(n, max(0, views[rid].total-1-views[rid].computed))
                                     for rid,n in assigned.items() if views[rid].output > 0),
            "execution_start": time.perf_counter(),
            "outputs_before": {rid: self.requests[rid].num_output_tokens
                               for rid in assigned if rid in self.requests},
        }
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        sample = self._pending_samples.pop(id(scheduler_output), None)
        if sample:
            sample["execution_ms"] = (time.perf_counter()-sample.pop("execution_start"))*1000
        result = super().update_from_output(scheduler_output, model_runner_output)
        if sample:
            now = time.monotonic()
            for rid, old in sample.pop("outputs_before").items():
                request = self.requests.get(rid)
                if request is None or request.num_output_tokens > old:
                    self._last_progress[rid] = now
            self._last_progress = {rid: t for rid, t in self._last_progress.items()
                                   if rid in self.requests}
            if sample["features"]:
                self._observe(sample)
            sample["free_blocks_after"] = self.kv_cache_manager.block_pool.get_num_free_blocks()
            sample["requests_after"] = len(self.requests)
            if hasattr(self, "ledger"):
                self.ledger.reconcile([self._view(r, True) for r in self.running])
                sample["ledger_after"] = dict(self.ledger.obligations)
            self._emit(sample)
        return result

    def _observe(self, sample):
        pass

    def finish_requests(self, *args, **kwargs):
        result = super().finish_requests(*args, **kwargs)
        record = {"event": "finish", "requests_after": len(self.requests),
                  "free_blocks_after": self.kv_cache_manager.block_pool.get_num_free_blocks()}
        if hasattr(self, "ledger"):
            self.ledger.reconcile([self._view(r, True) for r in self.running])
            record["ledger_after"] = dict(self.ledger.obligations)
        self._emit(record)
        return result


class BudgetScheduler(TraceScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw = json.loads(Path(os.environ["TURBOLLM_CONFIG"]).read_text()) if os.environ.get("TURBOLLM_CONFIG") else {}
        self.config = PolicyConfig(**raw)
        self.cost = StepCost(self.config)
        self.planner = Planner(self.config, self.cost)
        self.ledger = ReservationLedger(self.block_size, self.config.decode_horizon,
                                        self.max_model_len)
        self._plan = None
        self._admission_denied = 0
        self._last_plan_record = None
        self._emit({"event": "policy", "config": asdict(self.config)})
        # Fail closed if someone installed the package but omitted its hooks.
        import inspect
        if "_turbollm_cap" not in inspect.getsource(Scheduler.schedule):
            raise RuntimeError("Run tools/patch_vllm.py in this isolated environment")

    def schedule(self, throttle_prefills=False):
        t0 = time.perf_counter()
        candidates = [self._view(r, True) for r in self.running]
        candidates += [self._view(r, False) for r in self.waiting
                       if r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)]
        self._plan = self.planner.plan(candidates, self.max_num_scheduled_tokens)
        rank = {rid: i for i, rid in enumerate(self._plan.order)}
        # Preserve native RUNNING order: the FCFS preemption path evicts its
        # final entry. Reordering execution here silently changes victim choice.
        # Quotas sum to <= the native budget, so a resident prefill cannot
        # consume tokens promised to a later decode even without this sort.
        queued = list(self.waiting)
        self.waiting.remove_requests(queued)
        for r in sorted(queued, key=lambda r: rank.get(r.request_id, 10**9)):
            self.waiting.add_request(r)
        self._admission_denied = 0
        self._last_plan_record = {"quotas": self._plan.quotas,
                                  "predicted_ms": self._plan.predicted_ms,
                                  "forced_progress": self._plan.forced_progress,
                                  "planner_ms": (time.perf_counter()-t0)*1000}
        output = super().schedule(throttle_prefills=throttle_prefills)
        record = self._pending_samples.get(id(output))
        if record:
            record.update(self._last_plan_record)
            record["scheduler_ms"] += record["planner_ms"]
            record["admission_denied"] = self._admission_denied
            record["reserved_blocks"] = sum(self.ledger.obligations.values())
        return output

    def _turbollm_cap(self, request, native_tokens):
        return min(native_tokens, self._plan.quotas.get(request.request_id, 0))

    def _turbollm_admit(self, request, num_new_computed_tokens, new_computed_blocks):
        if not self.config.enable_reservations:
            return True
        # Reconcile every attempt: earlier admissions/allocations in this SAME
        # step change both residents and block tables. Stale step-start totals
        # would double reserve blocks already allocated by earlier requests.
        self.ledger.reconcile([self._view(r, True) for r in self.running])
        manager = self.kv_cache_manager
        other_reservations = self.ledger.reserved_for_others(request.request_id)
        future = min(self.config.decode_horizon,
                     max(0, request.max_tokens - request.num_output_tokens))
        # With no residents, do not strand a request whose prompt fits but whose
        # optional forecast does not. Native current-step checks still apply.
        if not self.running:
            future = 0
        frontier = min(self.max_model_len, request.num_tokens + future)
        cached = (new_computed_blocks.blocks if new_computed_blocks is not None
                  else manager.empty_kv_cache_blocks.blocks)
        needed = manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=frontier,
            new_computed_blocks=cached, num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens + num_new_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens + num_new_computed_tokens,
            num_tokens_main_model=frontier, apply_admission_cap=True,
            prefill_end=max(request.num_prompt_tokens, request.num_tokens-1))
        free = manager.block_pool.get_num_free_blocks()
        allowed = needed + other_reservations + (manager.watermark_blocks if self.running else 0) <= free
        if not allowed:
            self._admission_denied += 1
        return allowed

    def _turbollm_reserved_blocks(self, request):
        if not self.config.enable_reservations:
            return 0
        self.ledger.reconcile([self._view(r, True) for r in self.running])
        return self.ledger.reserved_for_others(request.request_id)

    def _observe(self, sample):
        self.cost.observe(sample["features"], sample["execution_ms"])
        sample["cost_scale"] = self.cost.scale

    def _free_request(self, request, *args, **kwargs):
        # Called for completion AND abort; no second allocator/refcount owner.
        result = super()._free_request(request, *args, **kwargs)
        if hasattr(self, "ledger"):
            self.ledger.release(request.request_id)
        return result

    def _preempt_request(self, request, *args, **kwargs):
        result = super()._preempt_request(request, *args, **kwargs)
        self.ledger.release(request.request_id)
        return result
