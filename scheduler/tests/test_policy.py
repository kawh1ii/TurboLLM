from dataclasses import replace
import random
import unittest

from turbollm.policy import (
    Planner, PolicyConfig, RequestView, ReservationLedger, StepCost, features,
)


def req(rid="r", **kw):
    values = dict(prompt=256, computed=0, total=256, output=0,
                  max_output=128, age_ms=10, idle_ms=0, running=False)
    values.update(kw)
    return RequestView(rid, **values)


class PolicyTests(unittest.TestCase):
    def test_replayed_output_is_not_decode_until_frontier_caught_up(self):
        replay = req("r", prompt=256, computed=280, total=336, output=80, running=True)
        self.assertFalse(replay.decode)
        plan=Planner(PolicyConfig()).plan([replay],128)
        self.assertGreater(plan.quotas["r"],1)
        self.assertTrue(replace(replay,computed=335).decode)

    def test_decode_served_under_long_prefill(self):
        d = req("d", computed=300, total=301, output=45, running=True)
        p = req("p", prompt=8000, total=8000)
        plan = Planner(PolicyConfig()).plan([p, d], 512)
        self.assertEqual(plan.quotas["d"], 1)
        self.assertLessEqual(sum(plan.quotas.values()), 512)

    def test_aging_progress_under_saturation(self):
        decodes = [req(str(i), computed=300, total=301, output=45, running=True)
                   for i in range(16)]
        old = req("old", age_ms=3000)
        plan = Planner(PolicyConfig()).plan(decodes + [old], 16)
        self.assertGreater(plan.quotas["old"], 0)
        self.assertTrue(plan.forced_progress)
        self.assertEqual(sum(plan.quotas.values()), 16)

    def test_single_token_budget_makes_progress(self):
        plan = Planner(PolicyConfig()).plan([req(age_ms=5000)], 1)
        self.assertEqual(plan.quotas, {"r": 1})

    def test_no_work(self):
        self.assertEqual(Planner(PolicyConfig()).plan([], 128).quotas, {})

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            Planner(PolicyConfig()).plan([req()], 0)

    def test_determinism_and_input_immutability(self):
        items = [req("b"), req("a")]
        before = list(items)
        planner = Planner(PolicyConfig())
        self.assertEqual(planner.plan(items, 512), planner.plan(items, 512))
        self.assertEqual(items, before)

    def test_random_budget_and_length_invariants(self):
        rng = random.Random(43)
        for _ in range(500):
            items = []
            for i in range(rng.randrange(1, 48)):
                prompt = rng.randrange(1, 4096)
                output = rng.choice([0, 0, 32])
                computed = prompt + output - 1 if output else rng.randrange(prompt)
                items.append(req(str(i), prompt=prompt, computed=computed,
                                 total=prompt+output, output=output,
                                 running=output > 0 or rng.choice([True, False]),
                                 age_ms=rng.uniform(0, 5000), idle_ms=rng.uniform(0, 50)))
            budget = rng.randrange(1, 2048)
            plan = Planner(PolicyConfig()).plan(items, budget)
            by_id = {r.request_id: r for r in items}
            self.assertLessEqual(sum(plan.quotas.values()), budget)
            for rid, n in plan.quotas.items():
                self.assertGreater(n, 0)
                self.assertLessEqual(n, by_id[rid].pending)
                self.assertLessEqual(n, 1024)

    def test_context_cost_is_monotonic(self):
        cost = StepCost(PolicyConfig())
        short = req(computed=256, total=257, output=1, running=True)
        long = replace(short, computed=4096, total=4097)
        self.assertGreater(cost.predict({"r": 1}, {"r": long}),
                           cost.predict({"r": 1}, {"r": short}))

    def test_feedback_bounded_and_bad_samples_ignored(self):
        cost = StepCost(PolicyConfig())
        x = (1, 1, 0, 0, 0)
        for bad in [-1, float("nan"), float("inf")]:
            cost.observe(x, bad)
        self.assertEqual(cost.observations, 0)
        for _ in range(1000):
            cost.observe(x, 1000000)
        self.assertLessEqual(cost.scale, 8)

    def test_feature_vector_distinguishes_context_and_phase(self):
        d = req("d", computed=512, total=513, output=257, running=True)
        p = req("p", computed=128, total=256)
        self.assertEqual(features({"d": 1, "p": 64}, {"d": d, "p": p}),
                         (1.0, 1.0, 0.5, 64.0, 64*160/1e6))

    def test_invalid_config(self):
        for kw in [{"itl_ms": 0}, {"min_chunk": 100, "max_chunk": 20},
                   {"decode_horizon": -1}, {"coefficients": [-1]*5},
                   {"ttft_ms": float("nan")}, {"safety": float("inf")},
                   {"min_chunk": 16.5}, {"coefficients": [float("nan")]*5}]:
            with self.assertRaises(ValueError):
                PolicyConfig(**kw)


class KVTests(unittest.TestCase):
    def test_only_future_unallocated_blocks_reserved(self):
        ledger = ReservationLedger(16, 16, 4096)
        resident = req(computed=256, total=257, output=1, running=True, held_blocks=16)
        ledger.reconcile([resident])
        self.assertEqual(ledger.obligations, {"r": 1})
        self.assertEqual(ledger.reserved_for_others("r"), 0)
        self.assertEqual(ledger.reserved_for_others("new"), 1)

    def test_cancel_or_complete_removes_obligation(self):
        ledger = ReservationLedger(16, 16, 4096)
        ledger.reconcile([req("a"), req("b")])
        ledger.release("a")
        self.assertNotIn("a", ledger.obligations)
        ledger.reconcile([])
        self.assertEqual(ledger.reserved_for_others("new"), 0)

    def test_reconcile_allocation_and_preemption(self):
        ledger = ReservationLedger(16, 16, 4096)
        r = req(running=True, held_blocks=4)
        ledger.reconcile([r])
        before = ledger.obligations["r"]
        ledger.reconcile([replace(r, held_blocks=8)])
        self.assertEqual(ledger.obligations["r"], before-4)
        ledger.reconcile([])  # Preempted requests no longer reserve as residents.
        self.assertEqual(ledger.obligations, {})

    def test_prefix_sharing_does_not_reserve_existing_blocks_twice(self):
        ledger = ReservationLedger(16, 16, 4096)
        # 16 logical blocks already mapped in each request, possibly shared.
        ledger.reconcile([req("a", held_blocks=16), req("b", held_blocks=16)])
        self.assertEqual(ledger.reserved_for_others("new"), 2)

    def test_hard_output_cap_and_context_cap(self):
        ledger = ReservationLedger(16, 128, 256)
        r = req(output=128, max_output=128, total=384, computed=383, held_blocks=16)
        self.assertEqual(ledger.obligation(r), 0)


if __name__ == "__main__":
    unittest.main()
