import sys
from pathlib import Path
import unittest
import json
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze import summarize,percentile,trace_summary


class MetricsTests(unittest.TestCase):
    def test_abort_after_last_model_step_updates_final_state(self):
        records=[{"event":"step","scheduler_ms":.1,"free_blocks_after":8,
                  "requests_after":1},
                 {"event":"finish","free_blocks_after":9,"requests_after":0,"ledger_after":{}}]
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"trace.jsonl"
            f.write_text("\n".join(json.dumps(x) for x in records))
            result=trace_summary(f)
        self.assertEqual(result["requests_at_end"],0)
        self.assertEqual(result["free_blocks_at_end"],9)
        self.assertEqual(result["steps"],1)

    def test_failures_stay_in_slo_denominator(self):
        rows=[{"status":"completed","ttft_ms":100,"tpot_ms":10,"output_tokens":20},
              {"status":"completed","ttft_ms":1500,"tpot_ms":10,"output_tokens":20},
              {"status":"timeout"},{"status":"cancelled"}]
        result=summarize({"requests":rows,"elapsed_s":2})
        self.assertEqual(result["slo_pass_fraction_all_arrivals"],.25)
        self.assertEqual(result["goodput_requests_s"],.5)
        self.assertEqual(result["output_tokens_s"],20)

    def test_missing_tokens_cannot_pass_slo(self):
        r={"status":"completed","ttft_ms":10,"tpot_ms":None,"output_tokens":1}
        self.assertEqual(summarize({"requests":[r],"elapsed_s":1})["slo_pass"],0)

    def test_quantiles_with_empty_and_small_samples(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([10],.95),10)
        self.assertEqual(percentile([0,100],.95),95)


if __name__=="__main__":
    unittest.main()
