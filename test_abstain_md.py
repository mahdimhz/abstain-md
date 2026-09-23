"""Policy and interface tests. The real model is exercised separately."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from abstain_md import evaluate_case, load_data, main, retrieve


class ConstantNLI:
    def __init__(self, label="entailment"):
        self.label = label

    def predict(self, evidence, claim):
        return {"label": self.label, "scores": {}, "reason": "test double"}


class ConflictNLI:
    def predict(self, evidence, claim):
        label = "contradiction" if evidence.startswith("Antibiotics treat viral colds") else "entailment"
        return {"label": label, "scores": {}, "reason": "test double"}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence, cls.cases = load_data()
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_retrieval_uses_query_and_returns_nothing_off_topic(self):
        public_evidence = [item for item in self.evidence if not item["synthetic"]]
        ranked = retrieve("Do antibiotics treat viral colds?", public_evidence)
        self.assertEqual(ranked[0]["id"], "cdc_antibiotics_virus")
        self.assertEqual(retrieve("repair a bicycle tyre", public_evidence), [])

    def test_source_provenance_blocks_a_supporting_model(self):
        for case_id in ("jurisdiction_mismatch", "simulated_stale_source"):
            with self.subTest(case_id=case_id):
                result = evaluate_case(self.by_id[case_id], self.evidence, ConstantNLI())
                self.assertEqual(result["action"], "abstain")
                self.assertFalse(result["claim_checks"][0]["evidence"][0]["eligible"])

    def test_uncertain_model_and_conflicting_sources_block_answer(self):
        unsupported = evaluate_case(self.by_id["supported_antibiotics"],
                                    self.evidence, ConstantNLI("neutral"))
        self.assertEqual(unsupported["action"], "abstain")
        conflict = evaluate_case(self.by_id["synthetic_conflict"],
                                 self.evidence, ConflictNLI())
        self.assertEqual(conflict["claim_checks"][0]["verdict"], "conflict")
        self.assertEqual(conflict["action"], "abstain")

    def test_red_flag_precedes_failed_claim_verification(self):
        result = evaluate_case(self.by_id["chest_red_flag"],
                               self.evidence, ConstantNLI("contradiction"))
        self.assertEqual(result["action"], "escalate")
        self.assertEqual(result["claim_checks"][0]["verdict"], "contradicted")

    def test_missing_information_leads_to_clarification(self):
        result = evaluate_case(self.by_id["missing_information"],
                               self.evidence, ConstantNLI())
        self.assertEqual(result["action"], "clarify")

    def test_json_cli_is_machine_readable(self):
        output = io.StringIO()
        with patch("model_nli.LocalNLI", return_value=ConstantNLI()):
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--case", "supported_antibiotics", "--json"]), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["scores_are_calibrated"])
        self.assertEqual(payload["environment"]["execution_provider"], "CPUExecutionProvider")
        self.assertIn("cpu_model", payload["environment"])
        self.assertIn("onnxruntime", payload["environment"])
        self.assertEqual(payload["results"][0]["case_id"], "supported_antibiotics")
        self.assertEqual(payload["results"][0]["action"], "answer")


if __name__ == "__main__":
    unittest.main()
