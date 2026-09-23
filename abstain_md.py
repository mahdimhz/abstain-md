"""Evidence retrieval, local NLI verification, and an auditable action policy.

Synthetic research demonstrator: no patient messages, generated answers, or medical advice.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent
STOP_WORDS = {"a", "an", "and", "are", "be", "can", "do", "does", "for",
              "how", "i", "in", "is", "of", "on", "the", "to", "with"}


class NLIModel(Protocol):
    def predict(self, evidence: str, claim: str) -> dict: ...


def load_data() -> tuple[list[dict], list[dict]]:
    with (ROOT / "evidence.json").open(encoding="utf-8") as file:
        evidence = json.load(file)
    with (ROOT / "cases.json").open(encoding="utf-8") as file:
        cases = json.load(file)
    return evidence, cases


def terms(value: str) -> set[str]:
    words = re.findall(r"[a-z]+", value.lower())
    return {word[:-1] if word.endswith("s") and not word.endswith("ss") else word
            for word in words if word not in STOP_WORDS}


def retrieve(query: str, evidence: list[dict], top_k: int = 1) -> list[dict]:
    """Rank local authored passages by transparent token overlap."""
    query_terms = terms(query)
    ranked = []
    for record in evidence:
        document_terms = terms(f"{record['title']} {record['passage']}")
        overlap = len(query_terms & document_terms)
        if overlap:
            ranked.append((overlap, record["id"], record))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{**record, "retrieval_overlap": score} for score, _, record in ranked[:top_k]]


def verify_claim(claim: str, retrieved: list[dict], jurisdiction: str,
                 overrides: dict[str, str], nli: NLIModel) -> dict:
    evidence_checks = []
    eligible_labels = set()
    for record in retrieved:
        status = overrides.get(record["id"], record["status"])
        model_result = nli.predict(record["passage"], claim)
        eligible = status == "approved" and record["jurisdiction"] == jurisdiction
        if eligible:
            eligible_labels.add(model_result["label"])
        evidence_checks.append({
            "source_id": record["id"],
            "source_url": record["url"],
            "jurisdiction": record["jurisdiction"],
            "snapshot_date": record["snapshot_date"],
            "source_date": record["source_date"],
            "status": status,
            "synthetic_source": record["synthetic"],
            "retrieval_overlap": record["retrieval_overlap"],
            "eligible": eligible,
            "nli": model_result,
        })
    if "entailment" in eligible_labels and "contradiction" in eligible_labels:
        verdict = "conflict"
    elif "entailment" in eligible_labels:
        verdict = "supported"
    elif "contradiction" in eligible_labels:
        verdict = "contradicted"
    else:
        verdict = "insufficient"
    return {"claim": claim, "verdict": verdict, "evidence": evidence_checks}


def choose_action(case: dict, claim_checks: list[dict],
                  retrieved: list[dict]) -> tuple[str, str]:
    tags = set(case.get("tags", []))
    if {"chest_pain", "shortness_of_breath"} <= tags:
        return "escalate", "Synthetic emergency red-flag rule fired"
    if {"warfarin", "ibuprofen_question"} <= tags:
        return "escalate", "Synthetic medication-review rule fired"
    if not retrieved:
        return "abstain", "No relevant local evidence passage retrieved"
    if any(not item["eligible"] for check in claim_checks for item in check["evidence"]):
        return "abstain", "Retrieved source failed status or jurisdiction check"
    if any(check["verdict"] != "supported" for check in claim_checks):
        return "abstain", "At least one candidate claim lacks consistent support"
    if case.get("missing_required_fields"):
        return "clarify", "Required information is missing in the synthetic case"
    if not claim_checks:
        return "abstain", "No candidate claim was supplied"
    return "answer", "All controlled candidate claims have eligible NLI support"


def evaluate_case(case: dict, evidence: list[dict], nli: NLIModel) -> dict:
    included_fixtures = set(case.get("include_synthetic_sources", []))
    corpus = [record for record in evidence
              if not record["synthetic"] or record["id"] in included_fixtures]
    if case.get("only_source_ids"):
        selected = set(case["only_source_ids"])
        corpus = [record for record in corpus if record["id"] in selected]
    retrieved = retrieve(case["query"], corpus, case.get("top_k", 1))
    overrides = case.get("source_status_overrides", {})
    claim_checks = [verify_claim(claim, retrieved, case["jurisdiction"], overrides, nli)
                    for claim in case["candidate_claims"]]
    action, reason = choose_action(case, claim_checks, retrieved)
    # Deliberately weak proxy: source presence alone, without verification or rules.
    proxy_action = "answer" if retrieved else "abstain"
    return {
        "case_id": case["id"],
        "query": case["query"],
        "jurisdiction": case["jurisdiction"],
        "candidate_claims": case["candidate_claims"],
        "missing_required_fields": case.get("missing_required_fields", []),
        "fixture_tags": case.get("tags", []),
        "retrieved_source_ids": [record["id"] for record in retrieved],
        "claim_checks": claim_checks,
        "action": action,
        "reason": reason,
        "retrieval_only_proxy_action": proxy_action,
        "authored_fixture_expectation": case["expected_action"],
        "matches_authored_expectation": action == case["expected_action"],
    }


def print_result(result: dict) -> None:
    print(f"{result['case_id']}: {result['action'].upper()} "
          f"(fixture: {result['authored_fixture_expectation']}; "
          f"retrieval-only proxy: {result['retrieval_only_proxy_action']})")
    print(f"  reason: {result['reason']}")
    print(f"  retrieved: {', '.join(result['retrieved_source_ids']) or 'none'}")
    for check in result["claim_checks"]:
        print(f"  {check['verdict']}: {check['claim']}")
        for record in check["evidence"]:
            print(f"    {record['source_id']}: {record['nli']['label']}, "
                  f"scores={record['nli']['scores']}, eligible={record['eligible']}, "
                  f"status={record['status']}")


def runtime_environment() -> dict[str, str]:
    """Record comparison-relevant host details without a user or host name."""
    import onnxruntime

    cpu = platform.processor() or "unknown"
    if platform.system() == "Windows":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as file:
                for line in file:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return {
        "cpu_model": cpu,
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "execution_provider": "CPUExecutionProvider",
        "model_export": "onnx/model_quint8_avx2.onnx",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic model-backed evidence audit; not medical advice")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", metavar="ID", help="Run one named synthetic case")
    group.add_argument("--all", action="store_true", help="Run all cases (default)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    parser.add_argument("--output", type=Path, help="Write JSON to a UTF-8 file; requires --json")
    args = parser.parse_args(argv)
    if args.output and not args.json:
        parser.error("--output requires --json")
    evidence, cases = load_data()
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            parser.error("Unknown case ID")
    from model_nli import LocalNLI, MODEL_ID, MODEL_REVISION

    try:
        model = LocalNLI()
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    results = [evaluate_case(case, evidence, model) for case in cases]
    if args.json:
        payload = json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION,
                              "environment": runtime_environment(),
                              "scores_are_calibrated": False, "results": results}, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    else:
        for result in results:
            print_result(result)
        matched = sum(result["matches_authored_expectation"] for result in results)
        print(f"\nAuthored fixture agreement: {matched}/{len(results)} "
              "(synthetic engineering check, not clinical accuracy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
