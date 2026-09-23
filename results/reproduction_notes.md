# Cross-host reproduction notes

The pinned quantized ONNX file and the same fixture do **not** guarantee identical floating-point scores across hosts. `model_run.json` is one observed run, not a canonical set of numerical outputs. Its `environment` object records the development host without a user name or computer name.

| Observation | Development host, directly rerun | Second x86-64 host, reported by user |
| --- | --- | --- |
| Case | `mixed_passage_nli_failure` | `mixed_passage_nli_failure` |
| Entailment score | 0.6486 | 0.4733 |
| Contradiction score | 0.3038 | 0.4804 |
| Top-two difference | 0.3448 | 0.0071 |
| Model label after the 0.15 ambiguity rule | entailment | uncertain |
| Final action | answer | abstain |
| Authored fixture agreement across all 13 cases | 12/13 | 13/13, per user report |

## Host details

| Field | Development host | Second host |
| --- | --- | --- |
| CPU | Intel Core i7-9750H @ 2.60 GHz | not yet provided |
| OS | Windows 11 Enterprise, build 26200 | not yet provided |
| Architecture | x86-64 (AMD64) | x86-64, per user report |
| Python | 3.12.14 | not yet provided |
| ONNX Runtime | 1.30.0 | not yet provided |
| Execution provider | CPUExecutionProvider | not yet verified |
| Model export | `onnx/model_quint8_avx2.onnx` at pinned revision | same pinned download and passing SHA-256 checks, per user report |

The development score was unchanged across three repeated predictions in one process and a run from a second fresh virtual environment on the same machine. The second-host values were supplied by the user; we have not reproduced that host locally or received its full environment metadata. Different CPU kernels for the int8 export are a plausible explanation, but the cause has **not been isolated**. Runtime version, CPU features, and other environment differences also need checking.

This case reveals a decision boundary that moves with numerical inference. The scores are uncalibrated and neither 12/13 nor 13/13 is a clinical performance estimate. To compare runs, use `python abstain_md.py --case mixed_passage_nli_failure --json` and include the emitted `environment` object and all three NLI scores. A later study should test multiple hosts and export formats, predefine an ambiguity policy on a separate development set, and evaluate against independently adjudicated labels.
