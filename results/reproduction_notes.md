# Runtime-version reproduction notes

`model_run.json` is the directly recorded Windows development run. Its `environment` object identifies the CPU, OS, Python, ONNX Runtime, execution provider, and model export without recording a user or computer name. The two Linux observations below were supplied by the user after rerunning the **same fixture and pinned model export on the same machine**, changing ONNX Runtime from 1.25.0 to the repository's required 1.30.0.

| Run | ONNX Runtime | Entailment | Contradiction | Label after 0.15 ambiguity rule | Action |
| --- | --- | ---: | ---: | --- | --- |
| Windows i7-9750H, directly recorded | 1.30.0 | 0.6486 | 0.3038 | entailment | answer |
| Linux Xeon, user report | 1.25.0 | 0.4733 | 0.4804 | uncertain | abstain |
| Same Linux Xeon, user rerun with repository pin | 1.30.0 | 0.6486 | 0.3038 | entailment | answer |

The Linux result with **ONNX Runtime 1.30.0 matches the Windows result to four decimal places**. Within the Linux host, changing to 1.25.0 moved the model scores across the ambiguity rule and changed the synthetic action. This controlled rerun isolates the **runtime version as the cause of the reported divergence**; it does not establish that every CPU and OS will produce identical results. The dependency pin is part of this demonstration's evidence chain.

## Recorded environments

| Field | Windows development run | Both Linux runs, user report |
| --- | --- | --- |
| CPU | Intel Core i7-9750H @ 2.60 GHz | Intel Xeon @ 2.80 GHz, AVX2 and AVX-512 |
| OS | Windows 11 Enterprise, build 26200 | Ubuntu 24.04.4, Linux 6.18 |
| Architecture | x86-64 (AMD64) | x86-64 |
| Python | 3.12.14 | 3.11.15 |
| Execution provider | CPUExecutionProvider | CPUExecutionProvider |
| Model export | pinned `onnx/model_quint8_avx2.onnx`, SHA-256 checked | same pinned export, SHA-256 checks passed |
| Tests | six passed | six passed on both runtime versions, per user report |

On the two 1.30.0 runs, the authored fixture agreement is **12/13**. On the Linux 1.25.0 run, the user reported **13/13** because the mixed-passage case abstained. Neither figure is clinical accuracy: the cases and expectations are author-written engineering fixtures. The mixed-passage case remains a real failure under the **pinned** runtime, where the model falsely supports “Antibiotics treat viral colds.”

To compare a new run, install exactly `requirements.txt`, then run `python abstain_md.py --case mixed_passage_nli_failure --json`. Include its emitted `environment` object and all three uncalibrated NLI scores. A later study should test more hosts and export formats, predefine the ambiguity policy on a separate development set, and evaluate against independently adjudicated labels.
