# Knowing When Not to Answer

A small, model-backed **research proof of concept** for the Milano-Bicocca 114R PROG.5 proposal. It tests whether a candidate claim follows from a retrieved evidence passage, then records why a policy would answer, ask for clarification, abstain, or escalate.

**This is not a clinical assistant or medical advice.** It uses authored synthetic cases, pre-written candidate claims, and manually supplied red-flag tags. It does not interpret unrestricted patient messages, generate answers, determine a diagnosis, or establish clinical safety. The model scores are **uncalibrated**.

## What runs

```text
synthetic query -> token-overlap retrieval -> local MiniLM NLI per controlled claim
                -> source status/jurisdiction check + explicit fixture rules
                -> action and inspectable trace
```

The real [MiniLM NLI model](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) judges evidence–claim pairs as entailment, contradiction, or neutral. Its model card describes training on general-language SNLI and MultiNLI, **not clinical validation**. This repository pins an ONNX export to a specific model commit and checks the downloaded files' SHA-256 hashes. Model weights stay in an ignored local cache. The evidence corpus contains short **author-written paraphrases**, not live guideline retrieval.

The comparison called **retrieval-only proxy** simply answers whenever any passage is found. It is deliberately weak and is **not** a full RAG system or a fair clinical baseline.

## Run locally

The directly tested setup is Python 3.12 on an x86-64 CPU with AVX2, required by the selected quantized ONNX export. A Linux Xeon user also reproduced the pinned result with Python 3.11.15. **Install from `requirements.txt`: ONNX Runtime 1.30.0 is part of the reproducibility contract.** On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe abstain_md.py --all
.\.venv\Scripts\python.exe abstain_md.py --case synthetic_conflict --json
.\.venv\Scripts\python.exe abstain_md.py --all --json --output results/model_run.json
.\.venv\Scripts\python.exe -m unittest -v test_abstain_md.py
```

On supported Linux x86-64 systems, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. The first model download needs internet access; subsequent runs use the local cache. No API key or patient data is required. `--all --json` emits every source ID, passage date, uncalibrated NLI score, policy decision, and authored fixture expectation.

## Evidence and fixtures

The three non-synthetic passage records in `evidence.json` were paraphrased and checked on **23 September 2026** against [CDC antibiotic-use guidance](https://www.cdc.gov/antibiotic-use/about/), [NHS warfarin guidance](https://www.nhs.uk/medicines/warfarin/), and [NHS chest-pain guidance](https://www.nhs.uk/symptoms/chest-pain/). Each record carries a source URL, jurisdiction, source date, and snapshot date. The `approved` flag means **allow-listed for this software demonstration**, not clinically approved. The two records marked `synthetic: true` are invented stress-test passages; they are excluded unless a fixture explicitly includes them. The stale-source condition is also explicitly simulated; it does not assert that CDC guidance is stale.

`cases.json` contains 13 authored cases. It covers supported, contradicted, unsupported, conflicting, no-evidence, jurisdiction-mismatched, and simulated stale evidence, plus clarification and red-flag escalation. These are engineering fixtures, not clinician annotations.

## Observed model run

On the **recorded development host** (Intel Core i7-9750H, Windows 11, ONNX Runtime 1.30.0), the policy matched the authored fixture action in **12 of 13 cases**; the retrieval-only proxy matched **3 of 13**. These designed engineering fixtures are **not evaluation data**: neither fraction is clinical accuracy or an unbiased comparison. The saved [model trace](results/model_run.json) includes that host's environment and the model's outputs.

The mixed-passage case is **runtime-version-sensitive**. On the Windows development host with pinned ONNX Runtime **1.30.0**, entailment was **0.6486** and contradiction **0.3038**, so the policy answered incorrectly. The user first ran the same pinned export on a Linux Xeon with **1.25.0** and obtained **0.4733 / 0.4804**, causing abstention. They then changed only ONNX Runtime to **1.30.0** on that Linux host and obtained **0.6486 / 0.3038**, matching Windows to four decimal places and restoring the answer. Their controlled rerun isolates the reported divergence to the runtime version; it does not prove universal numerical identity across hosts. See [runtime-version reproduction notes](results/reproduction_notes.md).

The development-host failure is why passage granularity and independent clinical review matter. It also shows why an entailment label cannot be treated as proof of safety. The current rule tags are supplied by fixtures, so escalation is not evidence of a working symptom parser. No probabilities are calibrated, no Estimated Calibration Index is computed, and no result supports deployment. The next research step is clinician-adjudicated claim/evidence labels, a separate development set for decision thresholds, and held-out source and wording tests.

## Reproduce the reported result

Run `python abstain_md.py --all --json` after installing the pinned dependencies and compare your **case decisions and environment** with `results/model_run.json`. The JSON includes the pinned model revision, host details, and `scores_are_calibrated: false`. A different ONNX Runtime version can change both scores and actions, as the [controlled rerun](results/reproduction_notes.md) shows. Run the unit tests to check retrieval, provenance blocking, conflict detection, action precedence, and the JSON interface.

The repository contains no applicant documents, identity files, patient records, model weights, or API credentials.
