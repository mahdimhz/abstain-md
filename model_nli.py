"""Pinned local ONNX natural-language-inference model.

The three output scores are uncalibrated model probabilities. They are not
clinical confidence estimates and must not be interpreted as safety scores.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
MODEL_FILES = ("onnx/model_quint8_avx2.onnx", "tokenizer.json", "config.json")
CACHE_DIR = Path(__file__).resolve().parent / ".model_cache"
LABELS = ("contradiction", "entailment", "neutral")
MODEL_SHA256 = {
    "onnx/model_quint8_avx2.onnx": "44391a5241a62e0083c1a8899a71e69a092b95aea5ba89e14062925468eceac7",
    "tokenizer.json": "82139106e603ee4e1d5bc99d056ccbed5a92bc24848b1b5a7137c26e00d0dbf6",
    "config.json": "8b0e41caff7567c0f53e6983f35591c3dec59507c9173ab125c5823394fb57f3",
}


def model_file(filename: str, *, download: bool = False) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=MODEL_ID,
        filename=filename,
        revision=MODEL_REVISION,
        cache_dir=CACHE_DIR,
        local_files_only=not download,
    )


def download_model() -> list[str]:
    """Download the pinned ONNX export and tokenizer into an ignored cache."""
    paths = [model_file(filename, download=True) for filename in MODEL_FILES]
    for filename, path in zip(MODEL_FILES, paths):
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256[filename]:
            raise RuntimeError(f"Unexpected model-file checksum: {filename}")
    return paths


class LocalNLI:
    """Score (evidence passage, candidate claim) pairs on the CPU."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        try:
            model_path = model_file(MODEL_FILES[0])
            tokenizer_path = model_file(MODEL_FILES[1])
            config_path = model_file(MODEL_FILES[2])
        except Exception as exc:
            raise RuntimeError("Model files missing. Run: python download_model.py") from exc
        with open(config_path, encoding="utf-8") as file:
            labels = json.load(file)["id2label"]
        if tuple(labels[str(index)] for index in range(3)) != LABELS:
            raise RuntimeError("Pinned model label mapping changed")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, evidence: str, claim: str) -> dict:
        import numpy as np

        pair = self.tokenizer.encode(evidence, claim)
        # The model supports 514 positions; short curated passages stay well below it.
        if len(pair.ids) > 512:
            return {"label": "uncertain", "scores": {}, "reason": "Input exceeds 512 tokens"}
        inputs = {
            "input_ids": np.asarray([pair.ids], dtype=np.int64),
            "attention_mask": np.asarray([pair.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.asarray([pair.type_ids], dtype=np.int64)
        logits = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        scores = {label: round(float(probabilities[i]), 4) for i, label in enumerate(LABELS)}
        order = np.argsort(probabilities)[::-1]
        label = LABELS[int(order[0])]
        # This is an engineering ambiguity rule, not a calibrated threshold.
        if probabilities[order[0]] - probabilities[order[1]] < 0.15:
            label = "uncertain"
        return {"label": label, "scores": scores, "reason": "uncalibrated NLI output"}
