"""Download the pinned MiniLM ONNX model. No model weights enter Git."""

from model_nli import MODEL_ID, MODEL_REVISION, download_model


if __name__ == "__main__":
    print(f"Downloading {MODEL_ID}@{MODEL_REVISION}")
    for path in download_model():
        print(path)
