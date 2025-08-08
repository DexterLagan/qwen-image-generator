#!/usr/bin/env python3
"""Stress test for model loading with blank weights."""

from pathlib import Path
from unittest.mock import patch

import qwen_gui


def fake_from_pretrained(*args, **kwargs):
    raise RuntimeError("safetensors file is empty")


def main():
    dummy_path = qwen_gui.MODEL_DIR / "cache"
    dummy_path.mkdir(parents=True, exist_ok=True)
    (dummy_path / "dummy.safetensors").write_bytes(b"")

    with patch("qwen_gui.DiffusionPipeline.from_pretrained", side_effect=fake_from_pretrained):
        result = qwen_gui.load_existing_model()
        print(result)
        if "❌" in result:
            print("Dummy model load test passed")
        else:
            print("Dummy model load test failed")


if __name__ == "__main__":
    main()

