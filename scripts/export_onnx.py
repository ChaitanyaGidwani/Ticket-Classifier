#!/usr/bin/env python3
"""
Export sentence-transformers/all-MiniLM-L6-v2 to ONNX and apply dynamic INT8 quantization.

Outputs:
  - .cache/minilm.onnx (FP32 model ~90MB)
  - .cache/minilm_int8.onnx (INT8 quantized model ~22MB)
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(".cache")


class BertWrapper(torch.nn.Module):
    """Wrapper that adapts BertModel forward for ONNX exporter compatibility."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp32_onnx_path = CACHE_DIR / "minilm.onnx"
    int8_onnx_path = CACHE_DIR / "minilm_int8.onnx"

    print(f"Loading Hugging Face model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.eval()
    wrapped_model = BertWrapper(base_model)

    dummy_text = ["This is a test support ticket."]
    inputs = tokenizer(dummy_text, padding=True, truncation=True, max_length=128, return_tensors="pt")

    print(f"Exporting FP32 ONNX model to {fp32_onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"FP32 ONNX export complete ({fp32_onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"Applying INT8 dynamic quantization to {int8_onnx_path}...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=str(fp32_onnx_path),
            model_output=str(int8_onnx_path),
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 ONNX model created successfully ({int8_onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"Quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
