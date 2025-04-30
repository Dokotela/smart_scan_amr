#!/usr/bin/env python3
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

CACHE_DIR = Path("trocr-base-handwritten-cache")
for model_name in [
    "encoder_model.onnx",
    "decoder_init_model.onnx",
    "decoder_model.onnx"
]:
    src = CACHE_DIR / model_name
    dst = CACHE_DIR / f"quant_{model_name}"
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True
    )
    print(f"✅ {model_name} → quant_{model_name}")
