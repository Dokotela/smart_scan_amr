from onnxruntime.quantization import quantize_dynamic, QuantType

# Your original FP32 decoder
src = "onnx/trocr-base-printed/with-past/decoder_model.onnx"
# A new output path (so you can compare)
dst = "onnx/trocr-base-printed/with-past/decoder_model_quant2.onnx"

quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=True
)
print("✅ Re-quantized decoder (per-channel + reduced-range) written to:", dst)
