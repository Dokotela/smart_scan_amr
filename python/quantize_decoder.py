from onnxruntime.quantization import quantize_dynamic, QuantType

# Path to your un-quantized decoder
src = "onnx/trocr-base-printed/with-past/decoder_model.onnx"
dst = "onnx/trocr-base-printed/with-past/decoder_model_quant.onnx"

quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QInt8
)
print("✅ Dynamic-quantized decoder written to:", dst)
