# quantize_decoder_dynamic.py

from onnxruntime.quantization import quantize_dynamic, QuantType

src = "onnx/trocr-base-printed/with-past/decoder_model.onnx"
dst = "onnx/trocr-base-printed/with-past/decoder_model_dyn.onnx"

quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QInt8,
    per_channel=True
)
print("✅ Dynamic-quantized decoder written to:", dst)
