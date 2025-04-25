from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="onnx/trocr-base-printed/no-past/encoder_model.onnx",
    model_output="onnx/trocr-base-printed/no-past/encoder_model_dyn.onnx",
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=['MatMul', 'Gemm']
)
print("✅ Encoder dyn-quant (MatMul/Gemm only)")
