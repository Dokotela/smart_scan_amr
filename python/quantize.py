from onnxruntime.quantization import quantize_dynamic, QuantType

# Path to your exported ONNX models
encoder_model_path = "trocr_onnx/encoder_model.onnx"
encoder_quantized_path = "trocr_onnx/encoder_model_quantized.onnx"

decoder_model_path = "trocr_onnx/decoder_model.onnx"
decoder_quantized_path = "trocr_onnx/decoder_model_quantized.onnx"

print(f"Quantizing encoder model from {encoder_model_path} to {encoder_quantized_path}...")
quantize_dynamic(
    encoder_model_path,
    encoder_quantized_path,
    weight_type=QuantType.QUInt8,  # Using UInt8 instead of Int8
    per_channel=False,             # Disable per-channel quantization
    reduce_range=False             # Disable range reduction
)
print("Encoder quantization complete!")

print(f"Quantizing decoder model from {decoder_model_path} to {decoder_quantized_path}...")
quantize_dynamic(
    decoder_model_path,
    decoder_quantized_path,
    weight_type=QuantType.QUInt8,  # Using UInt8 instead of Int8
    per_channel=False,             # Disable per-channel quantization
    reduce_range=False             # Disable range reduction
)
print("Decoder quantization complete!")

print("Quantization finished successfully!")