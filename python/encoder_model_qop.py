from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

quantize_static(
    model_input="onnx/trocr-base-printed/no-past/encoder_model.onnx",
    model_output="onnx/trocr-base-printed/no-past/encoder_model_qop.onnx",
    calibration_data_reader=CalibrationDataReader(...),
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
