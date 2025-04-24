import numpy as np
import onnxruntime as ort

# Path to your newly QDQ-quantized encoder
MODEL_PATH = "onnx/trocr-base-printed/no-past/encoder_model_qdq.onnx"

# Create an ONNX Runtime session
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Build a dummy input ([1,3,384,384] of zeros)
inp_name = sess.get_inputs()[0].name
dummy = np.zeros((1, 3, 384, 384), dtype=np.float32)

# Run inference
out = sess.run(None, {inp_name: dummy})

# Print the output shape
print("Encoder QDQ output shape:", out[0].shape)
