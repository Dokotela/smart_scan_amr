import numpy as np
import onnxruntime as ort

# Path to your quantized decoder
MODEL_PATH = "onnx/trocr-base-printed/with-past/decoder_model_quant.onnx"

# Create an ONNX Runtime session
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Print I/O metadata for sanity
print("Decoder Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("Decoder Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

# Dummy inputs
input_ids = np.zeros((1, 1), dtype=np.int64)               # one BOS token
encoder_hidden = np.zeros((1, 577, 768), dtype=np.float32)  # dummy encoder output

# Run inference
outputs = sess.run(
    None,
    {
        sess.get_inputs()[0].name: input_ids,
        sess.get_inputs()[1].name: encoder_hidden,
    }
)
print("✅ Decoder quant logits shape:", outputs[0].shape)
