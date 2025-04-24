import numpy as np
import onnxruntime as ort
from transformers import TrOCRProcessor

# Paths to your ONNX files
ENC_PATH = "onnx/trocr-base-printed/no-past/encoder_model.onnx"
DEC_PATH = "onnx/trocr-base-printed/with-past/decoder_model.onnx"

# 1. Load processor to prepare dummy image input
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
dummy_image = np.zeros((1, 3, 384, 384), dtype=np.float32)  # zeros OK for smoke test

# 2. Create ONNX sessions
enc_sess = ort.InferenceSession(ENC_PATH, providers=["CPUExecutionProvider"])
dec_sess = ort.InferenceSession(DEC_PATH, providers=["CPUExecutionProvider"])

# 3. Print model I/O metadata
print("Encoder I/O:")
for inp in enc_sess.get_inputs():
    print(f"  input: {inp.name} shape={inp.shape} type={inp.type}")
for out in enc_sess.get_outputs():
    print(f"  output: {out.name} shape={out.shape} type={out.type}")

print("\nDecoder I/O:")
for inp in dec_sess.get_inputs():
    print(f"  input: {inp.name} shape={inp.shape} type={inp.type}")
for out in dec_sess.get_outputs():
    print(f"  output: {out.name} shape={out.shape} type={out.type}")

# 4. Run dummy encoder inference
enc_inputs = {enc_sess.get_inputs()[0].name: dummy_image}
enc_outs = enc_sess.run(None, enc_inputs)
print("\nEncoder output shapes:", [o.shape for o in enc_outs])

# 5. Run dummy decoder inference
# For no-past decoder (use "image-to-text" no-past), you’d feed encoder hidden directly.
# For with-past decoder, you need initial decoder input_ids + encoder_hidden_states.

# Here’s a minimal with-past call:
# assume input_ids shape [1,1] with BOS token (id=processor.tokenizer.bos_token_id)
input_ids = np.array([[processor.tokenizer.bos_token_id]], dtype=np.int64)
dec_inputs = {
    dec_sess.get_inputs()[0].name: input_ids,
    dec_sess.get_inputs()[1].name: enc_outs[0],
}
dec_outs = dec_sess.run(None, dec_inputs)
print("Decoder output shapes:", [o.shape for o in dec_outs])
