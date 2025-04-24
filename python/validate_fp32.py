# validate_fp32.py

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import TrOCRProcessor

# ← POINT AT THE ORIGINAL FP32 ONNX FILES
ENC_PATH = "onnx/trocr-base-printed/no-past/encoder_model.onnx"
DEC_PATH = "onnx/trocr-base-printed/with-past/decoder_model.onnx"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# img = Image.open("hello_world.png").convert("RGB")
img = Image.open("goodbye_love.jpg").convert("RGB")
pv = processor(images=img, return_tensors="np").pixel_values.astype(np.float32)

# encoder
enc_sess = ort.InferenceSession(ENC_PATH, providers=["CPUExecutionProvider"])
enc_out = enc_sess.run(None, {"pixel_values": pv})[0]

# decoder (greedy loop)
dec_sess = ort.InferenceSession(DEC_PATH, providers=["CPUExecutionProvider"])
tokens = [processor.tokenizer.bos_token_id]
for _ in range(64):
    logits = dec_sess.run(
        None,
        {
          dec_sess.get_inputs()[0].name: np.array([tokens], dtype=np.int64),
          dec_sess.get_inputs()[1].name: enc_out,
        },
    )[0]
    nxt = int(np.argmax(logits[0, -1]))
    if nxt == processor.tokenizer.eos_token_id:
        break
    tokens.append(nxt)

print("FP32 model OCR →", processor.tokenizer.decode(tokens, skip_special_tokens=True))
