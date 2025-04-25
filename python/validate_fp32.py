# validate_quant_debug.py

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import TrOCRProcessor

# static‐QD encoder + dynamic‐quant decoder
ENC_PATH = "onnx/trocr-base-printed/no-past/encoder_model_dyn.onnx"
DEC_PATH = "onnx/trocr-base-printed/with-past/decoder_model_dyn.onnx"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)

def run_ocr_on(img_name):
    print(f"\n=== {img_name} ===")
    # 1) Load & preprocess
    img = Image.open(img_name).convert("RGB")
    pv = processor(images=img, return_tensors="np").pixel_values.astype(np.float32)
    # print the first 10 values of channel 0 after normalize
    flat = pv.flatten()
    print("pixel_values[0..9]:", np.round(flat[:10], 6))

    # 2) Encoder
    enc_sess = ort.InferenceSession(ENC_PATH, providers=["CPUExecutionProvider"])
    enc_out = enc_sess.run(None, {"pixel_values": pv})[0]
    # print a tiny slice of the encoder output
    eo_flat = enc_out.reshape(-1)
    print("encoder_out[0..9]:", np.round(eo_flat[:10], 6))

    # 3) Greedy decode
    dec_sess = ort.InferenceSession(DEC_PATH, providers=["CPUExecutionProvider"])
    tokens = [processor.tokenizer.bos_token_id]
    for step in range(64):
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
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    print("Decoded text →", text)

for name in ["hello_world.png", "goodbye_love.jpg"]:
    run_ocr_on(name)
