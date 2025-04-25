# quantize_encoder_static.py

import os
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationDataReader
)
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor

class CalibReader(CalibrationDataReader):
    def __init__(self, model_path: str, image_dir: str, processor: TrOCRProcessor):
        # prepare an initial session just to grab the input name
        sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = sess.get_inputs()[0].name

        self.processor = processor
        self.image_dir = image_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self._iter = iter(self._batches())

    def _batches(self):
        print(f"📡 Calibrating on {len(self.image_paths)} images in {self.image_dir}")
        for fn in self.image_paths:
            path = os.path.join(self.image_dir, fn)
            img = Image.open(path).convert("RGB")
            # HuggingFace processor to get pixel_values
            pv = self.processor(images=img, return_tensors="np").pixel_values.astype(np.float32)
            yield {self.input_name: pv}

    def get_next(self):
        return next(self._iter, None)

def main():
    # 1) your FP32 encoder ONNX
    src = "onnx/trocr-base-printed/no-past/encoder_model.onnx"
    # 2) where we’ll write the static QDQ quantized version
    dst = "onnx/trocr-base-printed/no-past/encoder_model_qdq.onnx"
    # 3) folder of real images (handwritten prescriptions + any printed text)
    calib_dir = "../assets/calibration_images"

    print("\n=== Quantizing encoder (static QDQ) ===")
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")  # use_fast can be left default

    quantize_static(
        model_input=src,
        model_output=dst,
        calibration_data_reader=CalibReader(src, calib_dir, proc),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )

    print(f"\n✅ Wrote quantized encoder to: {dst}")

if __name__ == "__main__":
    main()
