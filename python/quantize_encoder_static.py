import os
import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor

class CalibReader(CalibrationDataReader):
    def __init__(self, model_path, image_dir, processor):
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.processor = processor
        self.image_paths = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self._iter = iter(self._batches())

    def _batches(self):
        for fn in self.image_paths:
            img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
            pv = self.processor(images=img, return_tensors="np").pixel_values
            yield {self.input_name: pv.astype(np.float32)}

    def get_next(self):
        return next(self._iter, None)

def main():
    src = "onnx/trocr-base-printed/no-past/encoder_model.onnx"
    dst = "onnx/trocr-base-printed/no-past/encoder_model_qdq.onnx"
    calib_dir = "../assets/calibration_images"

    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

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
    print("✅ Static QDQ-quantized encoder written to:", dst)

if __name__ == "__main__":
    main()
