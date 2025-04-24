import os
import onnxruntime
from PIL import Image
import numpy as np                                         # make sure this is imported
from transformers import TrOCRProcessor                     # <— changed here
from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat, CalibrationDataReader
)

class TrOCRCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, image_dir, processor):
        # no need for batch_size arg if you’re not using it
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.image_paths = sorted(os.listdir(image_dir))
        self.processor = processor
        self.image_dir = image_dir
        self._iter = iter(self._batches())

    def _batches(self):
        for fn in self.image_paths:
            img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
            # THIS now calls the ViTFeatureExtractor internally, not the tokenizer
            pv = self.processor(images=img, return_tensors="np").pixel_values
            yield {self.input_name: pv.astype(np.float32)}

    def get_next(self):
        return next(self._iter, None)

def main():
    src = "onnx/trocr-base.onnx/model.onnx"
    dst = "onnx/trocr-base-quant.onnx"
    calib_dir = "calibration_images"

    # Load the full TrOCRProcessor (feature_extractor + tokenizer)
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    quantize_static(
        model_input=src,
        model_output=dst,
        calibration_data_reader=TrOCRCalibrationDataReader(src, calib_dir, proc),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False
    )
    print("✅ Quantized INT8 model written to:", dst)

if __name__ == "__main__":
    main()
