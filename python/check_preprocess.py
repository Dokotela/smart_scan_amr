from PIL import Image
import numpy as np
from transformers import TrOCRProcessor

# load the same HF processor
proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

# load your actual hello_world.png
img = Image.open("hello_world.png").convert("RGB")

# this will resize→normalize→to NCHW float32
pv = proc(images=img, return_tensors="np").pixel_values.astype(np.float32)

flat = pv.flatten()
print("Python preprocess [0..9]:", flat[:10])
