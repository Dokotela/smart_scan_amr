import onnxruntime as ort
import numpy as np
from transformers import TrOCRProcessor
from PIL import Image

# Load the processor
processor = TrOCRProcessor.from_pretrained("trocr_onnx/")

# Set up inference sessions
encoder_session = ort.InferenceSession("trocr_onnx/encoder_model.onnx")
decoder_session = ort.InferenceSession("trocr_onnx/decoder_model.onnx")

def recognize_text(image_path):
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    input_tensor = inputs.pixel_values.numpy()
    
    # Run encoder
    encoder_inputs = {encoder_session.get_inputs()[0].name: input_tensor}
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    encoder_hidden_states = encoder_outputs[0]
    print(f"Encoder output shape: {encoder_hidden_states.shape}")
    
    # Initialize with start token
    decoder_input_ids = processor.tokenizer.convert_tokens_to_ids([processor.tokenizer.bos_token])
    decoder_input_ids = [decoder_input_ids]
    decoder_input_ids = np.array(decoder_input_ids, dtype=np.int64)
    
    # Autoregressive decoding
    max_length = 30
    output_ids = []
    
    for _ in range(max_length):
        # Run decoder step
        decoder_inputs = {
            decoder_session.get_inputs()[0].name: decoder_input_ids,
            decoder_session.get_inputs()[1].name: encoder_hidden_states
        }
        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]
        
        # Get next token
        next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
        
        # Stop if we hit the end token
        if next_token_id == processor.tokenizer.eos_token_id:
            break
            
        output_ids.append(next_token_id)
        
        # Update input_ids for next iteration
        decoder_input_ids = np.concatenate([
            decoder_input_ids, 
            np.array([[next_token_id]], dtype=np.int64)
        ], axis=1)
    
    # Decode the predicted tokens
    predicted_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return predicted_text

# Test on your images
for image_path in ["hello_world.png", "goodbye_love.png", "121.png"]:
    try:
        text = recognize_text(image_path)
        print(f"Image: {image_path}")
        print(f"Recognized text: '{text}'")
        print("-" * 50)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()