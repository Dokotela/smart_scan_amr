from transformers import TrOCRProcessor
import json

# Load the processor
processor = TrOCRProcessor.from_pretrained("trocr_onnx/")

# Get the tokenizer's vocabulary
vocab = processor.tokenizer.get_vocab()

# Extract and print special token IDs
print(f"BOS token: '{processor.tokenizer.bos_token}', ID: {processor.tokenizer.bos_token_id}")
print(f"EOS token: '{processor.tokenizer.eos_token}', ID: {processor.tokenizer.eos_token_id}")
print(f"PAD token: '{processor.tokenizer.pad_token}', ID: {processor.tokenizer.pad_token_id}")

# Print a few sample tokens
print("\nSample tokens:")
for i, (token, idx) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:20]):
    print(f"ID: {idx}, Token: '{token}'")

# Dump the vocabulary to a file
with open("vocab_dump.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"\nVocabulary saved to vocab_dump.json ({len(vocab)} tokens)")