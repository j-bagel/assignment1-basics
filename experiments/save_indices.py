import os
import numpy as np
from cs336_basics.tokenizer import Tokenizer


"""
Why uint16 is an appropriate choice for storing token IDs:

1. Vocabulary size: The TinyStories tokenizer has vocab_size=10,000
   - uint16 can represent values from 0 to 65,535 (2^16 - 1)
   - This is more than enough to represent all 10,000 token IDs

2. Memory efficiency:
   - uint16 uses 2 bytes per token
   - uint32 would use 4 bytes per token (2x more memory)
   - uint8 only goes up to 255, which is insufficient for vocab_size=10,000

3. For context:
   - Most modern tokenizers have vocab sizes between 10k-50k
   - uint16 (max 65,535) covers this range perfectly
   - GPT-2 has vocab_size=50,257, still fits in uint16
   - Only very large tokenizers (e.g., 100k+ vocab) would need uint32
"""


if __name__ == '__main__':
    # Load TinyStories tokenizer
    print("Loading TinyStories tokenizer...")
    tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_tinystories')
    vocab_path = os.path.join(tok_dir, 'vocab.pkl')
    merges_path = os.path.join(tok_dir, 'merges.pkl')
    special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_tinystories')
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode training data
    print("\nEncoding TinyStories training data...")
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 
                              'data/TinyStoriesV2-GPT4-train.txt')
    
    with open(train_path, 'rb') as f:
        train_ids = tok.encode_batch(f, num_processes=4)
    
    # Convert to numpy array with uint16 dtype
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    
    # Save
    train_output = os.path.join(output_dir, 'train.npy')
    np.save(train_output, train_ids_np)
    print(f"Saved {len(train_ids_np):,} tokens to {train_output}")
    print(f"File size: {train_ids_np.nbytes / (1024**2):.2f} MB")
    
    # Encode validation data
    print("\nEncoding TinyStories validation data...")
    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 
                              'data/TinyStoriesV2-GPT4-valid.txt')
    
    with open(valid_path, 'rb') as f:
        valid_ids = tok.encode_batch(f, num_processes=4)
    
    # Convert to numpy array with uint16 dtype
    valid_ids_np = np.array(valid_ids, dtype=np.uint16)
    
    # Save
    valid_output = os.path.join(output_dir, 'valid.npy')
    np.save(valid_output, valid_ids_np)
    print(f"Saved {len(valid_ids_np):,} tokens to {valid_output}")
    print(f"File size: {valid_ids_np.nbytes / (1024**2):.2f} MB")
    
    print("\nDone!")

