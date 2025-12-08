import os
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer

# OWT train: 2.7e9 tokens
# OWT valid: 6.6e7 tokens

"""
Why uint16 is an appropriate choice for storing token IDs:

1. Vocabulary size: The OWT tokenizer has vocab_size=32,000
   - uint16 can represent values from 0 to 65,535 (2^16 - 1)
   - This is more than enough to represent all 32,000 token IDs

2. Memory efficiency:
   - uint16 uses 2 bytes per token
   - uint32 would use 4 bytes per token (2x more memory)
   - uint8 only goes up to 255, which is insufficient for vocab_size=32,000

3. For context:
   - Most modern tokenizers have vocab sizes between 10k-50k
   - uint16 (max 65,535) covers this range perfectly
   - GPT-2 has vocab_size=50,257, still fits in uint16
   - Only very large tokenizers (e.g., 100k+ vocab) would need uint32
"""


if __name__ == '__main__':
    # Load OWT tokenizer
    print("Loading OWT tokenizer...")
    tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_owt')
    vocab_path = os.path.join(tok_dir, 'vocab.pkl')
    merges_path = os.path.join(tok_dir, 'merges.pkl')
    special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data/tokenized_owt')
    os.makedirs(output_dir, exist_ok=True)

    def encode_and_save_streaming(input_path: str, output_path: str, chunk_size: int = 1_000_000):
        """Encode a file and save tokens iteratively to avoid loading everything in memory."""
        # Count lines first for progress bar
        with open(input_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        
        # First write to a temporary raw binary file
        tmp_path = output_path + '.tmp'
        total_tokens = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(tmp_path, 'wb') as f_out:
            buffer = []
            for line in tqdm(f_in, total=num_lines, unit='line'):
                buffer.extend(tok.encode(line))
                if len(buffer) >= chunk_size:
                    # Write chunk as uint16
                    chunk = np.array(buffer, dtype=np.uint16)
                    chunk.tofile(f_out)
                    total_tokens += len(buffer)
                    buffer = []
            # Write remaining
            if buffer:
                chunk = np.array(buffer, dtype=np.uint16)
                chunk.tofile(f_out)
                total_tokens += len(buffer)
        
        print(f"  Total: {total_tokens:,} tokens")
        
        # Load raw binary and save as .npy (this is memory-efficient since we just read raw)
        tokens = np.fromfile(tmp_path, dtype=np.uint16)
        np.save(output_path, tokens)
        os.remove(tmp_path)
        
        print(f"Saved to {output_path}")
        print(f"File size: {tokens.nbytes / (1024**2):.2f} MB")

    # Encode validation data
    print("\nEncoding OWT validation data...")
    valid_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                              'data/owt_valid.txt')
    valid_output = os.path.join(output_dir, 'valid.npy')
    encode_and_save_streaming(valid_path, valid_output)
    
    # Encode training data
    print("\nEncoding OWT training data...")
    train_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 
                              'data/owt_train.txt')
    train_output = os.path.join(output_dir, 'train.npy')
    encode_and_save_streaming(train_path, train_output)
    
    print("\nDone!")

