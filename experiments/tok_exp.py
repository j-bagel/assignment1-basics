import time
import os
import random
from cs336_basics.tokenizer import Tokenizer
random.seed(123)


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`"""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens

def sample_lines(f_path, n=50):
    with open(f_path, 'r') as f:
        lines = f.readlines()
    sampled_lines = random.sample(lines, n)
    return ''.join(sampled_lines)


print("-------- tinystories tok on tinystories ------------")
txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/TinyStoriesV2-GPT4-valid.txt')
sample_text = sample_lines(txt_path)

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_tinystories')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

# encode
indices = tok.encode(sample_text)
print(f"compression ratio: {get_compression_ratio(sample_text, indices)}")

# show some
print("let's 'visualize' some indices...")
print(tok.decode_to_bytes_list(indices[100:120]))
print(tok.decode_to_bytes_list(indices[400:420]))


print("-------- OWT tok on OWT ------------")
txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/owt_valid.txt')
sample_text = sample_lines(txt_path)

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_owt')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

# encode
indices = tok.encode(sample_text)
print(f"compression ratio: {get_compression_ratio(sample_text, indices)}")

# show some
print("let's 'visualize' some indices...")
print(tok.decode_to_bytes_list(indices[100:120]))
print(tok.decode_to_bytes_list(indices[400:420]))


print("-------- tinystories tok on OWT ------------")
txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/owt_valid.txt')
sample_text = sample_lines(txt_path)

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_tinystories')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

# encode
indices = tok.encode(sample_text)
print(f"compression ratio: {get_compression_ratio(sample_text, indices)}")

# show some
print("let's 'visualize' some indices...")
print(tok.decode_to_bytes_list(indices[100:120]))
print(tok.decode_to_bytes_list(indices[400:420]))


print("-------- OWT tok on tinystories ------------")
txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/TinyStoriesV2-GPT4-valid.txt')
sample_text = sample_lines(txt_path)

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_owt')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

# encode
indices = tok.encode(sample_text)
print(f"compression ratio: {get_compression_ratio(sample_text, indices)}")

# show some
print("let's 'visualize' some indices...")
print(tok.decode_to_bytes_list(indices[100:120]))
print(tok.decode_to_bytes_list(indices[400:420]))

"""
TS tok is obviously not powerful enough for OWT, whereas OWT tok is fine for TS too.
"""


print("\n-------------- throughput ---------------")

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_owt')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

print("1. batch tokenization")

txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/owt_valid.txt')
with open(txt_path, 'r') as f:
    text = f.read()

# encode
start = time.time()
indices = tok.encode(text)
end = time.time()
print(f"Took: {end-start: .2f} seconds on 290MB")
print(f"{290/(end-start):.2f} MB per second")
print(f"Estimated time: {(end-start)*825*1024/290/60/60: .2f} hours on The Pile (825GB)")


print("2. iterable tokenization takes ~35% more time")
# # encode
# start = time.time()
# with open(txt_path, 'r') as f:
#     count = 0
#     for _ in tok.encode_iterable(f):
#         count += 1
# end = time.time()
# print(f"Took: {end-start: .2f} seconds on 290MB ({count} lines)")
# print(f"Estimated time: {(end-start)*825*1024/290/60/60: .2f} hours on The Pile (825GB)")

print("3. batch encoding, 4 processes")

start = time.time()
with open(txt_path, 'rb') as f:
    all_ids = tok.encode_batch(f)
end = time.time()
print(f"Took: {end-start: .2f} seconds on 290MB")
print(f"{290/(end-start):.2f} MB per second")
print(f"Estimated time: {(end-start)*825*1024/290/60/60: .2f} hours on The Pile (825GB)")



print("\n-------- throughput, using the weaker tinystories tokenizer --------")

# load tok
tok_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models/tokenizer_tinystories')
vocab_path = os.path.join(tok_dir, 'vocab.pkl')
merges_path = os.path.join(tok_dir, 'merges.pkl')
special_tokens_path = os.path.join(tok_dir, 'special_tokens.pkl')
tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens_path)

print("1. batch tokenization")

txt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                        'data/owt_valid.txt')
with open(txt_path, 'r') as f:
    text = f.read()

# encode
start = time.time()
indices = tok.encode(text)
end = time.time()
print(f"Took: {end-start: .2f} seconds on 290MB")
print(f"{290/(end-start):.2f} MB per second")
print(f"Estimated time: {(end-start)*825*1024/290/60/60: .2f} hours on The Pile (825GB)")

