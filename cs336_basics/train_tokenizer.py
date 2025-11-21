import json
from multiprocessing import Pool
from functools import partial
from cs336_basics.pretokenization import pretokenize, find_chunk_boundaries, pretokenize_to_list

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 4
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Input:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
        initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary.
    Return:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    special_tokens_bytes = [s.encode('utf-8') for s in special_tokens]

    # --------------------- 1. pretokenize ---------------------
    with open(input_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        # Read all chunks
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    # pretok chunks in parallel
    with Pool(processes=num_processes) as pool:
        pretok_results = pool.map(
            partial(pretokenize_to_list, special_tokens=special_tokens),
            chunks
        )  # list of lists of bytes

    # bytes count dict
    bytes_count = {}
    for b_list in pretok_results:
        for b in b_list:
            if b not in special_tokens_bytes:
                if b in bytes_count:
                    bytes_count[b] += 1
                else:
                    bytes_count[b] = 1

    return None, None


if __name__ == "__main__":
    train_bpe('/Users/jouyang/Documents/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt', 1000, ["<|endoftext|>"])