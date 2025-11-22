import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from cs336_basics.pretokenization import pretokenize, find_chunk_boundaries, pretokenize_to_bytes_count
from cs336_basics.tokenizer_utils import pretok_to_dll, merge_one_pair_bpe


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
    print("pretokenizing...")
    with open(input_path, 'rb') as file:
        boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        # Read all chunks
        chunks = []
        start_end_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_end_list.append((start, end))

    # pretok chunks in parallel
    with Pool(processes=num_processes) as pool:
        bytes_count_list = pool.map(
            partial(pretokenize_to_bytes_count, special_tokens=special_tokens, file_path=input_path),
            start_end_list
        )

    # --------------------- 2. prepare ---------------------
    print("preparing...")
    # aggregate bytes count dict
    bytes_count = bytes_count_list[0]
    for d in bytes_count_list[1:]:
        for b in d:
            if b in bytes_count:
                bytes_count[b] += d[b]
            else:
                bytes_count[b] = d[b]

    # bytes should be represented as double linked list so that merging can be easily performed
    bytes_dll = {}
    # TOTAL counts of pairs
    pair_count = {}
    pair_bytes_pointers = {}
    for b in bytes_count:
        bytes_dll[b] = pretok_to_dll(b)
        node = bytes_dll[b].head
        while node.next:
            pair = (node.data, node.next.data)

            if pair in pair_count:
                pair_count[pair] += bytes_count[b]
            else:
                pair_count[pair] = bytes_count[b]
                # and pair won't be in pair_pytes_pointers
                pair_bytes_pointers[pair] = {}

            if b in pair_bytes_pointers[pair]:
                pair_bytes_pointers[pair][b].append(node)
            else:
                pair_bytes_pointers[pair][b] = [node]

            node = node.next

    # --------------------- 3. merge loop ---------------------
    print("merging...")
    vocab = {i: bytes([i]) for i in range(256)}
    for b in special_tokens_bytes:
        if b not in set(vocab.values()):
            vocab[len(vocab)] = b
    merges = []

    steps = vocab_size - len(vocab)
    for _ in tqdm(range(steps)):
        # the pair for this merge step
        max_count = max(pair_count.values())
        # When computing merges, deterministically break ties in pair frequency by
        # preferring the lexicographically greater pair
        pair = max([pair for pair in pair_count if pair_count[pair] == max_count])

        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append(pair)

        # merge
        merge_one_pair_bpe(pair, bytes_count, bytes_dll, pair_count, pair_bytes_pointers)

    return vocab, merges
