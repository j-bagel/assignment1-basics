import os
from typing import BinaryIO, Iterable
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # TODO: not really a generator, double memory usage
    """
    Example:
        split_with_special_tokens("Hello<|endoftext|>World", ["<|endoftext|>"])
        ["Hello", "<|endoftext|>", "World"]
    """
    if not special_tokens:
        return [text] if text else []

    # Escape special tokens and join with | to create pattern
    # Use capturing group () to keep the special tokens in the result
    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"

    # Split but keep the delimiters (special tokens)
    segments = re.split(pattern, text)

    # Filter out empty strings
    return [seg for seg in segments if seg]


def pretokenize(text: str, special_tokens: list[str]) -> Iterable[bytes]:
    for s in split_with_special_tokens(text, special_tokens):
        if special_tokens and s in special_tokens:
            yield s.encode("utf-8")
        else:
            for m in re.finditer(PAT, s):
                yield m.group(0).encode("utf-8")


def pretokenize_to_bytes_count(start_end: tuple[int, int], file_path: str, special_tokens: list[str]) -> dict[bytes, int]:
    special_tokens_bytes = [s.encode('utf-8') for s in special_tokens]

    start, end = start_end

    with open(file_path, 'rb') as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8", errors="ignore")

    bytes_count = {}
    for s in split_with_special_tokens(text, special_tokens):
        if s not in special_tokens:
            res_now = [x.encode("utf-8") for x in re.findall(PAT, s)]
            for b in res_now:
                if b in bytes_count:
                    bytes_count[b] += 1
                else:
                    bytes_count[b] = 1

    return bytes_count


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# print(list(pretokenize("Hello world<|endoftext|>Beautiful World", ["<|endoftext|>"])))
