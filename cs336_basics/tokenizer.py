import json
from typing import BinaryIO, Iterable, Iterator
from cs336_basics.pretokenization import pretokenize
from cs336_basics.tokenizer_utils import merge_pretok


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            # sort so that longer special token will be matched first
            # example corner case: special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
            special_tokens.sort(key=len, reverse=True)
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges_ranking = {bytes_tuple: i for i, bytes_tuple in enumerate(merges)}


    def encode(self, text: str) -> list[int]:
        pretok_iter = pretokenize(text, self.special_tokens)  # Iterable[bytes]
        res = []
        for pretok in pretok_iter:
            id_ = self.token_to_id.get(pretok, None)
            if id_ is None:
                bytes_list = merge_pretok(pretok, self.merges_ranking)
                res.extend([self.token_to_id.get(b) for b in bytes_list])
            else:
                res.append(id_)
        return res


    def decode(self, ids: list[int]) -> str:
        replacement_bytes = b"\xEF\xBF\xBD"
        bytes_list = [self.vocab.get(i, replacement_bytes) for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            pretok_iter = pretokenize(text, self.special_tokens)  # Iterable[bytes]
            for pretok in pretok_iter:
                id_ = self.token_to_id.get(pretok, None)
                if id_ is None:
                    bytes_list = merge_pretok(pretok, self.merges_ranking)
                    for b in bytes_list:
                        yield self.token_to_id.get(b)
                else:
                    yield id_
