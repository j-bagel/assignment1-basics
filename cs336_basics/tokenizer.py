import json
from cs336_basics.pretokenization import pretokenize


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        self.token_to_id = {v: k for k, v in vocab.items()}


    def encode(self, text: str) -> list[int]:
        pretok_iter = pretokenize(text)
        res = []
        # TODO: decode non vocab str
        for pretok in pretok_iter:
            id_ = self.token_to_id.get(pretok, None)
            if id_ is None:
                pass
            else:
                res.append(self.token_to_id[pretok])
        return res


    def decode(self, ids: list[int]) -> str:
        replacement_bytes = b"\xEF\xBF\xBD"
        bytes_list = [self.vocab.get(i, replacement_bytes) for i in ids]
        return b"".join(bytes_list).decode("utf-8")
