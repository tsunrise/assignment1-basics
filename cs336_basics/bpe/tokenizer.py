from collections.abc import Generator, Iterable
from cs336_basics.bpe.utils import split_on_special_tokens

class BpeTokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.word_to_token = {b: i for i, b in vocab.items()}
        self.merges_order = {(self.word_to_token[left], self.word_to_token[right]): i for i, (left, right) in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()

    def to_files(self, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_chunk(text))

    def encode_chunk(self, text: str) -> Generator[int]:
        # pre tokenize the text
        pretokens: list[list[int]] = [] # each pretoken is represented by a tuple of token id in `self.vocab`
        pretokens_idx: dict[str, int] = {} # token string -> pretokens id
        text_in_pretokens_indices = [] # text represented by an ordered sequence of pretokens
        def append_pretoken(pretoken: str):
            if pretoken not in pretokens_idx:
                    pretokens.append([self.word_to_token[pretoken.encode()]])
                    pretokens_idx[pretoken] = len(pretoken)
            text_in_pretokens_indices.append(pretokens_idx[pretoken])
        for lo, hi, is_special in split_on_special_tokens(text, self.special_tokens):
            if is_special:
                append_pretoken(text[lo:hi])
            else:
                for c in range(lo, hi):
                    append_pretoken(text[c])
        
        # merge each pre token
        for pretoken_idx in range(len(pretokens)):
            pretoken = pretokens[pretoken_idx]
            # for now use a simple O(len(pretoken)^2) time algorithm
            while True:
                best_i = None
                # find the lowest rank adjacent pair to merge, if any
                for i in range(len(pretoken) - 1):
                    left, right = pretoken[i], pretoken[i+1]
                    if (left, right) in self.merges_order:
                        if best_i is None or self.merges_order[(left, right)] < self.merges_order[(pretoken[best_i], pretoken[best_i + 1])]:
                            best_i = left
                # if no pair to merge, we are done!
                if best_i is None:
                    break

                new_pretoken: list[int] = []
                new_pretoken.extend(pretoken[j] for j in range(best_i))
                new_pretoken.append(self.word_to_token[self.vocab[pretoken[best_i]] + self.vocab[pretoken[best_i + 1]]])
                new_pretoken.extend(pretoken[j] for j in range(best_i + 2, len(pretoken)))
                pretoken = new_pretoken
            pretokens[pretoken_idx] = pretoken

        for pretoken_idx in text_in_pretokens_indices:
            yield from pretokens[pretoken_idx]

    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int]:
        raise NotImplementedError
