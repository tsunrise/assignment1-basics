from collections.abc import Generator, Iterable
from cs336_basics.bpe.utils import (
    BpeParameters,
    compile_special_token_patterns,
    gpt2_bytes_to_unicode,
    split_into_pretokens,
)
from collections import OrderedDict, deque
import json


class BpeTokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]
    ):
        self.vocab = vocab
        self.merges = merges
        self.word_to_token = {b: i for i, b in vocab.items()}
        self.merges_order = {
            (self.word_to_token[left], self.word_to_token[right]): i for i, (left, right) in enumerate(merges)
        }
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pretoken_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.pretoken_cache_max_mize = 4096

    @classmethod
    def from_bpe_parameters(cls, params: BpeParameters):
        return cls(params.vocab, params.merges, params.special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]):
        gpt2_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        # decode vocab from json
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_inv: dict[str, int] = json.load(f)
            vocab = {i: bytes(gpt2_decoder[t] for t in vocab_encoded) for vocab_encoded, i in vocab_inv.items()}
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (bytes([gpt2_decoder[token] for token in left]), bytes([gpt2_decoder[token] for token in right]))
                for left, right in merges
            ]

        return cls(vocab, merges, special_tokens)

    def to_files(self, vocab_filepath: str, merges_filepath: str):
        gpt2_encoder = gpt2_bytes_to_unicode()
        # encode vocab to json
        vocabs_ser = {"".join(gpt2_encoder[b] for b in vocab_bytes): i for (i, vocab_bytes) in self.vocab.items()}
        with open(vocab_filepath, mode="w", encoding="utf-8") as f:
            json.dump(vocabs_ser, f)
        # each merge is a line with gpt2_encoding(left_bytes) gpt2_encoding(right_bytes)
        with open(merges_filepath, mode="w", encoding="utf-8") as f:
            f.writelines(
                " ".join("".join(gpt2_encoder[b] for b in x) for x in (left, right)) + "\n" for (left, right) in self.merges
            )

    def encode(self, text: str) -> list[int]:
        return list(self.encode_chunk(text))

    def encode_chunk(self, text: str) -> Generator[int]:
        # pre tokenize the text
        pretokens: list[tuple[int, ...]] = []  # each pretoken is represented by a tuple of token id in `self.vocab`
        pretoken_to_idx: dict[str, int] = {}  # token string -> pretokens id
        text_in_pretokens_indices = []  # text represented by an ordered sequence of pretokens
        pretokens_text: list[str] = []

        def append_pretoken(pretoken: str, is_special: bool):
            if pretoken not in pretoken_to_idx:
                if pretoken in self.pretoken_cache:
                    pretokens.append(self.pretoken_cache[pretoken])
                elif is_special:
                    pretokens.append((self.word_to_token[pretoken.encode()],))
                else:
                    pretokens.append(tuple(self.word_to_token[bytes([b])] for b in pretoken.encode()))
                pretokens_text.append(pretoken)
                pretoken_to_idx[pretoken] = len(pretokens) - 1
            text_in_pretokens_indices.append(pretoken_to_idx[pretoken])

        for lo, hi, is_special in split_into_pretokens(text, self.special_tokens):
            append_pretoken(text[lo:hi], is_special)

        # merge each pre token
        for pretoken_idx in range(len(pretokens)):
            if pretokens_text[pretoken_idx] not in self.pretoken_cache:
                pretoken = pretokens[pretoken_idx]
                # for now use a simple O(len(pretoken)^2) time algorithm
                while True:
                    best_i = None
                    # find the lowest rank adjacent pair to merge, if any
                    for i in range(len(pretoken) - 1):
                        left, right = pretoken[i], pretoken[i + 1]
                        if (left, right) in self.merges_order:
                            if (
                                best_i is None
                                or self.merges_order[(left, right)]
                                < self.merges_order[(pretoken[best_i], pretoken[best_i + 1])]
                            ):
                                best_i = i
                    # if no pair to merge, we are done!
                    if best_i is None:
                        break

                    new_pretoken: list[int] = []
                    new_pretoken.extend(pretoken[j] for j in range(best_i))
                    new_pretoken.append(
                        self.word_to_token[self.vocab[pretoken[best_i]] + self.vocab[pretoken[best_i + 1]]]
                    )
                    new_pretoken.extend(pretoken[j] for j in range(best_i + 2, len(pretoken)))
                    pretoken = tuple(new_pretoken)
                pretokens[pretoken_idx] = pretoken
                self.pretoken_cache[pretokens_text[pretoken_idx]] = pretoken
                if len(self.pretoken_cache) > self.pretoken_cache_max_mize:
                    self.pretoken_cache.popitem(last=False)
            else:
                self.pretoken_cache.move_to_end(pretokens_text[pretoken_idx], last=True)

        for pretoken_idx in text_in_pretokens_indices:
            yield from pretokens[pretoken_idx]

    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int]:
        special_token_patterns = compile_special_token_patterns(self.special_tokens)
        buf = deque()
        # release the buffer before the beginning of next special token
        # that does not overlap on two string
        # TODO: we could randomize this threshold to prevent malicious dataset putting all special token
        # at boundaries, causing OOM
        stream = iter(iterable)
        while True:
            nxt = None

            # continue accumulate buffer until we see a special token that does not overlap in boundaries
            # TODO: we could further optimize this so that it accumulate until we see a special token that may overlap in boundaries
            while True:
                nxt = next(stream, None)
                if nxt is None:
                    break
                m = next(special_token_patterns.finditer(nxt), None)
                if m is not None:
                    buf.append(nxt[:m.start()])
                    # release buffer
                    yield from self.encode_chunk("".join(buf))
                    buf.clear()
                    buf.append(nxt[m.start():])
                else:
                    buf.append(nxt)

            if nxt is None:
                break

        if len(buf) > 0:
            yield from self.encode_chunk("".join(buf))
            buf.clear()

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[x] for x in ids).decode("utf-8", errors="replace")
