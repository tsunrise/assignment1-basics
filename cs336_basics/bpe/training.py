from collections import Counter
import regex as re
from collections.abc import Generator

class BpeModel:
    vocab: dict[int, bytes]
    """
    The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes)
    """

    merges: list[tuple[bytes, bytes]]
    """
    A list of BPE merges produced from training. Each list item is a tuple of bytes `(<token1>, <token2>)`, representing
    that `<token1>` was merged with `<token2>`. The merges should be ordered by order of creation.
    """


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> BpeModel:
    """
    Train a BPE model.

    - `input_path`: Path to a text file with BPE tokenizer training data.
    - `vocab_size`: A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    - `special_tokens`: A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
    """

    # read the file as bytes
    with open(input_path, "rb") as f:
        data = f.read()

    # pre-tokenize the data
    pre_tokens = Counter()
    for pre_token in pre_tokenize(data):
        pre_tokens[tuple(pre_token)] += 1

    # initialize vocab as byte 0 to 255
    vocabs = [bytes(i) for i in range(256)]
    vocabs_to_idx = {bytes(i): i for i in range(256)}
    pairs = PairCounter()

    for token, count in pre_tokens.items():
        for i in range(len(token) - 1):
            pairs.add((bytes([token[i]]), bytes([token[i + 1]])), count)

    # merge most common pair in pre_tokens and count again until we have vocab_size
    merges = []
    while len(pairs) > vocab_size:
        ...

class PairCounter:
    _inner: Counter[tuple[bytes, bytes]]
    _count_to_key: dict[int, set[tuple[bytes, bytes]]]
    _max_count: int

    def __init__(self) -> None:
        self._inner = Counter()
        self._count_to_key = {}
        self._max_count = 0

    def __len__(self):
        return len(self._inner)

    def most_common(self) -> tuple[bytes, bytes]:
        assert self._max_count > 0, "most_common should only be called when there is already pairs here"
        candidate_pairs = iter(self._count_to_key[self._max_count])
        best_pair = next(candidate_pairs)
        for candidate_pair in candidate_pairs:
            # lexicographically greater pair wins
            if self._is_right_greater_pair(best_pair, candidate_pair):
                best_pair = candidate_pair
        return best_pair
    
    def add(self, pair: tuple[bytes, bytes], count: int):
        prev_total = self._inner[pair]
        self._inner[pair] += count
        curr_total = self._inner[pair]
        if prev_total in self._count_to_key:
            self._count_to_key[prev_total].remove(pair)
        self._count_to_key.get(curr_total, set()).add(pair)
        return self._inner[pair]

    def _is_right_greater_pair(self, left: tuple[bytes, bytes], right: tuple[bytes, bytes]) -> bool:
        """
        
        """
        left0, left1 = left
        right0, right1 = right

        i = 0
        left_total = len(left0) + len(left1)
        right_total = len(right0) + len(right1)
        min_total = min(left_total, right_total)

        # Compare (left0 + left1) vs (right0 + right1) byte by byte
        # without allocating concatenated bytestrings.
        while i < min_total:
            lb = left0[i] if i < len(left0) else left1[i - len(left0)]
            rb = right0[i] if i < len(right0) else right1[i - len(right0)]
            if lb != rb:
                return rb > lb
            i += 1

        # If one virtual concatenation is a prefix of the other,
        # the longer one is lexicographically larger.
        return right_total > left_total

    


PRETOKENIZE_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenize(input: bytes) -> Generator[bytes]:
    """
    Split the input bytes into pre-tokens, each represented as a sequence of UTF-8 bytes.
    """
    text = input.decode("utf-8")
    for match in PRETOKENIZE_PATTERN.finditer(text):
        yield match.group(0).encode("utf-8")
