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

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
        self.vocab = vocab
        self.merges = merges


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

    # initialize vocab as byte 0 to 255
    vocab = [bytes(i) for i in range(256)]
    for token in special_tokens:
        vocab.append(token.encode("utf-8"))
    word_to_idx = {value: i for (i, value) in enumerate(vocab)}
    pairs_tracker = PairPositions()

    # pre-tokenize the data
    pre_tokens_counter: Counter[bytes] = Counter()
    for pre_token in pre_tokenize(data, special_tokens):
        pre_tokens_counter[pre_token] += 1

    pre_tokens = list((list(bytes([t]) for t in token), count) for token, count in pre_tokens_counter.items())

    for pretoken_idx, (token, count) in enumerate(pre_tokens):
        for i in range(len(token) - 1):
            pairs_tracker.add((token[i], token[i + 1]), count, pretoken_idx)

    # merge most common pair in pre_tokens and count again until we have vocab_size
    merges = []
    while len(vocab) < vocab_size:
        (most_common_pair_left, most_common_pair_right), pretoken_idxs = pairs_tracker.most_common()
        merged_token = most_common_pair_left + most_common_pair_right
        vocab.append(merged_token)
        word_to_idx[merged_token] = len(word_to_idx)
        merges.append((most_common_pair_left, most_common_pair_right))

        # only look at pretokens where most common pair is in
        # most common pairs after merge must include currently merged token
        # so we could look for potential most common pairs in the same time
        for pretoken_idx in pretoken_idxs:
            old_pre_token, count = pre_tokens[pretoken_idx]
            new_pre_token = []

            i = 0
            while i < len(old_pre_token) - 1:
                if old_pre_token[i] == most_common_pair_left and old_pre_token[i + 1] == most_common_pair_right:
                    new_pre_token.append(merged_token)
                    i += 2
                else:
                    new_pre_token.append(old_pre_token[i])
                    i += 1
            if i == len(old_pre_token) - 1:
                new_pre_token.append(old_pre_token[i])

            pre_tokens[pretoken_idx] = new_pre_token, count

            all_pairs: set[tuple[bytes, bytes]] = set()
            old_pair_counts: Counter[tuple[bytes, bytes]] = Counter()
            new_pair_counts: Counter[tuple[bytes, bytes]] = Counter()

            for i in range(len(old_pre_token) - 1):
                pair = (old_pre_token[i], old_pre_token[i + 1])
                all_pairs.add(pair)
                old_pair_counts[pair] += 1
            for i in range(len(new_pre_token) - 1):
                pair = (new_pre_token[i], new_pre_token[i + 1])
                all_pairs.add(pair)
                new_pair_counts[pair] += 1

            for pair in all_pairs:
                delta = new_pair_counts[pair] - old_pair_counts[pair]
                if delta > 0:
                    pairs_tracker.add(pair, delta, pretoken_idx)
                elif delta < 0:
                    pairs_tracker.decrement(pair, -delta)
                    if delta == -old_pair_counts[pair]:
                        pairs_tracker.remove_pretoken_idx_from_pair(pair, pretoken_idx)

        pairs_tracker.delete((most_common_pair_left, most_common_pair_right))

    vocab = {i: token for i, token in enumerate(vocab)}
    return BpeModel(vocab, merges)


class PairPositions:
    _pair_to_count_indices: dict[tuple[bytes, bytes], tuple[int, set[int]]]  # pair to count and pretoken indices
    _count_to_pair: dict[int, set[tuple[bytes, bytes]]]
    _max_count: int
    _max_count_up_to_date: bool

    def __init__(self) -> None:
        self._pair_to_count_indices = {}
        self._count_to_pair = {}
        self._max_count = 0
        self._max_count_up_to_date = True

    def __len__(self):
        return len(self._pair_to_count_indices)

    def most_common(self) -> tuple[tuple[bytes, bytes], set[int]]:
        """
        Return the most common pair and pre-token ids where it's in.
        """
        self._recompute_max_count_if_necessary()
        assert self._max_count > 0, "most_common should only be called when there is already pairs here"
        candidate_pairs = iter(self._count_to_pair[self._max_count])
        best_pair = next(candidate_pairs)
        for candidate_pair in candidate_pairs:
            # lexicographically greater pair wins
            if self._is_right_greater_pair(best_pair, candidate_pair):
                best_pair = candidate_pair
        return best_pair, self._pair_to_count_indices[best_pair][1]

    def add(self, pair: tuple[bytes, bytes], count: int, pretoken_idx: int):
        assert count > 0
        if pair not in self._pair_to_count_indices:
            self._pair_to_count_indices[pair] = 0, set()
        prev_total, pretoken_ids = self._pair_to_count_indices[pair]
        curr_total = prev_total + count
        pretoken_ids.add(pretoken_idx)
        self._pair_to_count_indices[pair] = curr_total, pretoken_ids
        if prev_total in self._count_to_pair:
            self._count_to_pair[prev_total].remove(pair)
            if len(self._count_to_pair[prev_total]) == 0:
                del self._count_to_pair[prev_total]
        self._count_to_pair.setdefault(curr_total, set()).add(pair)
        if self._max_count_up_to_date:
            self._max_count = max(self._max_count, curr_total)

    def delete(self, pair: tuple[bytes, bytes]):
        count, _ = self._pair_to_count_indices[pair]
        del self._pair_to_count_indices[pair]
        assert pair in self._count_to_pair[count]
        self._count_to_pair[count].remove(pair)
        if len(self._count_to_pair[count]) == 0:
            del self._count_to_pair[count]
            if count == self._max_count:
                self._max_count_up_to_date = False

    def decrement(self, pair: tuple[bytes, bytes], count: int):
        old_count, pretoken_idx = self._pair_to_count_indices[pair]
        new_count = old_count - count
        assert new_count >= 0
        if new_count == 0:
            self.delete(pair)
        else:
            self._pair_to_count_indices[pair] = new_count, pretoken_idx
            self._count_to_pair[old_count].remove(pair)
            self._count_to_pair.setdefault(new_count, set()).add(pair)
            if len(self._count_to_pair[old_count]) == 0:
                del self._count_to_pair[old_count]
                if old_count == self._max_count:
                    self._max_count_up_to_date = False

    def remove_pretoken_idx_from_pair(self, pair: tuple[bytes, bytes], pretoken_idx: int):
        assert pretoken_idx in self._pair_to_count_indices[pair][1]
        self._pair_to_count_indices[pair][1].remove(pretoken_idx)

    def _recompute_max_count_if_necessary(self):
        if not self._max_count_up_to_date:
            if len(self._pair_to_count_indices) > 0:
                # scan the entire pairs to find maximum count for now, could use heap to keep track of second largest count
                self._max_count = max(self._count_to_pair.keys())
            else:
                self._max_count = 0

            self._max_count_up_to_date = True

    def _is_right_greater_pair(self, left: tuple[bytes, bytes], right: tuple[bytes, bytes]) -> bool:
        """ """
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

def pre_tokenize(input: bytes, special_tokens: list[str]) -> Generator[bytes]:
    """
    Split the input bytes into pre-tokens, each represented as a sequence of UTF-8 bytes.
    Pre-tokens do not include special tokens.
    """

    text = input.decode("utf-8")
    def split_on_special_tokens(text: str) -> Generator[str]: 
        """
        Return a generator text items splitted by special tokens. Those text items do not include special items themselves.
        """
        if len(special_tokens) == 0:
            yield text
            return
        
        special_token_escaped_patterns = [re.escape(t) for t in sorted(set(special_tokens), key=len, reverse=True)]
        special_token_pat = re.compile("|".join(special_token_escaped_patterns))

        window_start = 0 # start of a potential non-specialized-token split item
        for m in special_token_pat.finditer(text):
            if m.start() > window_start:
                yield text[window_start:m.start()]
            window_start = m.end()
        if window_start < len(text):
            yield text[window_start:]
        
    for item in split_on_special_tokens(text):
        for m in PRETOKENIZE_PATTERN.finditer(item):
            yield m.group(0).encode("utf-8")
