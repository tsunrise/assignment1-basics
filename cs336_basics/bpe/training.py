from collections import Counter
import os
from typing import BinaryIO
import regex as re
from collections.abc import Generator
import multiprocessing

from cs336_basics.bpe.tokenizer import BpeTokenizer
from cs336_basics.bpe.utils import compile_special_token_patterns, split_on_special_tokens

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
) -> BpeTokenizer:
    """
    Train a BPE model.

    - `input_path`: Path to a text file with BPE tokenizer training data.
    - `vocab_size`: A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    - `special_tokens`: A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.
    """

    # read the file as bytes and get pre-tokens
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # initialize vocab as byte 0 to 255
    vocab = [bytes([i]) for i in range(256)]
    for token in special_tokens:
        vocab.append(token.encode("utf-8"))
    pairs_tracker = PairPositions()

    # pre-tokenize the data
    print("BPE training: making pre-tokens")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        chunk_counters = pool.starmap(
            _pre_tokenize_worker_fn,
            (
                (input_path, special_tokens, start_offset, end_offset)
                for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:])
            ),
        )
    pre_tokens_counter = Counter()
    for chunk_counter in chunk_counters:
        for pretoken, count in chunk_counter.items():
            pre_tokens_counter[pretoken] += count

    pre_tokens = list((list(bytes([t]) for t in token), count) for token, count in pre_tokens_counter.items())

    print("Merging Pairs")
    for pretoken_idx, (token, count) in enumerate(pre_tokens):
        for i in range(len(token) - 1):
            pairs_tracker.add((token[i], token[i + 1]), count, pretoken_idx)

    # merge most common pair in pre_tokens and count again until we have vocab_size
    merges = []
    while len(vocab) < vocab_size and len(pairs_tracker) > 0:
        (most_common_pair_left, most_common_pair_right), pretoken_idxs = pairs_tracker.most_common()
        merged_token = most_common_pair_left + most_common_pair_right
        vocab.append(merged_token)
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
                    pairs_tracker.add(pair, delta * count, pretoken_idx)
                elif delta < 0:
                    pairs_tracker.decrement(pair, -delta * count)
                    if delta == -old_pair_counts[pair] and pairs_tracker.count(pair) > 0:
                        # that means the pair is still in some other pretokens, but no longer in this pretoken.
                        # remove this pretoken index from pair for tracking
                        pairs_tracker.remove_pretoken_idx_from_pair(pair, pretoken_idx)

        if pairs_tracker.count((most_common_pair_left, most_common_pair_right)) > 0:
            pairs_tracker.delete((most_common_pair_left, most_common_pair_right))

    vocab = {i: token for i, token in enumerate(vocab)}
    return BpeTokenizer(vocab, merges, special_tokens)

def _pre_tokenize_worker_fn(input_path: str, special_tokens: list[str], start_offset: int, end_offset: int):
    chunk_counter: Counter[bytes] = Counter()
    with open(input_path, "rb") as f:
        f.seek(start_offset)
        chunk = f.read(end_offset - start_offset)
        for pre_token in _pre_tokenize(chunk, special_tokens):
            chunk_counter[pre_token] += 1
    return chunk_counter

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

    def most_common(self) -> tuple[tuple[bytes, bytes], list[int]]:
        """
        Return the most common pair and pre-token ids where it's in.
        """
        self._recompute_max_count_if_necessary()
        assert self._max_count > 0, "most_common should only be called when there is already pairs here"
        candidate_pairs = iter(self._count_to_pair[self._max_count])
        best_pair = next(candidate_pairs)
        for candidate_pair in candidate_pairs:
            # lexicographically greater pair wins
            if candidate_pair > best_pair:
                best_pair = candidate_pair
        return best_pair, list(self._pair_to_count_indices[best_pair][1])

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

    def count(self, pair: tuple[bytes, bytes]) -> int:
        result = self._pair_to_count_indices.get(pair)
        if result is not None:
            return result[0]
        return 0

    def _recompute_max_count_if_necessary(self):
        if not self._max_count_up_to_date:
            if len(self._pair_to_count_indices) > 0:
                # scan the entire pairs to find maximum count for now, could use heap to keep track of second largest count
                self._max_count = max(self._count_to_pair.keys())
            else:
                self._max_count = 0

            self._max_count_up_to_date = True


PRETOKENIZE_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _pre_tokenize(input: bytes, special_tokens: list[str]) -> Generator[bytes]:
    """
    Split the input bytes into pre-tokens, each represented as a sequence of UTF-8 bytes.
    Pre-tokens do not include special tokens.
    """

    text = input.decode("utf-8", errors="ignore")

    for lo, hi, is_special in split_on_special_tokens(text, special_tokens):
        if is_special: 
            continue
        for m in PRETOKENIZE_PATTERN.finditer(text[lo:hi]):
            yield m.group(0).encode("utf-8")


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
