from collections.abc import Generator, Iterable
from cs336_basics.bpe.utils import BpeParameters, compile_special_token_patterns, split_into_pretokens
from collections import OrderedDict, deque

class BpeTokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.word_to_token = {b: i for i, b in vocab.items()}
        self.merges_order = {(self.word_to_token[left], self.word_to_token[right]): i for i, (left, right) in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pretoken_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.pretoken_cache_max_mize = 4096

    @classmethod
    def from_bpe_parameters(cls, params: BpeParameters):
        return cls(params.vocab, params.merges)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()

    def to_files(self, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_chunk(text))

    def encode_chunk(self, text: str) -> Generator[int]:
        # pre tokenize the text
        pretokens: list[tuple[int, ...]] = [] # each pretoken is represented by a tuple of token id in `self.vocab`
        pretoken_to_idx: dict[str, int] = {} # token string -> pretokens id
        text_in_pretokens_indices = [] # text represented by an ordered sequence of pretokens
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
                        left, right = pretoken[i], pretoken[i+1]
                        if (left, right) in self.merges_order:
                            if best_i is None or self.merges_order[(left, right)] < self.merges_order[(pretoken[best_i], pretoken[best_i + 1])]:
                                best_i = i
                    # if no pair to merge, we are done!
                    if best_i is None:
                        break

                    new_pretoken: list[int] = []
                    new_pretoken.extend(pretoken[j] for j in range(best_i))
                    new_pretoken.append(self.word_to_token[self.vocab[pretoken[best_i]] + self.vocab[pretoken[best_i + 1]]])
                    new_pretoken.extend(pretoken[j] for j in range(best_i + 2, len(pretoken)))
                    pretoken = tuple(new_pretoken)
                pretokens[pretoken_idx] = pretoken
                self.pretoken_cache[pretokens_text[pretoken_idx]] = pretoken
                if len(self.pretoken_cache) > self.pretoken_cache_max_mize:
                    self.pretoken_cache.popitem(last=False)
            else:
                self.pretoken_cache.move_to_end(pretokens_text[pretoken_idx],last=True)

        for pretoken_idx in text_in_pretokens_indices:
            yield from pretokens[pretoken_idx]

    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int]:
        special_token_patterns = compile_special_token_patterns(self.special_tokens)
        buf = deque()
        buf_release_threshold = 4 * 4096 * 4096 # when buffer size >= this threshold, 
                                                # release the buffer before the beginning of next special token 
                                                # that does not overlap on two string
                                                # TODO: we could randomize this threshold to prevent malicious dataset putting all special token
                                                # at boundaries, causing OOM
        stream = iter(iterable)
        while True:
            nxt = None
            # accumulate buffer until reaching release threshold
            while len(buf) < buf_release_threshold:
                nxt = next(stream, None)
                if nxt is None:
                    break
                buf.append(nxt)

            # continue accumulate buffer until we see a special token that does not overlap in boundaries
            # TODO: we could further optimize this so that it accumulate until we see a special token that may overlap in boundaries
            while True:
                nxt = next(stream, None)
                if nxt is None:
                    break
                m = next(special_token_patterns.finditer(nxt), None)
                if m is not None:
                    buf.append(nxt[:m])
                    # release buffer
                    yield from self.encode_chunk("".join(buf))
                    buf.clear()
                    buf.append(nxt[m:])
            
            if nxt is None:
                break
        
        if len(buf) > 0:
            yield from self.encode_chunk("".join(buf))
            buf.clear()
        

                
            

            

