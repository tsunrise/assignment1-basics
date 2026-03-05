from collections.abc import Generator
from dataclasses import dataclass

import regex as re

def compile_special_token_patterns(special_tokens: list[str]) -> re.Pattern[str]:
    if len(special_tokens) == 0:
        return re.compile("^$") # nothing
    special_token_escaped_patterns = [re.escape(t) for t in sorted(set(special_tokens), key=len, reverse=True)]
    return re.compile("|".join(special_token_escaped_patterns))

def compile_special_token_patterns_binary(special_tokens: list[str]) -> re.Pattern[bytes]:
    if len(special_tokens) == 0:
        return re.compile(b"^$") # nothing
    special_token_escaped_patterns = [re.escape(t.encode("utf-8")) for t in sorted(set(special_tokens), key=len, reverse=True)]
    return re.compile(b"|".join(special_token_escaped_patterns))

PRETOKENIZE_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def split_into_pretokens(text: str, special_tokens: list[str]) -> Generator[tuple[int, int, bool]]:
    """
    Yields the token start offset (inclusive), token end offset (exclusive), and a boolean that is true iff it's a special token.
    """
    if len(special_tokens) == 0:
        yield from  _get_pretokens_from_text_without_specialized_tokens(text, 0)
        return

    special_token_pat = compile_special_token_patterns(special_tokens)

    window_start = 0  # start of a potential non-specialized-token split item
    for m in special_token_pat.finditer(text):
        if m.start() > window_start:
            yield from _get_pretokens_from_text_without_specialized_tokens(text[window_start:m.start()], window_start)
        yield m.start(), m.end(), True
        window_start = m.end()
    if window_start < len(text):
        yield from _get_pretokens_from_text_without_specialized_tokens(text[window_start:len(text)], window_start)

def _get_pretokens_from_text_without_specialized_tokens(text: str, starting_offset: int) -> Generator[tuple[int, int, bool]]:
    for m in PRETOKENIZE_PATTERN.finditer(text):
        yield m.start() + starting_offset, m.end() + starting_offset, False

@dataclass
class BpeParameters:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]