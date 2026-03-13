from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache

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
    special_tokens: list[str]

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d