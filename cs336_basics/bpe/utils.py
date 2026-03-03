from collections.abc import Generator

import regex as re

def compile_special_token_patterns(special_tokens: list[str]) -> re.Pattern:
    special_token_escaped_patterns = [re.escape(t) for t in sorted(set(special_tokens), key=len, reverse=True)]
    return re.compile("|".join(special_token_escaped_patterns))

def split_on_special_tokens(text: str, special_tokens: list[str]) -> Generator[tuple[int, int, bool]]:
    """
    Yields the token start offset (inclusive), token end offset (exclusive), and a boolean that is true iff it's a special token.
    """
    if len(special_tokens) == 0:
        yield 0, len(text), False
        return

    special_token_pat = compile_special_token_patterns(special_tokens)

    window_start = 0  # start of a potential non-specialized-token split item
    for m in special_token_pat.finditer(text):
        if m.start() > window_start:
            yield window_start ,m.start(), False
        yield m.start(), m.end(), True
        window_start = m.end()
    if window_start < len(text):
        yield window_start, len(text), False
