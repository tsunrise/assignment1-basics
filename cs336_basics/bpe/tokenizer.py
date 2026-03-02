from collections.abc import Generator, Iterable


class BpeTokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()
    
    @classmethod
    def to_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        raise NotImplementedError()
    
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError()
    
    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int]:
        raise NotImplementedError
    
