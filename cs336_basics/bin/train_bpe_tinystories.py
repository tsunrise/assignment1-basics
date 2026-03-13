from cs336_basics.bpe.training import train_bpe
from cs336_basics.bpe.tokenizer import BpeTokenizer
from pathlib import Path
import os

TRAIN_TOKENS_PATH = "artifacts/bpe/tinystories/traindata_tokens.npy"
VAL_TOKENS_PATH = "artifacts/bpe/tinystories/valdata_tokens.npy"

def main():
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"

    model = train_bpe(
        input_path=str(input_path),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=8
    )
    print(f"vocab size: {len(model.vocab)}")
    print(f"num merges: {len(model.merges)}")

    tokenizer = BpeTokenizer.from_bpe_parameters(model)
    directory = "artifacts/bpe/tinystories"
    os.makedirs(directory, exist_ok=True)
    tokenizer.to_files(directory + "/vocab.json", directory + "/merges.txt")
    print(f"Saved to {directory}")

if __name__ == "__main__":
    main()
