from cs336_basics.bpe.training import train_bpe
from pathlib import Path


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

if __name__ == "__main__":
    main()
