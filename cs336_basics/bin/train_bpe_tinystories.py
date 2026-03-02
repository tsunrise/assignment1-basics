from cs336_basics.bpe.training import train_bpe
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"

    model = train_bpe(
        input_path=str(input_path),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    model.save(str(vocab_path), str(merges_path))

    print(f"vocab size: {len(model.vocab)}")
    print(f"num merges: {len(model.merges)}")
    print(f"saved vocab: {vocab_path}")
    print(f"saved merges: {merges_path}")

if __name__ == "__main__":
    main()
