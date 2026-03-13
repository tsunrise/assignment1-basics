import numpy as np
from cs336_basics.bpe.tokenizer import BpeTokenizer
from tqdm import tqdm

TOKENIZER_PATH = "artifacts/bpe/tinystories"
TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKENS_PATH = "artifacts/bpe/tinystories/traindata_tokens"
VAL_TOKENS_PATH = "artifacts/bpe/tinystories/valdata_tokens"

def save_dataset(dataset_path: str, out_path: str):
    tk = BpeTokenizer.from_files(TOKENIZER_PATH + "/vocab.json", TOKENIZER_PATH + "/merges.txt", special_tokens=["<|endoftext|>"])
    # count tokens
    print(f"Count tokens for {dataset_path}")
    def file_data():
        with open(dataset_path, encoding="utf-8") as f:
            while True:
                chunk = f.read(4 * 1024 * 1024) # 4MB
                if not chunk:
                    break
                yield chunk
    num_tokens = sum(1 for _ in tk.encode_iterable(tqdm(file_data())))
    print(f"Saving {num_tokens} tokens")
    with open(out_path + ".count.txt", "w+") as t:
        t.write(f"{num_tokens}\n")
    arr = np.memmap(out_path + ".bin", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    tq = tqdm(unit="tk", total=num_tokens)
    for i, t in enumerate(tk.encode_iterable(file_data())):
        arr[i] = t
        tq.update()
    arr.flush()
    print("Done")

def main():
    save_dataset(VAL_PATH, VAL_TOKENS_PATH)
    save_dataset(TRAIN_PATH, TRAIN_TOKENS_PATH)


if __name__ == "__main__":
    main()
