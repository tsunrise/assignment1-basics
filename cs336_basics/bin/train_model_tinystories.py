import wandb
from cs336_basics.bpe.tokenizer import BpeTokenizer
from cs336_basics.modules.transformer import TransformerLM
from cs336_basics.adamw import AdamW
import yaml
from cs336_basics.training import load_checkpoint, get_batch, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
import numpy as np
import os
import torch

TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKENS_PATH = "artifacts/bpe/tinystories/traindata_tokens"
VAL_TOKENS_PATH = "artifacts/bpe/tinystories/valdata_tokens"
CHECKPOINT_DIR_PATH = "artifacts/checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR_PATH + "/tinystories.ckpt"
TOKENIZER_PATH = "artifacts/bpe/tinystories"


def load_token_data(base_path: str) -> np.memmap:
    with open(base_path + ".count.txt") as f:
        num_tokens = int(f.read().strip())
    return np.memmap(base_path + ".bin", dtype=np.uint16, mode="r", shape=(num_tokens,))


def main(plan_path: str):
    """
    - `val_interval`: perform one validation step after `val_interval` training step
    """

    with open(plan_path) as f:
        config = yaml.safe_load(f)["parameters"]

    wdb = wandb.init(
        entity="tomshen",
        project="cs336-a1-tinystories",
        config=config,
    )

    tokenizer = BpeTokenizer.from_files(
        TOKENIZER_PATH + "/vocab.json", TOKENIZER_PATH + "/merges.txt", special_tokens=["<|endoftext|>"]
    )

    model = TransformerLM(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        num_layers=config["num_layers"],
    ).to(config["device"])
    optimizer = AdamW(
        model.parameters(), config["lr"], config["weight_decay"], (config["beta1"], config["beta2"]), config["eps"]
    )

    step = 0
    if os.path.exists(CHECKPOINT_PATH):
        step = load_checkpoint(CHECKPOINT_PATH, model, optimizer, map_location=config["device"])

    # load training and validation data
    data_train = load_token_data(TRAIN_TOKENS_PATH)
    data_val = load_token_data(VAL_TOKENS_PATH)

    while step < config["total_steps"]:
        train_input, train_target = get_batch(
            data_train, config["batch_size"], config["context_length"], config["device"]
        )

        optimizer.zero_grad()

        model.train()
        train_logits = model(train_input)  # (batch, tokens, vocabs)
        train_loss = cross_entropy(train_logits, train_target)
        train_loss.backward()

        optimizer.step()

        if step % config["val_interval"] == 0 and step > 0:
            val_input, val_target = get_batch(
                data_val, config["batch_size"], config["context_length"], config["device"]
            )
            with torch.no_grad():
                model.eval()
                val_logits = model(val_input)
                val_loss = cross_entropy(val_logits, val_target)
                wdb.log({"val_loss": val_loss.item()}, step=step)

        wdb.log({"train_loss": train_loss.item()}, step=step, commit=True)
        step += 1

        if step % config["checkpoint_save_intervals"] == 0:
            os.makedirs(CHECKPOINT_DIR_PATH, exist_ok=True)
            save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)
