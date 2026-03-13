import wandb
from cs336_basics.bpe.tokenizer import BpeTokenizer
from cs336_basics.modules.transformer import TransformerLM
from cs336_basics.adamw import AdamW
import yaml
from training import load_checkpoint, save_checkpoint
import numpy as np

TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
TRAIN_TOKENS_PATH = "artifacts/bpe/tinystories/traindata_tokens"
VAL_TOKENS_PATH = "artifacts/bpe/tinystories/valdata_tokens"
CHECKPOINT_PATH = "artifacts/checkpoints/tinystories.ckpt"
import os

def main(plan_path: str):
    """
    - `val_interval`: perform one validation step after `val_interval` training step
    """

    with open(plan_path) as f:
        config = yaml.safe_load(f)

    run = wandb.init(
        entity="tomshen",
        project="cs336-a1-tinystories",
        config=config,
    )

    tokenizer = BpeTokenizer.from_files(TOKENIZER_PATH + "/vocab.json", TOKENIZER_PATH + "/merges.txt")

    model = TransformerLM(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        vocab_size=len(tokenizer.vocab),
        context_length=config["context_length"],
        num_layers=config["num_layers"],
    )
    optimizer = AdamW(
        model.parameters(), config["lr"], config["weight_decay"], (config["beta1"], config["beta2"]), config["eps"]
    )

    steps = 0
    if os.path.exists(CHECKPOINT_PATH):
        steps = load_checkpoint(CHECKPOINT_PATH, model, optimizer, map_location=config["device"])
    
    # load training and validation data 
    
    tokens_train = np.memmap()

    while steps < config["total_steps"]:
        ...
        
        
