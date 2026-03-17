from typing import cast

import toml
import torch

from cs336_basics.bpe.tokenizer import BpeTokenizer
from cs336_basics.modules.transformer import TransformerLM
TOKENIZER_PATH = "artifacts/bpe/tinystories"
CHECKPOINT_DIR_PATH = "artifacts/checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR_PATH + "/tinystories.ckpt"
PLAN_PATH = "training_parameters/tinystories/mac.toml"

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 5):
    """
    logits: (...,vocab)
    return: (...,)
    """
    logits, idx = torch.topk(logits, top_k, dim=-1) # (...,k), (...,k)
    logits = logits / temperature
    probs = torch.softmax(logits, 0)
    sampled = torch.multinomial(probs, num_samples=1) # (...,1)
    nxt_token = idx.gather(-1, sampled) # (...,1)
    nxt_token = nxt_token.squeeze(dim=-1) # (...)
    return nxt_token
    
    

def sample(prefix: str, temperature: float = 1.0, top_k: int = 5, max_tokens: int = 200):
    
    with open(PLAN_PATH) as f:
        config = toml.load(f)["parameters"]
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
    
    tokenizer = BpeTokenizer.from_files(
        TOKENIZER_PATH + "/vocab.json", TOKENIZER_PATH + "/merges.txt", special_tokens=["<|endoftext|>"]
    )
    e2t_token = tokenizer.word_to_token[b"<|endoftext|>"]
    if config["device"] == "mps":
        model = cast(TransformerLM, torch.compile(model, backend="aot_eager"))
    else:
        model = cast(TransformerLM, torch.compile(model))
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=config["device"])["model"])
    prefix_tokens = torch.tensor(tokenizer.encode(prefix), device=config["device"], dtype=torch.long) # (tokens,)
    logits, kv = model.prefill(prefix_tokens) # logits: (tokens, vocabs)
    nxt_token = sample_next_token(logits[-1,...], temperature, top_k) # ()
    print(prefix,end="")
    print(tokenizer.decode([int(nxt_token.item())]), end = "")
    
    for _ in range(max_tokens - 1):
        logits, kv = model.decode_step(kv, nxt_token) # (vocabs,)
        nxt_token = sample_next_token(logits, temperature, top_k) # ()
        nxt_token_id = int(nxt_token.item())
        print(tokenizer.decode([nxt_token_id]), end = "")
        if nxt_token_id == e2t_token:
            break
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    sample(prefix=args.prefix)
