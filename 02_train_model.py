#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DS, collate_fn

STEPS = 2000
BATCH_SIZE = 8
D_MODEL = 64
LR = 1e-3
SAVE_DIR = "checkpoints"
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))  # [B, T, V]

if __name__ == "__main__":
    torch.manual_seed(SEED)
    ds = DS("data/chats.json")
    vocab_size = ds.tokenizer.vocab_size
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = TinyLM(vocab_size, D_MODEL).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    model.train()
    while step < STEPS:
        for x, y in loader:
            step += 1
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), y.reshape(B * T))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            #if step % 50 == 0 or step == 1 or step == STEPS:
            print(f"step={step} loss={loss.item():.6f}")
            if step >= STEPS:
                break

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "tinylm.pt")
    torch.save(
        {"model_state": model.state_dict(),
         "config": {"vocab_size": vocab_size, "d_model": D_MODEL, "ignore_index": -100}},
        path,
    )
    print(f"saved to {path}")
