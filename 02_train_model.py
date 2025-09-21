#!/usr/bin/env python3

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import DS, collate_fn
from model import MyModel

STEPS = 1200
BATCH_SIZE = 64
LR = 1e-4
SAVE_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

if __name__ == "__main__":
    torch.manual_seed(SEED)

    ds = DS("data/chats.json")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = MyModel().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    model.train()
    while step < STEPS:
        for x, y in loader:
            step += 1
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)  # [B, T, V]
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            print(f"step={step} loss={loss.item():.6f}")
            if step >= STEPS:
                break

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, "mymodel.pt")
    torch.save({"model_state": model.state_dict()}, path)
    print(f"saved to {path}")
