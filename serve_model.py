#!/usr/bin/env python3

import os
import torch
from typing import List, Dict, Optional
from model import MyModel
from tokenizer import Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/mymodel.pt"

model = MyModel().to(DEVICE)
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
model.eval()

tok: Tokenizer = model.tokenizer  # same tokenizer used for training

def _ids_from_messages(messages: List[Dict[str, str]]) -> torch.LongTensor:
    # dataset.py: encode each message with BOS+...+EOS and concatenate
    ids: List[int] = []
    is_conversation = len(messages[0]['content']) < 100
    if is_conversation:
        print("messages:", messages)
    for m in messages:
        ids.extend(tok.encode(m["content"], add_bos=True, add_eos=True))
    # start assistant reply with a BOS
    ids.append(tok.BOS)
    if is_conversation:
        print("ids:", ids)
    return torch.tensor(ids, dtype=torch.long)

@torch.no_grad()
def _generate_from_ids(prompt_ids: torch.LongTensor, max_new_tokens: int = 128, temperature: float = 1.0) -> List[int]:
    x = prompt_ids.to(DEVICE).unsqueeze(0)  # [1, T]
    start = x.size(1)
    for _ in range(max_new_tokens):
        x_cut = x[:, -model.max_len :]
        logits = model(x_cut)[:, -1, :]  # [1, V]
        if temperature and temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        next_id = int(torch.argmax(logits, dim=-1))
        x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == tok.EOS:
            break
    return x[0, start:].tolist()

def generate(messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    prompt_ids = _ids_from_messages(messages)
    gen_ids = _generate_from_ids(
        prompt_ids,
        max_new_tokens=int(max_tokens) if max_tokens is not None else 128,
        temperature=float(temperature) if temperature is not None else 1.0,
    )
    # return clean text without special tokens
    return tok.decode(gen_ids, skip_special=True)
