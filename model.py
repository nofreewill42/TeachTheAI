# model.py
from torch import nn
import torch
from tokenizer import Tokenizer

class _Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + h
        h = self.ln2(x)
        h = self.mlp(h)
        return x + h

class MyModel(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 4, n_layers: int = 6, max_len: int = 512):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.PAD = self.tokenizer.PAD
        self.BOS = self.tokenizer.BOS
        self.EOS = self.tokenizer.EOS

        self.tok = nn.Embedding(self.vocab_size, d_model, padding_idx=self.PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.max_len = max_len

    def _causal_mask(self, T, device):
        # shape [T, T] with True where future positions are masked
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos)
        mask = self._causal_mask(T, input_ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)  # [B, T, V]

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 16, temperature: float = 1.0) -> str:
        self.eval()
        device = next(self.parameters()).device
        # add BOS, no EOS on input
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        ids += [self.BOS]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            x_cut = x[:, -self.max_len :]
            logits = self(x_cut)[:, -1, :]  # [1, V]
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            next_id = int(torch.argmax(logits, dim=-1))
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == self.EOS:
                break

        tokens = x[0].tolist()
        return self.tokenizer.decode(tokens, skip_special=True)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MyModel().to(device)

    ckpt = torch.load("checkpoints/mymodel.pt", map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    model.eval()

    prompt = "nagyon "
    print("prompt:", prompt)
    print(model.generate(prompt))
