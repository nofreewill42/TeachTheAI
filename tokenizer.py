from dataclasses import dataclass
from typing import List, Iterable

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

# Basic printable ASCII set + newline and tab
_ASCII_CHARS = (
    ["\t", "\n", "\r"]
    + [chr(i) for i in range(32, 127)]  # 32..126
)

@dataclass
class TokenizerConfig:
    add_bos: bool = True
    add_eos: bool = True

class Tokenizer:
    """
    Simple ASCII character tokenizer with fixed vocabulary.
    id 0..3 are reserved for special tokens: PAD, BOS, EOS, UNK
    ASCII chars start from index 4.
    """

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        # Build vocab
        self.id_to_token: List[str] = list(SPECIAL_TOKENS) + list(_ASCII_CHARS)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

        self.PAD = self.token_to_id["<PAD>"]
        self.BOS = self.token_to_id["<BOS>"]
        self.EOS = self.token_to_id["<EOS>"]
        self.UNK = self.token_to_id["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(
        self,
        text: str,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> List[int]:
        add_bos = self.config.add_bos if add_bos is None else add_bos
        add_eos = self.config.add_eos if add_eos is None else add_eos

        ids: List[int] = []
        if add_bos:
            ids.append(self.BOS)
        for ch in text:
            tok = ch
            ids.append(self.token_to_id.get(tok, self.UNK))
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out_chars: List[str] = []
        for i in ids:
            if i < 0 or i >= self.vocab_size:
                continue
            tok = self.id_to_token[i]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out_chars.append(tok)
        return "".join(out_chars)

if __name__ == "__main__":
    # Quick usage demo
    tok = Tokenizer()
    print("Vocab size:", tok.vocab_size)

    sample = "Hello\tworld!\nThis is a test."
    print("\n--- Basic encode/decode ---")
    ids = tok.encode(sample)  # defaults add BOS and EOS
    print("Input text:", repr(sample))
    print("Encoded ids (head [:32]):", ids[:32], "... len =", len(ids))
    print("Decoded:", repr(tok.decode(ids)))

    print("\n--- Without BOS/EOS ---")
    ids_no_special = tok.encode(sample, add_bos=False, add_eos=False)
    print("Encoded ids (no BOS/EOS) head:", ids_no_special[:32], "... len =", len(ids_no_special))
    print("Decoded:", repr(tok.decode(ids_no_special)))

    print("\n--- Unknown characters become <UNK> ---")
    tricky = "naïve café 🙂"
    ids_tricky = tok.encode(tricky)  # non ASCII chars map to <UNK>
    print("Input text:", repr(tricky))
    print("Encoded ids:", ids_tricky)
    print("Decoded skip_special=True:", repr(tok.decode(ids_tricky, skip_special=True)))
    print("Decoded skip_special=False:", repr(tok.decode(ids_tricky, skip_special=False)))
