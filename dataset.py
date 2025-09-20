# dataset.py
import json
from pathlib import Path
from torch.utils.data import Dataset

class DS(Dataset):
    """
    Reads data/chats.json shaped like:
      [ [ {role, message, train}, ... ], [ ... ], ... ]
    Returns one whole conversation per index: DS[i] -> list[dict]
    """

    def __init__(self, path="data/chats.json"):
        self.convs = json.loads(Path(path).read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.convs)

    def __getitem__(self, idx):
        return self.convs[idx]


if __name__ == "__main__":
    ds = DS("data/chats.json")
    print("conversations:", len(ds))
    if len(ds):
        conv = ds[0]
        print("conv[0] turns:", len(conv))
        for j, m in enumerate(conv):
            role = m.get("role")
            train = m.get("train")
            msg = m.get("message", "").replace("\n", "\\n")
            print(f"{j:02d} | role={role} | train={train} | msg={msg}")

    # Optional - quick DataLoader smoke test that keeps conversations intact
    # from torch.utils.data import DataLoader
    # dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=lambda batch: batch)
    # first_batch = next(iter(dl))
    # print("batch size:", len(first_batch), "| conv0 turns:", len(first_batch[0]))
