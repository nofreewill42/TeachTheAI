# dataset.py
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer

class DS(Dataset):
    """
    Reads data/chats.json shaped like:
      [ [ {role, message, train}, ... ], [ ... ], ... ]
    Returns 
    """

    def __init__(self, path="data/chats.json"):
        self.convs = json.loads(Path(path).read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer()  # TODO: add_bos=True, add_eos=True without the config - having a config makes it harder to just use it however I want to

    def __len__(self):
        return len(self.convs)

    def __getitem__(self, idx):
        conv = self.convs[idx]
        input_ids_list = []
        target_ids_list = []
        for m in conv:
            tokens = self.tokenizer.encode(m["message"])
            input_ids = tokens[:-1]
            if m["train"]:
                target_ids = tokens[1:]
            else:
                # -100 for all the input_ids as their target output
                target_ids = [-100] * len(input_ids)
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
        input_ids = torch.cat([torch.tensor(ids, dtype=torch.long) for ids in input_ids_list])
        target_ids = torch.cat([torch.tensor(ids, dtype=torch.long) for ids in target_ids_list])
        return input_ids, target_ids

def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
    return input_ids, target_ids


if __name__ == "__main__":
    ds = DS("data/chats.json")
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for input_ids, target_ids in dl:
        print(input_ids.shape, target_ids.shape)
        print(input_ids)
        print(target_ids)
        break
        """
        torch.Size([2, 34]) torch.Size([2, 34])
    tensor([[ 1, 79, 72, 83, 86,  1, 79, 72,  7, 83, 86,  1, 79, 76, 83, 86,  1, 79,
            76, 83, 86,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1, 79, 72, 83, 86,  1, 79, 72, 83, 86,  1, 85, 72, 78, 96, 86, 85,  7,
            81,  3,  7,  8,  1, 85, 72, 78, 96, 86, 85,  7, 81,  3,  7,  8]])
    tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100,   79,   76,   83,   86,    2, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100,   79,   72,   83,   86,    2, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,   85,   72,
            78,   96,   86,   85,    7,   81,    3,    7,    8,    2]])
        """