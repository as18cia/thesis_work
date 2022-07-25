import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        try:
            adfa = {}
            for key, val in self.encodings.items():
                adfa[key] = torch.tensor(val[idx])
            return adfa
        except:
            pass

    def __len__(self):
        return len(self.encodings.input_ids)
