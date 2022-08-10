import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)



if __name__ == '__main__':

    print("{:.2f}".format(0.2345))