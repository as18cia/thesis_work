import torch
from torch.utils.data import Dataset

from training.tokinizing_and_train.tokinizing import tokenize


class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)




if __name__ == '__main__':
    train_encoding, eval_encoding = tokenize()
    s = MyDataset(train_encoding)
    print(s[0])