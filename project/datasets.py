import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]


class HybridDataset(Dataset):
    def __init__(self, X_seq, X_feat, y):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.X_feat = torch.from_numpy(X_feat).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_feat[idx], self.y[idx]