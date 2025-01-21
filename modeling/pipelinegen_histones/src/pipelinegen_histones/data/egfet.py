from histones_modeling.egfet.load_data import load_egfet_data

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch


class EGFETDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
