import torch
import numpy as np
from torch.utils.data import Dataset



class Dataset(Dataset):
    def __init__(self, train_input, train_input_LiDAR, train_label, train_sparse_label):
        self.train_input = train_input
        self.train_LiDAR = train_input_LiDAR
        self.train_label = train_label
        self.train_sparse_label = train_sparse_label
        

    def __len__(self):
        return len(self.train_input)

    def __getitem__(self, item):
        channel = torch.from_numpy(self.train_label[item])
        channel_real = channel.real
        channel_imag = channel.imag
        channel = torch.cat([channel_real, channel_imag], dim=0).squeeze(1)
        return self.train_input[item], self.train_LiDAR[item], channel, self.train_sparse_label[item]

class naiveDataset_CL(Dataset):
    def __init__(self, train_input, train_input_LiDAR, train_label, train_sparse_label, w):
        self.train_input = train_input
        self.train_LiDAR = train_input_LiDAR
        self.train_label = train_label
        self.train_sparse_label = train_sparse_label
        self.w = w
        

    def __len__(self):
        return len(self.train_input)

    def __getitem__(self, item):
        channel = torch.from_numpy(self.train_label[item])
        channel_real = channel.real
        channel_imag = channel.imag
        channel = torch.cat([channel_real, channel_imag], dim=0).squeeze(1)
        return self.train_input[item], self.train_LiDAR[item], channel, self.train_sparse_label[item], self.w[item]
    