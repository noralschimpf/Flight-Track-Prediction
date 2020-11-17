import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_name = os.listdir(self.root_dir)
        self.file_name.sort()
        print(self.file_name)
        self.path = []
        for i in range(len(self.file_name)):
            self.path.append(os.path.join(self.root_dir, self.file_name[i]))

    def __len__(self):
      return len(self.path)

    def __getitem__(self, idx):
        sample = np.loadtxt(self.path[idx], dtype='float', delimiter=',', usecols=(1, 2))
        if self.transform:
            sample = self.transform(sample)
        # print(idx, sample)
        return sample
        # for i in range(len(self.path)):
        #     print(self.path[i])
        #     sample = np.loadtxt(self.path[i], dtype='float', delimiter=',', usecols=(2, 3, 4))
        #     print(i, sample)




class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
