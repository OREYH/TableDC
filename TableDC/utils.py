import numpy as np
import os

import torch
from torch.utils.data import Dataset



class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        label_path = 'data/label.txt'
        if os.path.exists(label_path):
            self.y = np.loadtxt(label_path, dtype=int)
        else:
            self.y = None
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(np.array(self.x[idx]))
        if self.y is not None:
            return sample, torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx))
        else:
            return sample, torch.tensor(-1), torch.from_numpy(np.array(idx))


