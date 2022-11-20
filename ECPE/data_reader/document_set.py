from torch.utils.data import Dataset
import numpy as np


class DocumentDataset(Dataset):
    def __init__(self, phase='train'):
        self.phase = phase


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def _load_data(self):
        data = np.load('../../data/corpus/all_data.npy')
        pairs = np.load('../../data/corpus/pairs.npy')

