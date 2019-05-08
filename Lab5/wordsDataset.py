import numpy as np
import torch
from torch.utils.data import Dataset

from Lab5.CharDict import CharDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class wordsDataset(Dataset):
    def __init__(self, train=True):
        if train:
            f = './data/train.txt'
        else:
            f = './data/test.txt'
        self.datas = np.loadtxt(f, dtype=np.str)

        if train:
            self.datas = self.datas.reshape(-1)
        else:
            '''
            sp -> p
            sp -> pg
            sp -> tp
            sp -> tp
            p  -> tp
            sp -> pg
            p  -> sp
            pg -> sp
            pg -> p
            pg -> tp
            '''
            self.targets = np.array([
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1],
            ])

        self.tenses = [
            'simple-present',
            'third-person',
            'present-progressive',
            'simple-past'
        ]
        self.chardict = CharDict()
        self.train = train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.train:
            c = index % len(self.tenses)
            return self.chardict.longtensorFromString(self.datas[index]), c
        else:
            i = self.chardict.longtensorFromString(self.datas[index, 0])
            ci = self.targets[index, 0]
            o = self.chardict.longtensorFromString(self.datas[index, 1])
            co = self.targets[index, 1]
            return i, ci, o, co
