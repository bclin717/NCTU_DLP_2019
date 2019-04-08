from __future__ import print_function

import torch
import torch.optim
from torch.cuda import device
from torch.utils.data import DataLoader
from torchvision import transforms

from Lab3.data.dataloader import *

transformTraining = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

transformTesting = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

batch_size = 64
epoch_size = 10

def main():
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainDataset = RetinopathyLoader(
        '/home/kevin/PycharmProjects/DLP Assignments/Lab3/data/data/',
        'train',
        transformTraining
    )

    testDataset = RetinopathyLoader(
        '/home/kevin/PycharmProjects/DLP Assignments/Lab3/data/data/',
        'test',
        transformTesting
    )

    trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True, pin_memory=True)


if __name__ == '__main__':
    main()
