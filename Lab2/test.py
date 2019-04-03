from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import *

from Lab2.data.dataloader import read_bci_data


def show_data(data):
    if len(data.shape) == 3:
        data = data[0]

    if len(data.shape) != 2:
        raise AttributeError("shape no ok")
        return

    plt.figure(figsize=(10, 4))
    for i in range(data.shape[0]):
        plt.subplot(2, 1, i + 1)
        plt.ylabel("Channel " + str(i + 1), fontsize=15)
        plt.plot(np.array(data[i, :]))
        plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = read_bci_data()
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    # show_data(x_train[0][0])

    # NN
    net = EGGNet(nn.ELU).to(device)

    # Training setting
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training
    for i in range(1000):
        y_hat = net(x_train.to(device))
        loss = loss_fn(y_hat, y_train.to(device).long())
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class EGGNet(nn.Module):
    def __init__(self, activation=None):
        if not activation:
            activation = nn.ELU

        super(EGGNet, self).__init__()
        # firstconv
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # depthwiseConv
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        # separableConv
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        # classify
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x


if __name__ == '__main__':
    main()
