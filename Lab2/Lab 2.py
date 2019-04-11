from __future__ import print_function

from functools import reduce

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim
from torch.cuda import device
from torch.utils.data import DataLoader

from Lab2.data.dataloader import read_bci_data


def showResult(title='', **kwargs):
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    for label, data in kwargs.items():
        plt.plot(range(len(data)), data, '--' if 'test' in label else '-', label=label)
    plt.ylim(0, 100)
    plt.xlim(0, 300)
    points = [(-5, 87), (310, 87)]
    (xpoints, ypoints) = zip(*points)

    plt.plot(xpoints, ypoints, linestyle='--', color='black')

    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.show()


def main():
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nets = {
        # "EEG_elu": EEGNet().to(device),
        "EEG_relu": EEGNet(nn.ReLU).to(device)
        # "EEG_leaky_relu": EEGNet(nn.LeakyReLU).to(device)
        # "DCN_elu": DeepConvNet().to(device),
        # "DCN_relu": DeepConvNet(nn.ReLU).to(device),
        # "DCN_leaky_relu": DeepConvNet(nn.LeakyReLU).to(device)
    }

    # Training setting
    loss_fn = nn.CrossEntropyLoss()
    # learning_rates = {0.025, 0.0018, 0.0018}
    # learning_rates = {0.0002, 0.0002, 0.0002}
    learning_rates = {0.0022}

    optimizer = torch.optim.Adam
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate)
        for key, value in nets.items()
        for learning_rate in learning_rates
    }

    epoch_size = 300
    batch_size = 64
    train(nets, epoch_size, batch_size, loss_fn, optimizers)


def train(nets, epoch_size, batch_size, loss_fn, optimizers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test = read_bci_data()
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    trainDataset = torch.utils.data.TensorDataset(x_train, y_train)
    testDataset = torch.utils.data.TensorDataset(x_test, y_test)

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True,
                                              pin_memory=True)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    accuracy = {
        **{key + "_train": [] for key in nets},
        **{key + "_test": [] for key in nets}
    }
    for epoch in range(epoch_size + 1):
        train_correct = {key: 0.0 for key in nets}
        test_correct = {key: 0.0 for key in nets}
        for step, (x, y) in enumerate(trainLoader):
            x = x.to(device)
            y = y.to(device).long().view(-1)

            for key, net in nets.items():
                net.train(mode=True)
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                train_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()

            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            for step, (x, y) in enumerate(testLoader):
                x = x.to(device)
                y = y.to(device).long().view(-1)
                for key, net in nets.items():
                    net.eval()
                    y_hat = net(x)
                    test_correct[key] += (torch.max(y_hat, 1)[1] == y).sum().item()

        for key, value in train_correct.items():
            accuracy[key + "_train"] += [(value * 100.0) / len(trainDataset)]

        for key, value in test_correct.items():
            accuracy[key + "_test"] += [(value * 100.0) / len(testDataset)]

        if epoch % 5 == 0:
            print('epoch : ', epoch, ' loss : ', loss.item())
            for key, value in accuracy.items():
                print(key, 'Acc: ', value[epoch])
            print('')
        torch.cuda.empty_cache()
    showResult(title='Activation function comparison(EEGNet)'.format(epoch + 1), **accuracy)


class EEGNet(nn.Module):
    def __init__(self, activation=None):
        if not activation:
            activation = nn.ELU
            activation.alpha = 1
        super(EEGNet, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.6)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.65)
        )

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


class DeepConvNet(nn.Module):
    def __init__(self, activation=None):
        if not activation:
            activation = nn.ELU
            activation.alpha = 1.0
        super(DeepConvNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

        flatten_size = 200 * reduce(lambda x, _: round((x - 4) / 2), [1, 1, 1, 1], 750)
        self.classify = nn.Sequential(
            nn.Linear(flatten_size, 2, bias=True),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x


if __name__ == '__main__':
    main()
