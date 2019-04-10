from __future__ import print_function

import torch
import torch.optim
import torchvision
from torch import nn
from torch import optim
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.backends.cudnn.enabled = True
    # Datasets
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

    batch_size = 4
    epoch_size = 10

    trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, pin_memory=True, num_workers=6)
    testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, pin_memory=True, num_workers=6)

    # Models
    nets = {
        "rs18": torchvision.models.resnet18(pretrained=False).to(device),
        # "rs18_pretrain": torchvision.models.resnet18(pretrained=True).to(device)
        # "rs50" : torchvision.models.resnet50(pretrained=False).to(device),
        # "rs50_pretrain" : torchvision.models.resnet50(pretrained=True).to(device)
    }

    # Optimizers
    criterion = nn.CrossEntropyLoss()
    learning_rates = {0.001}

    optimizer = optim.SGD
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        for key, value in nets.items()
        for learning_rate in learning_rates
    }

    accuracy = {
        **{key + "_train": [] for key in nets},
        **{key + "_test": [] for key in nets}
    }

    train("rs18", nets["rs18"], optimizers["rs18"], criterion, trainLoader, trainDataset, accuracy, epoch_size)
    test("rs18", nets["rs18"], testLoader, testDataset, accuracy)


def train(key, model, optimizer, criterion, trainLoader, trainDataset, accuracy, epoch_size):
    key += "_train"
    model.train(mode=True)
    for epoch in range(epoch_size + 1):
        print('epoch : ', epoch)
        train_correct = 0.0

        for step, (x, y) in enumerate(trainLoader):
            if step % 2000 == 0 and step != 0:
                print('training step : ', step)
            x = x.to(device)
            y = y.to(device).long().view(-1)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            train_correct += (torch.max(y_hat, 1)[1] == y).sum().item()

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()
        accuracy[key] += [(train_correct * 100.0) / len(trainDataset)]
        print(key, 'Acc: ', accuracy.__getitem__(key)[epoch])
        print('')
        torch.save(model.state_dict(), key + '.pkl')


def test(key, model, testLoader, testDataset, accuracy):
    key += "_test"
    model.eval()
    test_correct = 0.0
    for step, (x, y) in enumerate(testLoader):
        if step % 2000 == 0 and step != 0:
            print('testing step : ', step)
        x = x.to(device)
        y = y.to(device).long().view(-1)

        y_hat = model(x)
        test_correct += (torch.max(y_hat, 1)[1] == y).sum().item()

    accuracy[key] += [(test_correct * 100.0) / len(testDataset)]

    print(key, 'Acc: ', accuracy.__getitem__(key)[epoch])
    print('')

if __name__ == '__main__':
    main()
