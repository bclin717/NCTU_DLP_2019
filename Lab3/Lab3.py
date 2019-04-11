from __future__ import print_function

import torch
import torch.optim
import torchvision
from data.dataloader import *
from torch import nn
from torch import optim
from torch.cuda import device
from torch.utils.data import DataLoader
from torchvision import transforms

# from Lab3.data.dataloader import *

transformTraining = transforms.Compose([
    transforms.ToTensor()
])

transformTesting = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataPath = '/home/kevin/PycharmProjects/DLP Assignments/Lab3/data/data/'
dataPath = '/home/ubuntu/DLP_Assignments/Lab3/data/data/'
batch_size = 4
epoch_size_resnet18 = 10
epoch_size_resnet50 = 5

trainDataset = RetinopathyLoader(
    dataPath,
    'train',
    transformTraining
)

testDataset = RetinopathyLoader(
    dataPath,
    'test',
    transformTesting
)

trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, pin_memory=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def main():
    torch.backends.cudnn.enabled = True

    # Models
    nets = {
        "rs18": torchvision.models.resnet18(pretrained=False).to(device),
        "rs18_pretrain": torchvision.models.resnet18(pretrained=True).to(device),
        "rs50": torchvision.models.resnet50(pretrained=False).to(device),
        "rs50_pretrain": torchvision.models.resnet50(pretrained=True).to(device)
    }

    # Optimizers
    criterion = nn.CrossEntropyLoss()
    learning_rates = {0.003, 0.003, 0.003, 0.003}

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

    train("rs18", nets["rs18"], optimizers["rs18"], criterion, accuracy, epoch_size_resnet18)
    train("rs18_pretrain", nets["rs18_pretrain"], optimizers["rs18_pretrain"], criterion, accuracy, epoch_size_resnet18)
    train("rs50", nets["rs50"], optimizers["rs50"], criterion, accuracy, epoch_size_resnet50)
    train("rs50_pretrain", nets["rs50_pretrain"], optimizers["rs50_pretrain"], criterion, accuracy, epoch_size_resnet50)

def train(key, model, optimizer, criterion, accuracy, epoch_size):
    print('Now training : ', key)
    name = key
    key += "_train"

    for epoch in range(epoch_size):
        print('epoch : ', epoch)
        train_correct = 0.0
        model.train(mode=True)
        for step, (x, y) in enumerate(trainLoader):
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
        torch.save(model.state_dict(), name + '.pkl')
        test(name, model, accuracy, epoch)

    f = open('terminal.txt', 'a')
    f.write(key + str(accuracy.__getitem__(key)))
    f.write('\n')


def test(key, model, accuracy, epoch):
    key += "_test"
    model.eval()
    test_correct = 0.0
    for step, (x, y) in enumerate(testLoader):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_hat = model(x)
        test_correct += (torch.max(y_hat, 1)[1] == y).sum().item()

    accuracy[key] += [(test_correct * 100.0) / len(testDataset)]

    print(key, 'Acc: ', accuracy.__getitem__(key)[epoch])
    print('')


if __name__ == '__main__':
    main()