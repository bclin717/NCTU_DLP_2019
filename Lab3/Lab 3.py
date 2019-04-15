from __future__ import print_function

import torch
import torch.optim
import torchvision
# from data.dataloader import *
# from Resnet import  *
from torch import nn
from torch import optim
from torch.cuda import device
from torch.utils.data import DataLoader
from torchvision import transforms

from Lab3.data.dataloader import *
from Lab3.showResult import plot_confusion_matrix

transformTraining = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.Resize(224),
    transforms.ToTensor()
])

transformTesting = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataPath = '/home/kevin/PycharmProjects/DLP Assignments/Lab3/data/data/'
# dataPath = '/home/ubuntu/DLP_Assignments/Lab3/data/data/'

batch_size = 8
epoch_size_resnet18 = 20
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

trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)


def main():
    torch.backends.cudnn.enabled = True
    # trainSetting()

    accuracy = {
        **{"rs18_test": []}
    }
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5)
    model.load_state_dict(
        torch.load("/home/kevin/PycharmProjects/DLP Assignments/Lab3/models/lr0.001/rs18_pretrain.pkl"))
    model.cuda()

    test('rs18', model, accuracy)


def trainSetting():
    nets = {
        "rs18_pretrain": torchvision.models.resnet18(pretrained=True)
    }

    nets["rs18_pretrain"].fc = nn.Linear(nets["rs18_pretrain"].fc.in_features, 5)
    nets["rs18_pretrain"].avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Optimizers
    criterion = nn.CrossEntropyLoss()
    learning_rates = {0.0013}

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

    train("rs18_pretrain", nets["rs18_pretrain"], optimizers["rs18_pretrain"], criterion, accuracy, epoch_size_resnet18)


def train(key, model, optimizer, criterion, accuracy, epoch_size):
    print('Now training : ', key)
    model.cuda()
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
        torch.save(model.state_dict(), './models/' + name + '.pkl')
        test(name, model, accuracy, epoch)

    f = open(name + './models/terminal.txt', 'a')
    f.write(key + str(accuracy.__getitem__(key)))
    f.write('\n' + name + '_test' + str(accuracy.__getitem__(name + '_test')))
    f.write('\n')


def test(key, model, accuracy, epoch=0):
    key += "_test"
    model.eval()
    test_correct = 0.0
    pred = []
    truth = []
    for step, (x, y) in enumerate(testLoader):
        x = x.to(device)
        y = y.to(device).long().view(-1)
        y_hat = model(x)

        test_correct += (torch.max(y_hat, 1)[1] == y).sum().item()

        _, preds = torch.max(y_hat, 1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            pred.append(t.item())
            truth.append(p.item())

    classes = ['0', '1', '2', '3', '4']
    plot_confusion_matrix(truth, pred, classes=classes, normalize=False,
                          title='Normalized confusion matrix')

    accuracy[key] += [(test_correct * 100.0) / len(testDataset)]
    print(key, 'Acc: ', accuracy.__getitem__(key)[epoch])
    print('')


if __name__ == '__main__':
    main()
