import os
import sys

import numpy as np

sys.path.append(os.pardir)
from NeuralNetwork import *

np.set_printoptions(suppress=True)


def generateLinear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generateXOReasy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def showResult(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()


NNArchitecture = [
    {"inputDimension": 2, "outputDimension": 4},
    {"inputDimension": 4, "outputDimension": 4},
    {"inputDimension": 4, "outputDimension": 4},
    {"inputDimension": 4, "outputDimension": 1},
]

epoch = 15000
learningRate = 0.1
weightInitial = 5

xTrain, yTrain = generateLinear(n=2000)
xTest, yTest = generateLinear(n=100)
# xTrain, yTrain = generateXOReasy() 
# xTest, yTest = generateXOReasy()

nn = NeuralNetwork(NNArchitecture, weightInitial)

# Training
lossHistory, accuracyHistory = nn.train(xTrain, yTrain, epoch, learningRate)

# Testing
Y_hat = nn.fullForwardPropagation(np.transpose(xTest))
testAccuracy = nn.accuracy(Y_hat, np.transpose(yTest.reshape((yTest.shape[0], 1))))

# Output
print(Y_hat)
Y_hat_ = nn.probToClass(Y_hat)
print("Test accu : ", testAccuracy)
showResult(xTest, yTest, Y_hat_.all(axis=0))
