import numpy as np


class NeuralNetwork:
    def __init__(self, NNArchitecture, weightInitial=0.7):
        np.set_printoptions(suppress=True)
        self.params = {}
        self.memory = {}
        self.grads_values = {}
        self.activationFunc = self.sigmoid
        self.backwardActivationFunc = self.sigmoidBackward
        self.NNArchitecture = NNArchitecture
        self.v = {}
        for i, layer in enumerate(NNArchitecture):
            layerInputSize = layer["inputDimension"]
            layerOutputSize = layer["outputDimension"]
            self.params["W" + str(i + 1)] = np.random.randn(layerOutputSize, layerInputSize) * weightInitial
            self.params["b" + str(i + 1)] = np.random.randn(layerOutputSize, 1) * weightInitial

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidBackward(self, dA, x):
        sig = self.sigmoid(x)
        return dA * sig * (1 - sig)

    def loss(self, y, y_hat):
        if y.ndim == 1:
            y_hat = y_hat.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if y_hat.size == y.size:
            y_hat = y_hat.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), y_hat] + 1e-7)) / batch_size

    def probToClass(self, probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_

    def accuracy(self, Y_hat, Y):
        Y_hat_ = self.probToClass(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def update(self, learningRate):
        # SGD
        # for i, layer in enumerate(self.NNArchitecture):
        #     self.params["W" + str(i+1)] -= learningRate * self.grads_values["dW" + str(i+1)]
        #     self.params["b" + str(i+1)] -= learningRate * self.grads_values["db" + str(i+1)]

        # Momentum
        # momentum = 0.9
        # if len(self.v) == 0 :
        #     for key, val in self.params.items():
        #         self.v[key] = np.zeros_like(val)

        # for key in self.params.keys():
        #     self.v[key] = momentum * self.v[key] - learningRate*self.grads_values['d' + key]
        #     self.params[key] += self.v[key]

        # AdaGrad
        if len(self.v) == 0:
            for key, val in self.params.items():
                self.v[key] = np.zeros_like(val)

        for key in self.params.keys():
            self.v[key] += self.grads_values['d' + key] * self.grads_values['d' + key]
            self.params[key] -= learningRate * self.grads_values['d' + key] / (np.sqrt(self.v[key]) + 1e-7)

    def forwardPropagation(self, A_prev, W_curr, b_curr):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return self.activationFunc(Z_curr), Z_curr

    def backwardPropagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev):
        m = A_prev.shape[1]

        dZ_curr = self.backwardActivationFunc(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def fullForwardPropagation(self, X):
        A_curr = X
        for i, layer in enumerate(self.NNArchitecture):
            A_prev = A_curr

            W_curr = self.params["W" + str(i + 1)]
            b_curr = self.params["b" + str(i + 1)]
            A_curr, Z_curr = self.forwardPropagation(A_prev, W_curr, b_curr)

            self.memory["A" + str(i)] = A_prev
            self.memory["Z" + str(i + 1)] = Z_curr

        return A_curr

    def fullBackwardPropagation(self, Y_hat, Y):
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        # partial L/Partial Y_hat = -(Y/Y_hat - (1-Y)/(1-Y_hat))
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layerIndexPrev, layer in reversed(list(enumerate(self.NNArchitecture))):
            layerIndexCurr = layerIndexPrev + 1
            dA_curr = dA_prev

            A_prev = self.memory["A" + str(layerIndexPrev)]
            Z_curr = self.memory["Z" + str(layerIndexCurr)]
            W_curr = self.params["W" + str(layerIndexCurr)]
            b_curr = self.params["b" + str(layerIndexCurr)]

            dA_prev, dW_curr, db_curr = self.backwardPropagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev)

            self.grads_values["dW" + str(layerIndexCurr)] = dW_curr
            self.grads_values["db" + str(layerIndexCurr)] = db_curr

    def train(self, X, Y, epoch, learningRate):
        X = np.transpose(X)
        Y = np.transpose(Y)
        lossHistory = []
        accuHistory = []

        for i in range(epoch):
            Y_hat = self.fullForwardPropagation(X)
            loss = self.loss(Y_hat, Y)
            accu = self.accuracy(Y_hat, Y)

            if i % 100 == 0:
                lossHistory.append(loss)
                accuHistory.append(accu)
                print("Epoch : ", i, " Loss : ", loss, " Accu : ", accu)

            self.fullBackwardPropagation(Y_hat, Y)
            self.update(learningRate)

        return lossHistory, accuHistory
