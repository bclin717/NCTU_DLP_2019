import matplotlib.pyplot as plt
import numpy as np


def showResult(accu):
    plt.figure('RNN BPTT Test Results', figsize=(15, 7))
    plt.title('RNN BPTT Test Results')
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy(%)')
    plt.xlim(0, iterNum)
    plt.ylim(0, 110)
    plt.grid()

    plt.plot(accu.keys(), accu.values(), 'b-', linewidth=1.5)
    plt.show()


def showResultErrors(errors):
    plt.figure('RNN BPTT Test Results', figsize=(15, 7))
    plt.title('RNN BPTT Test Results')
    plt.xlabel('Iteration Number')
    plt.ylabel('Error')
    plt.xlim(0, iterNum)
    plt.grid()

    plt.plot(errors.keys(), errors.values(), 'b-', linewidth=1.5)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(output):
    return output * (1 - output)


# Hyper Parameters
dataDim = 8
alpha = 0.1
inputDim = 2
hiddenDim = 16
outputDim = 1
iterNum = 20000
iterPrintNum = 200
accu = {}
errors = {}

largest_num = int(pow(2, dataDim) / 2)
smallest_num = int(-pow(2, dataDim) / 2)
binary = np.unpackbits(np.array([range(smallest_num, largest_num)], dtype=np.uint8).T, axis=1)
num_table = {}
for i in range(smallest_num, largest_num):
    num_table[i + largest_num] = i

U = np.random.normal(0, 1, [inputDim, hiddenDim])
V = np.random.normal(0, 1, [hiddenDim, outputDim])
W = np.random.normal(0, 2, [hiddenDim, hiddenDim])

dU = np.zeros_like(U)
dV = np.zeros_like(V)
db = np.zeros_like(W)

for i in range(iterNum + 1):
    error = 0
    dU = 0
    dV = 0
    db = 0

    a_int = num_table[np.random.randint(64, 192)]
    a = binary[a_int]

    b_int = num_table[np.random.randint(64, 192)]
    b = binary[b_int]

    c_int = a_int + b_int
    c = binary[c_int + largest_num]

    predict = np.zeros_like(c)

    Deltas = list()
    St = list()
    St.append(np.zeros(hiddenDim))

    for position in range(dataDim):
        X = np.array([[a[dataDim - position - 1], b[dataDim - position - 1]]])
        y = np.array([[c[dataDim - position - 1]]]).T

        St_now = sigmoid(np.dot(X, U) + np.dot(St[-1], W))
        y_hat = sigmoid(np.dot(St_now, V))

        l2_error = y - y_hat
        Deltas.append((l2_error) * deriv_sigmoid(y_hat))
        error += np.abs(l2_error[0])

        predict[dataDim - position - 1] = np.round(y_hat[0][0])

        St.append(np.ndarray.copy(St_now))

    St_now_delta = np.zeros(hiddenDim)

    for position in range(dataDim):
        X = np.array([[a[position], b[position]]])
        St_now = St[-position - 1]
        St_prev = St[-position - 2]

        Delta_L = Deltas[-position - 1]
        St_now_delta = (St_now_delta.dot(W.T) + Delta_L.dot(V.T)) * deriv_sigmoid(St_now)

        dV += np.atleast_2d(St_now).T.dot(Delta_L)
        db += np.atleast_2d(St_prev).T.dot(St_now_delta)
        dU += X.T.dot(St_now_delta)

    U += dU * alpha
    V += dV * alpha
    W += db * alpha

    if (i % iterPrintNum == 0):
        wrong = 0
        for j in range(0, 8):
            if predict[j] != c[j]:
                wrong += 1
        accu[i] = ((1 - (wrong / dataDim)) * 100)
        errors[i] = error
        ans = int(np.packbits(predict, axis=-1))
        if ans >= largest_num:
            ans -= pow(2, dataDim)

        print("--------------------------------------\n")
        print("      Iter :" + str(i))
        print("      Error:" + str(error))
        print("      Accu :" + str(accu[i]) + "%")
        print("    A : " + str(a) + ' = ' + str(a_int))
        print("+   B : " + str(b) + ' = ' + str(b_int))
        print("──────────────────────────────────")
        print("=       " + str(c) + ' = ' + str(c_int))
        print("Pred :  " + str(predict) + ' = ' + str(ans))
        print("\n--------------------------------------")

showResult(accu)
showResultErrors(errors)
