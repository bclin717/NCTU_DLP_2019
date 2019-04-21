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

largest_num = int(pow(2, dataDim) / 2)
smallest_num = int(-pow(2, dataDim) / 2)
binary = np.unpackbits(np.array([range(smallest_num, largest_num)], dtype=np.uint8).T, axis=1)
num_table = {}
for i in range(smallest_num, largest_num):
    num_table[i + largest_num] = i

w0 = np.random.normal(0, 1, [inputDim, hiddenDim])
w1 = np.random.normal(0, 1, [hiddenDim, outputDim])
wh = np.random.normal(0, 2, [hiddenDim, hiddenDim])

d0 = np.zeros_like(w0)
d1 = np.zeros_like(w1)
dh = np.zeros_like(wh)

for i in range(iterNum + 1):
    error = 0

    a_int = num_table[np.random.randint(64, 192)]
    a = binary[a_int]

    b_int = num_table[np.random.randint(64, 192)]
    b = binary[b_int]

    c_int = a_int + b_int
    c = binary[c_int + largest_num]

    predict = np.zeros_like(c)

    l2_deltas = list()
    l1_value = list()
    l1_value.append(np.zeros(hiddenDim))

    for position in range(dataDim):
        X = np.array([[a[dataDim - position - 1], b[dataDim - position - 1]]])
        y = np.array([[c[dataDim - position - 1]]]).T

        l1 = sigmoid(np.dot(X, w0) + np.dot(l1_value[-1], wh))
        y_hat = sigmoid(np.dot(l1, w1))

        l2_error = y - y_hat
        l2_deltas.append((l2_error) * deriv_sigmoid(y_hat))
        error += np.abs(l2_error[0])

        predict[dataDim - position - 1] = np.round(y_hat[0][0])

        l1_value.append(np.ndarray.copy(l1))

    future_l1_delta = np.zeros(hiddenDim)

    for position in range(dataDim):
        X = np.array([[a[position], b[position]]])
        l1 = l1_value[-position - 1]
        prev_l1 = l1_value[-position - 2]

        l2_d = l2_deltas[-position - 1]
        l1_delta = (future_l1_delta.dot(wh.T) + l2_d.dot(w1.T)) * deriv_sigmoid(l1)

        d1 += np.atleast_2d(l1).T.dot(l2_d)
        dh += np.atleast_2d(prev_l1).T.dot(l1_delta)
        d0 += X.T.dot(l1_delta)

        future_l1_delta = l1_delta

    w0 += d0 * alpha
    w1 += d1 * alpha
    wh += dh * alpha

    d0 *= 0
    d1 *= 0
    dh *= 0

    if (i % iterPrintNum == 0):
        wrong = 0
        for j in range(0, 8):
            if predict[j] != c[j]:
                wrong += 1
        accu[i] = ((1 - (wrong / dataDim)) * 100)

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