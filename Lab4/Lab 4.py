import matplotlib.pyplot as plt
import numpy as np


def showResult(accu):
    plt.figure('RNN BPTT Test Results', figsize=(15, 7))
    plt.title('RNN BPTT Test Results')
    plt.xlabel('Iteration Number')
    plt.ylabel('Accuracy(%)')
    plt.xlim(0, iter_num)
    plt.ylim(0, 110)
    plt.grid()

    plt.plot(accu.keys(), accu.values(), 'b-', linewidth=1.5)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(output):
    return output * (1 - output)


# Hyper Parameters
binary_dim = 8
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
iter_num = 20000
iter_print_num = 200
accu = {}

# Setting of generating numbers
largest_num = int(pow(2, binary_dim) / 2)
smallest_num = int(-pow(2, binary_dim) / 2)
binary = np.unpackbits(np.array([range(smallest_num, largest_num)], dtype=np.uint8).T, axis=1)
num_table = {}
for i in range(smallest_num, largest_num):
    num_table[i + largest_num] = i

# initialize neural network weights
w0 = np.random.normal(0, 1, [input_dim, hidden_dim])
w1 = np.random.normal(0, 1, [hidden_dim, output_dim])
wh = np.random.normal(0, 2, [hidden_dim, hidden_dim])

d0 = np.zeros_like(w0)
d1 = np.zeros_like(w1)
dh = np.zeros_like(wh)

for i in range(iter_num + 1):
    error = 0

    a_int = num_table[np.random.randint(64, 192)]
    a = binary[a_int]

    b_int = num_table[np.random.randint(64, 192)]
    b = binary[b_int]

    c_int = a_int + b_int
    c = binary[c_int + largest_num]

    predict = np.zeros_like(c)

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer
        layer_1 = sigmoid(np.dot(X, w0) + np.dot(layer_1_values[-1], wh))

        # output layer
        y_hat = sigmoid(np.dot(layer_1, w1))

        layer_2_error = y - y_hat
        layer_2_deltas.append((layer_2_error) * deriv_sigmoid(y_hat))
        error += np.abs(layer_2_error[0])

        predict[binary_dim - position - 1] = np.round(y_hat[0][0])

        layer_1_values.append(np.ndarray.copy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # error
        layer_2_delta = layer_2_deltas[-position - 1]
        layer_1_delta = (future_layer_1_delta.dot(wh.T) + layer_2_delta.dot(w1.T)) * deriv_sigmoid(layer_1)

        # update
        d1 += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        dh += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        d0 += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    w0 += d0 * alpha
    w1 += d1 * alpha
    wh += dh * alpha

    d0 *= 0
    d1 *= 0
    dh *= 0

    if (i % iter_print_num == 0):
        wrong = 0
        for j in range(0, 8):
            if predict[j] != c[j]:
                wrong += 1
        accu[i] = ((1 - (wrong / binary_dim)) * 100)

        ans = int(np.packbits(predict, axis=-1))
        if ans >= largest_num:
            ans -= pow(2, binary_dim)

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
