import numpy as np
import pandas as pd

data = pd.read_csv('~/Downloads/train.csv')

# print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]

training_data = data[1000:m].T
y_train = training_data[0]
x_train = training_data[1:n]


def constructor():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 784)
    b2 = np.random.randn(10, 1)


def forward(w1, b1, w2, b2, input):
    z1 = (w1 @ input) + b1
    a1 = np.tanh(z1)
    z2 = (w2 @ input) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    inputs = probabilities
    return inputs


def relu(output):
    return np.maximum(0, output)


def one_hot(y):
    one_hot_y = np.zeros((m, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def backProp(ok1, output1, ok2, output2, expected_output):
    one_hot_y = one_hot(expected_output)

