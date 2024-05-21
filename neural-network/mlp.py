from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score


def plot_digit(digits, index):
    plt.gray()
    plt.matshow(digits.images[index])
    plt.show()


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(
            2.0 / input_size
        )
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return self.weights @ self.input + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient @ self.input.T
        input_gradient = self.weights.T @ output_gradient
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        relu_prime = lambda x: (x > 0).astype(int)
        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class Flatten(Layer):
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten().reshape(-1, 1)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)


x, y = load_digits(return_X_y=True)
x_train, y_train = x[:1500], y[:1500]
x_test, y_test = x[1500:], y[1500:]

# normalize - grayscale values for a pixel are between 0 and 16 for this dataset
x_train = x_train / 16
x_test = x_test / 16

# one-hot
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# network
network = [Flatten(), Dense(64, 128), ReLU(), Dense(128, 10), Sigmoid()]

# training
epochs = 50
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    for i in range(len(x_train)):
        input = x_train[i].reshape(-1, 1)
        for layer in network:
            input = layer.forward(input)
        error += mse(y_train[i].reshape(-1, 1), input)
        output_gradient = mse_prime(y_train[i].reshape(-1, 1), input)
        for layer in reversed(network):
            output_gradient = layer.backward(output_gradient, learning_rate)
    error /= len(x_train)
    print(f"Epoch {epoch + 1}/{epochs} - Error: {error}")

test_error = 0
correct_pred = 0
dataset_len = 0
for i in range(len(x_test)):
    dataset_len += 1
    input = x_test[i].reshape(-1, 1)
    for layer in network:
        input = layer.forward(input)
    predicted = np.argmax(input)
    target = np.argmax(y_test[i])
    if predicted == target:
        correct_pred += 1
    test_error += mse(y_test[i].reshape(-1, 1), input)
test_error /= len(x_test)
print(f"Test Error after training: {test_error}")
print(correct_pred / dataset_len * 100)
