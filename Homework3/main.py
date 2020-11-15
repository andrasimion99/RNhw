import gzip
import pickle
import numpy as np
import random


class Network:
    def __init__(self, training_data, learning_rate, iterations, mini_batch_size, layer_sizes):
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.mini_batch_size = mini_batch_size
        # layer_sizes contains all layer sizes [input, hidden, output]
        self.layer_sizes = layer_sizes
        self.weights = self.weights_initialisation()
        self.biases = self.bias_initialisation()

    def weights_initialisation(self):
        return {'2': np.random.rand(self.layer_sizes[0], self.layer_sizes[1]),
                '3': np.random.rand(self.layer_sizes[1], self.layer_sizes[2])}

    def bias_initialisation(self):
        return {'2': np.zeros(self.layer_sizes[1]),
                '3': np.zeros(self.layer_sizes[2])}

    def train(self):
        index = 0
        n = len(self.training_data)
        while self.iterations > index:
            print(f"Iteratii ramase: {self.iterations - index}")
            random.shuffle(self.training_data)
            mini_batches = [training_data[k:k + self.mini_batch_size] for k in range(0, n, self.mini_batch_size)]
            for mini_batch in mini_batches:
                self.compute_mini_batch(mini_batch)
                # print("eroare:", self.cross_entropy(mini_batch))

            print("eroare:", self.cross_entropy(self.training_data))
            print("acuratete: ", self.accuracy(self.training_data), "%")

            index += 1

    def compute_mini_batch(self, mini_batch):
        # gradient_bias = [np.zeros(b.shape) for b in self.biases.values()]
        gradient_weights = [np.zeros_like(self.weights['2']), np.zeros_like(self.weights['3'])]
        gradient_bias = [np.zeros_like(self.biases['2']), np.zeros_like(self.biases['3'])]
        for sample, label in mini_batch:
            partial_derivative_weights, partial_derivative_bias = self.backpropagation(sample, label)
            gradient_weights += partial_derivative_weights
            gradient_bias += partial_derivative_bias
            # gradient_weights = [gw + nw for gw, nw in zip(gradient_weights, nabla_weights)]
            # gradient_bias = [gb + nb for gb, nb in zip(gradient_bias, nabla_biases)]
        # print(np.shape(gradient_weights[0]))
        # self.weights['2'] -= self.learning_rate * gradient_weights[0]
        # self.weights['3'] -= self.learning_rate * gradient_weights[1]
        # self.biases['2'] -= self.learning_rate * gradient_bias[0]
        # self.biases['3'] -= self.learning_rate * gradient_bias[1]
        self.weights['2'] = self.weights['2'] - self.learning_rate / len(mini_batch) * gradient_weights[0]
        self.weights['3'] = self.weights['3'] - self.learning_rate / len(mini_batch) * gradient_weights[1]
        self.biases['2'] = self.biases['2'] - self.learning_rate / len(mini_batch) * gradient_bias[0]
        self.biases['3'] = self.biases['3'] - self.learning_rate / len(mini_batch) * gradient_bias[1]

    def backpropagation(self, sample, label):
        partial_derivative_weights = [np.zeros_like(self.weights['2']), np.zeros_like(self.weights['3'])]
        partial_derivative_bias = [np.zeros_like(self.biases['2']), np.zeros_like(self.biases['3'])]
        delta_error = [np.zeros_like(self.biases['2']), np.zeros_like(self.biases['3'])]
        # print(type(delta_error))
        # net_input_layers = []
        sigmoid_layers = [sample]
        x = sample
        for w, b in zip(self.weights.values(), self.biases.values()):
            z = np.dot(w.T, x) + b
            # net_input_layers.append(z)
            x = self.sigmoid(z)
            sigmoid_layers.append(x)
        # # compute the error for the final layer
        layer = 1
        # delta_error[layer] = self.sigmoid_derivative(sigmoid_layers[layer + 1]) * error[layer]
        delta_error[layer] = sigmoid_layers[layer + 1] - label
        sigmoid = sigmoid_layers[layer].reshape(sigmoid_layers[layer].shape[0], 1)
        delta = delta_error[layer].reshape(1, delta_error[layer].shape[0])
        partial_derivative_weights[layer] = np.dot(sigmoid, delta)
        partial_derivative_bias[layer] = delta_error[layer]

        layer = 0
        delta_error[layer] = self.sigmoid_derivative(sigmoid_layers[layer + 1]) * np.dot(self.weights['3'],
                                                                                         delta_error[layer + 1])
        sigmoid = sigmoid_layers[layer].reshape(sigmoid_layers[layer].shape[0], 1)
        delta = delta_error[layer].reshape(1, delta_error[layer].shape[0])
        partial_derivative_weights[layer] = np.dot(sigmoid, delta)
        partial_derivative_bias[layer] = delta_error[layer]

        return partial_derivative_weights, partial_derivative_bias

    def sigmoid(self, z):
        return np.divide(1.0, (1.0 + np.exp(-z)))

    def sigmoid_derivative(self, y):
        # y = self.sigmoid(z)
        return np.multiply(y, 1.0 - y)

    def cross_entropy(self, data):
        cost = 0.0
        for image, label in data:
            y = self.feedforward(image)
            # print(y)
            cost += np.sum(label * np.log(y + 0.00001) + (1.0 - label) * np.log(1.0 - y + 0.00001))

        return -np.divide(1.0, len(data)) * cost

    def accuracy(self, data):
        accuracy = 0.0
        for image, label in data:
            output = np.argmax(self.feedforward(image))
            label = np.argmax(label)
            if output == label:
                accuracy += 1
        return accuracy / len(data) * 100

    def feedforward(self, image):
        y = image
        for w, b in zip(self.weights.values(), self.biases.values()):
            y = self.sigmoid(np.dot(w.T, y) + b)
        return y


def digit_array(digit):
    nums = np.zeros(10)
    nums[digit] = 1
    return nums


if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')

    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    f.close()
    training_data = [(train_set[0][i], digit_array(train_set[1][i])) for i in range(len(train_set[0]))]
    network = Network(training_data, 0.1, 20, 1000, [len(train_set[0][0]), 100, 10])
    network.train()
