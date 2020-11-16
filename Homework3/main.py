import gzip
import pickle
import numpy as np
import random


class Network:
    def __init__(self, images, labels, learning_rate, iterations, mini_batch_size, layer_sizes):
        # self.training_data = training_data
        self.images = images
        self.labels = labels
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.mini_batch_size = mini_batch_size
        # layer_sizes contains all layer sizes [input, hidden, output]
        self.layer_sizes = layer_sizes
        self.weights = self.weights_initialisation()
        self.biases = self.bias_initialisation()
        self.digit_array = self.digit_array_initialisation()

    def weights_initialisation(self):
        return {'2': np.random.normal(size=(self.layer_sizes[0], self.layer_sizes[1]), loc=0.0,
                                      scale=1.0 / self.layer_sizes[0]),
                '3': np.random.normal(size=(self.layer_sizes[1], self.layer_sizes[2]), loc=0.0,
                                      scale=1.0 / self.layer_sizes[1])}

    def bias_initialisation(self):
        return {'2': np.random.normal(size=self.layer_sizes[1], scale=1.0, loc=0.0),
                '3': np.random.normal(size=self.layer_sizes[2], scale=1.0, loc=0.0)}

    def digit_array_initialisation(self):
        digit_labels = np.zeros([len(self.images), 10])
        for digit in range(10):
            digit_labels[:, digit] = self.digit_label_initialisation(digit)
        print(digit_labels)
        return digit_labels

    def digit_label_initialisation(self, digit):
        return (self.labels == digit) * 1

    def get_mini_data(self, i, index_list):
        # mini_samples = self.images[(i - 1) * self.mini_batch_size:i * self.mini_batch_size]
        # print(len(index_list))
        mini_samples = [self.images[index_list[k]] for k in
                        range((i - 1) * self.mini_batch_size, i * self.mini_batch_size)]
        # mini_labels = self.digit_array[(i - 1) * self.mini_batch_size:i * self.mini_batch_size]
        mini_labels = [self.digit_array[index_list[k]] for k in
                       range((i - 1) * self.mini_batch_size, i * self.mini_batch_size)]

        return np.array(mini_samples), np.array(mini_labels)

    def train(self):
        index = 0
        n = len(self.images)
        index_list = [*range(n)]
        while self.iterations > index:
            print(f"Iteratii ramase: {self.iterations - index}")
            random.shuffle(index_list)
            nr_mini_batches = n // self.mini_batch_size
            lr = self.learning_rate / self.mini_batch_size

            for i in range(1, nr_mini_batches + 1):
                mini_samples, mini_labels = self.get_mini_data(i, index_list)
                # print(mini_labels)
                # print(list(mini_samples[0]))
                # print(list(mini_samples[1]))
                gradient_weights, gradient_bias = self.backpropagation(mini_samples, mini_labels)
                self.weights['2'] -= lr * gradient_weights[0]
                self.weights['3'] -= lr * gradient_weights[1]
                self.biases['2'] -= self.learning_rate * gradient_bias[0]
                self.biases['3'] -= self.learning_rate * gradient_bias[1]
            print("eroare: ", self.cross_entropy())
            print("accuracy:", self.accuracy(self.images, self.labels))
            index += 1

    def backpropagation(self, samples, labels):
        partial_derivative_weights = [np.zeros_like(self.weights['2']), np.zeros_like(self.weights['3'])]
        partial_derivative_bias = [np.zeros_like(self.biases['2']), np.zeros_like(self.biases['3'])]
        delta_error = [np.zeros_like(self.biases['2']), np.zeros_like(self.biases['3'])]
        sigmoid_layers = [samples]
        x = samples
        for l in self.weights:
            z = np.dot(x, self.weights[l]) + self.biases[l]
            if l == '2':
                x = self.sigmoid(z)
            if l == '3':
                x = self.softmax(z, 1)
            sigmoid_layers.append(np.copy(x))

        layer = 1
        delta_error[layer] = sigmoid_layers[layer + 1] - labels

        partial_derivative_weights[layer] = np.dot(sigmoid_layers[layer].T, delta_error[layer])
        partial_derivative_bias[layer] = np.mean(delta_error[layer], axis=0)

        layer = 0
        delta_error[layer] = self.sigmoid_derivative(sigmoid_layers[layer + 1]) * np.dot(delta_error[layer + 1],
                                                                                         self.weights['3'].T)
        # partial_derivative_weights[layer] = np.dot(sigmoid_layers[layer].T, delta_error[layer])
        partial_derivative_weights[layer] = np.dot(sigmoid_layers[layer].T, delta_error[layer])
        partial_derivative_bias[layer] = np.mean(delta_error[layer], axis=0)

        return partial_derivative_weights, partial_derivative_bias

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return np.divide(1.0, (1.0 + np.exp(-z)))

    def sigmoid_derivative(self, y):
        return np.multiply(y, 1.0 - y)

    def cross_entropy(self):
        eps = 0.00001
        y = self.feedforward(self.images)
        cost = np.sum(np.multiply(self.digit_array, np.log(y + eps)))
        # return -np.divide(1.0, len(self.images)) * cost
        print(len(self.images))
        return -1.0 / len(self.images) * cost

    def accuracy(self, data, labels):
        output = np.argmax(self.feedforward(data), 1)
        accuracy = np.count_nonzero((output - labels) == 0)
        return accuracy / len(data) * 100

    def feedforward(self, data):
        y = data
        for l in self.weights:
            z = np.dot(y, self.weights[l]) + self.biases[l]
            if l == '2':
                y = self.sigmoid(z)
            if l == '3':
                y = self.softmax(z, 1)
        return y

    def softmax(self, z, axis=0):
        return np.divide(np.exp(z).T, np.sum(np.exp(z), axis=axis)).T


if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')

    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    f.close()
    network = Network(train_set[0], train_set[1], 0.1, 100, 10, [len(train_set[0][0]), 100, 10])
    network.train()
