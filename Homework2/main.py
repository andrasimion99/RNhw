import gzip
import pickle
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')

train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
f.close()


def activation(input):
    return (input > 0) * 1


def digit_label_initialisation(digit, labels):
    return (labels == digit) * 1


def bias_initialisation():
    return np.random.rand(10)


def weights_initialisation(images):
    return np.random.rand(images.shape[1], 10)


def get_mini_data(images, i, nr_mini_batches, labels):
    mini_batch_size = len(images) // nr_mini_batches
    mini_batch = images[(i - 1) * mini_batch_size:i * mini_batch_size]
    mini_labels = labels[(i - 1) * mini_batch_size:i * mini_batch_size]
    return mini_batch, mini_labels


def get_mini_batch_count(images):
    batch_size = len(images)
    mini_batch_size = 10
    return batch_size // mini_batch_size


def train_all_perceptrons(weights, labels, iterations, images, learning_rate, nr_mini_batches, bias):
    best_accuracy = 0
    model = dict()
    model["accuracy"] = best_accuracy
    index = 0
    model["iteration"] = index
    while iterations > index:
        print(f"Iteratii ramase: {iterations - index}")
        delta = np.zeros_like(weights)
        beta = np.zeros_like(bias)
        for i in range(1, nr_mini_batches):
            mini_batch, mini_labels = get_mini_data(images, i, nr_mini_batches, labels)
            z = np.dot(mini_batch, weights) + bias
            y = activation(z)

            error = mini_labels - y
            # print(f"In minibach-ul {i} avem {np.count_nonzero(error != 0)} clasificari gresite.")

            delta += np.dot(mini_batch.T, error) * learning_rate
            beta += np.sum(error) * learning_rate

        for i in range(1, nr_mini_batches):
            weights += delta
            bias += beta

        accuracy = validation(valid_set, weights, bias)
        print(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model["accuracy"] = best_accuracy
            model["iteration"] = index
            model["weights"] = weights
            model["bias"] = bias
        elif index - model["iteration"] >= 10:
            break
        index += 1
    print("best iteration:", model["iteration"])
    print("best accuracy:", model["accuracy"])
    return model["weights"], model["bias"]


def train_all_digits(iterations, images, labels, learning_rate):
    nr_mini_batches = get_mini_batch_count(images)
    weights = weights_initialisation(images)
    bias = bias_initialisation()

    digit_labels = np.zeros([len(images), 10])
    for digit in range(len(np.unique(labels))):
        digit_labels[:, digit] = digit_label_initialisation(digit, labels)
    best_weights, best_bias = train_all_perceptrons(weights, digit_labels, iterations, images, learning_rate,
                                                    nr_mini_batches, bias)

    return best_weights, best_bias


def validation(data_set, weights, bias):
    images = data_set[0]
    labels = data_set[1]
    z = np.dot(images, weights) + bias
    z = np.argmax(z, 1)
    total_error = z - labels
    misclassified = np.count_nonzero(total_error != 0)
    accuracy = (len(images) - misclassified) / len(images)
    return accuracy


def test(data_set, weights, bias):
    print("Testare:")
    images = data_set[0]
    labels = data_set[1]
    z = np.dot(images, weights) + bias
    z = np.argmax(z, 1)
    total_error = z - labels
    misclassified = np.count_nonzero(total_error != 0)
    accuracy = (len(images) - misclassified) / len(images)
    print(f"Avem {misclassified} clasificari gresite.")
    print(f"Avem acuratetea totala de {accuracy * 100} %")
    return accuracy


if __name__ == "__main__":
    iterations = 100
    images = train_set[0]
    labels = train_set[1]
    learning_rate = 0.5
    weights, bias = train_all_digits(iterations, images, labels, learning_rate)
    test(test_set, weights, bias)
