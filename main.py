import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return z > 0


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T


def forward_prop(W1, b1, W2, b2, X):
    z1 = W1.dot(X) + b1  # (10, 41000)
    a1 = relu(z1)  # (10, 41000)
    z2 = W2.dot(a1) + b2  # (10, 41000)
    a2 = softmax(z2)  # (10, 41000)
    return z1, a1, z2, a2


def backward_prop(z1, a1, a2, w2, X, Y):
    m = Y.size
    one_hot_y = one_hot(Y, num_classes=10)  # (10, 41000)
    dz2 = a2 - one_hot_y  # (10, 41000)
    dw2 = 1 / m * dz2.dot(a1.T)  # (10, 10)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)  # (10, 1)
    dz1 = w2.T.dot(dz2) * relu_derivative(z1)  # (10, 41000)
    dw1 = 1 / m * dz1.dot(X.T)  # (10, 784)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)  # (10, 1)
    return dw1, db1, dw2, db2


def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def train_fn(X, Y, epochs, learning_rate):

    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        z1, a1, z2, a2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(z1, a1, a2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        # if i % 10 == 0:
        #     print("Iteration: ", i)
        #     predictions = get_predictions(a2)
        #     print(get_accuracy(predictions, y_train))

    return W1, b1, W2, b2


def show_predictions(X, predictions, Y, num_examples_to_show=10):
    for i in range(num_examples_to_show):
        img = X.T[i].reshape((28, 28)) * 255
        print("Prediction: ", predictions[i])
        print("Label: ", Y.T[i])
        plt.imshow(img)
        plt.show()


def make_predictions(X, W1, b1, W2, b2):
    z1, a1, z2, a2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(a2)
    return predictions


def main():

    TEST_SIZE = 1000
    EPOCHS = 500
    LEARNING_RATE = 0.1

    train_df = pd.read_csv('data/train.csv')
    train_data = np.array(train_df)
    np.random.shuffle(train_data)

    x_train, y_train = train_data[TEST_SIZE:, 1:].T, train_data[TEST_SIZE:, :1].T
    x_train = x_train / 255.
    x_test, y_test = train_data[:TEST_SIZE, 1:].T, train_data[:TEST_SIZE, :1].T
    x_test = x_test / 255.

    W1, b1, W2, b2 = train_fn(x_train, y_train, EPOCHS, LEARNING_RATE)
    predictions = make_predictions(x_test, W1, b1, W2, b2)

    print(get_accuracy(predictions, y_test))
    show_predictions(x_test, predictions, y_test)


if __name__ == "__main__":
    main()


