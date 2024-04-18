from mnist_loader import load_data_wrapper
import numpy as np
import matplotlib.pyplot as plt
import random

training_data, validation_data, test_data = load_data_wrapper()


######################################## functions ########################################

# plot one image
def plot_image(image):
    fig, axes = plt.subplots()
    image = image.reshape(28, 28)
    axes.matshow(image, cmap=plt.cm.binary)
    plt.show()

# plot a list of images
def plot_images(images):
    fig, axes = plt.subplots(nrows=1, ncols=len(images))
    for i, ax in enumerate(axes):
        ax.matshow(images[i][0].reshape(28, 28), cmap = plt.cm.binary)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# sigmoid function
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# derivative of sigmoid
def sigmoid_prime(x):
    s = sigmoid(x) * (1-sigmoid(x))
    return s

# score function
def f(x, W1, W2, B1, B2):
    Z1 = np.dot(W1, x) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)
    return A2

def vectorize_mini_batch(mini_batch):
    mini_batch_x = []
    mini_batch_y = []
    for i in range(0, len(mini_batch)):
        mini_batch_x.append(mini_batch[i][0])
        mini_batch_y.append(mini_batch[i][1])
    
    X = np.hstack(mini_batch_x)
    Y = np.hstack(mini_batch_y)
    return X, Y

# stochastic gradient descent
def gradient_descent(training_data, epochs, mini_batch_size, eta, test_data):
    n = len(training_data)
    n_test = len(test_data)

    W1 = np.random.randn(30, 784)
    W2 = np.random.randn(10, 30)
    B1 = np.random.randn(30, 1)
    B2 = np.random.randn(10, 1)

    for i in range(epochs):
        random.shuffle(training_data)
        for k in range(0, n, mini_batch_size):
            mini_batch = training_data[k: k+mini_batch_size]
            X, Y = vectorize_mini_batch(mini_batch)

            # feed forward
            Z1 = np.dot(W1, X) + B1
            A1 = sigmoid(Z1)
            Z2 = np.dot(W2, A1) + B2
            A2 = sigmoid(Z2)

            # backpropagate
            dZ2 = 1 / mini_batch_size * (A2-Y) * sigmoid_prime(Z2)
            dW2 = np.dot(dZ2, A1.T)
            dB2 = 1 / mini_batch_size * np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = 1 / mini_batch_size * np.dot(W2.T, dZ2) * sigmoid_prime(Z1)
            dW1 = np.dot(dZ1, X.T)
            dB1 = 1 / mini_batch_size * np.sum(dZ1, axis=1, keepdims=True)

            # update parameters
            W2 -= eta * dW2
            W1 -= eta * dW1
            B2 -= eta * dB2
            B1 -= eta * dB1

        test_results = [(np.argmax(f(x, W1, W2, B1, B2)), y) for (x, y) in test_data]
        num_correct = sum(int(x==y) for (x, y) in test_results)
        print(f"Epoch {i+1:2} : {num_correct:5} / {n_test:5}")

    return W1, B1, W2, B2

# predict images after training
def predict(images, W1, W2, B1, B2):
    X, Y = vectorize_mini_batch(images)
    A = f(X, W1, W2, B1, B2)
    predictions = np.argmax(A, axis = 0)
    predictions = list(predictions)
    return predictions

######################################## main ########################################

def main():
    W1, B1, W2, B2 = gradient_descent(training_data, 30, 10, 3, test_data)
    sample_data = training_data[10:20]

    predictions = predict(sample_data, W1, W2, B1, B2)
    print(predictions)
    plot_images(sample_data)

if __name__ == "__main__":
    main()
