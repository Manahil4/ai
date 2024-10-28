import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize and reshape the dataset
X_train = X_train.reshape(X_train.shape[0], 784).T / 255  # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], 784).T / 255

m_train = X_train.shape[1]  # Number of training examples

# def init_params():
#     w1 = np.random.randn(10, 784) * 0.01  # Initialize weights for 10 neurons and 784 input features
#     b1 = np.zeros((10, 1))  # Biases for the hidden layer
#     w2 = np.random.randn(10, 10) * 0.01  # Weights for 10 output neurons and 10 hidden layer neurons
#     b2 = np.zeros((10, 1))  # Biases for output layer
#     return [w1, b1, w2, b2]
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.zeros((10, 1))  # Use zero initialization for biases
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.zeros((10, 1))  # Use zero initialization for biases
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

# def softmax(Z):
#     exp = np.exp(Z - np.max(Z))  # For numerical stability
#     return exp / exp.sum(axis=0)
def softmax(Z):
    # Subtract the max value from each column for numerical stability
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max of each column (sample-wise)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)  # Normalize across columns

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def forward_propagation(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def der_ReLU(Z):
    return Z > 0  # Derivative of ReLU: 1 when Z > 0, otherwise 0

def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y,m):
    m = X.shape[1]  # Number of examples
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * der_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# def backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y, m):
#     one_hot_Y = one_hot(Y)  # Convert labels to one-hot encoded format
#     dZ2 = A2 - one_hot_Y
#     dW2 = 1/m * dZ2.dot(A1.T)
#     db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
#     dZ1 = w2.T.dot(dZ2) * der_ReLU(Z1)
#     dW1 = 1/m * dZ1.dot(X.T)
#     db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
#     return dW1, db1, dW2, db2

def update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2 = init_params()  # Initialize parameters
    m = X.shape[1]  # Number of examples
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(w1, b1, w2, b2, X)  # Forward pass
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, w1, w2, X, Y, m)  # Backward pass
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)  # Update weights

        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}, Accuracy: {accuracy * 100:.2f}%")

    return w1, b1, w2, b2

# Train the model
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 0.001, 500)

# Test the model
Z1_test, A1_test, Z2_test, A2_test = forward_propagation(w1, b1, w2, b2, X_test)
predictions_test = get_predictions(A2_test)
accuracy_test = get_accuracy(predictions_test, Y_test)
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
