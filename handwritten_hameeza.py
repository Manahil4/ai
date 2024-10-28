# import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# # Activation functions and their derivatives
# def relu(z):
#     return np.maximum(0, z)

# def relu_derivative(z):
#     return (z > 0).astype(float)

# def softmax(z):
#     exps = np.exp(z - np.max(z, axis=0, keepdims=True))
#     return exps / np.sum(exps, axis=0, keepdims=True)

# # Initialize weights and biases for multi-class classification
# def initialize_parameters(n_x, n_h, n_y):
#     np.random.seed(42)
#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros((n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros((n_y, 1))
    
#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     return parameters

# # Forward propagation with ReLU and softmax
# def forward_prop(X, parameters):
#     W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
#     Z1 = np.dot(W1, X) + b1
#     A1 = relu(Z1)
#     Z2 = np.dot(W2, A1) + b2
#     A2 = softmax(Z2)
#     cache = {"A1": A1, "A2": A2, "Z1": Z1, "Z2": Z2}
#     return A2, cache

# # Compute cost using cross-entropy for multi-class classification
# def compute_cost(A2, Y):
#     m = Y.shape[1]
#     cost = -np.sum(Y * np.log(A2 + 1e-8)) / m  # Adding epsilon to avoid log(0)
#     return np.squeeze(cost)

# # Backpropagation for ReLU and softmax
# def backward_prop(X, Y, parameters, cache):
#     A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]
#     W2 = parameters["W2"]
#     m = Y.shape[1]

#     # Output layer error
#     dZ2 = A2 - Y
#     dW2 = np.dot(dZ2, A1.T) / m
#     db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
#     # Hidden layer error
#     dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
#     dW1 = np.dot(dZ1, X.T) / m
#     db1 = np.sum(dZ1, axis=1, keepdims=True) / m

#     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#     return grads

# # Update weights and biases
# def update_parameters(parameters, grads, learning_rate):
#     W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
#     dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

#     W1 -= learning_rate * dW1
#     b1 -= learning_rate * db1
#     W2 -= learning_rate * dW2
#     b2 -= learning_rate * db2

#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     return parameters

# # Model function for training
# def model(X, Y, n_x, n_h, n_y, iterations, learning_rate):
#     parameters = initialize_parameters(n_x, n_h, n_y)

#     for i in range(iterations):
#         # Forward propagation
#         A2, cache = forward_prop(X, parameters)
        
#         # Cost calculation
#         cost = compute_cost(A2, Y)
        
#         # Backward propagation
#         grads = backward_prop(X, Y, parameters, cache)
        
#         # Parameter update
#         parameters = update_parameters(parameters, grads, learning_rate)
        
#         if i % 100 == 0:
#             print(f"Iteration {i}: Cost {cost:.4f}")
    
#     return parameters

# # Prediction function to evaluate on test data
# def predict(X, parameters):
#     A2, _ = forward_prop(X, parameters)
#     predictions = np.argmax(A2, axis=0)
#     return predictions

# # Load MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Preprocess data
# x_train, x_test = x_train.reshape(x_train.shape[0], -1).T / 255, x_test.reshape(x_test.shape[0], -1).T / 255
# y_train_one_hot = np.eye(10)[y_train].T  # One-hot encoding for multi-class labels

# # Set up network dimensions
# n_x = x_train.shape[0]
# n_h = 64
# n_y = 10
# learning_rate = 0.1
# iterations = 1000

# # Train the model
# parameters = model(x_train, y_train_one_hot, n_x, n_h, n_y, iterations, learning_rate)

# # Test the model
# predictions = predict(x_test, parameters)
# accuracy = np.mean(predictions == y_test)
# print(f"Test accuracy: {accuracy:.4f}")
#change above code by below because it is very slow, takes large no of iterations,not batch gradient descent
# 
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X = X / 255.0  # Normalize pixel values between 0 and 1

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
input_size = 784  # 28x28 flattened images
hidden_size = 64  # Number of neurons in hidden layer
output_size = 10  # Number of classes (0-9 digits)
epochs = 10       # Reduced for testing
batch_size = 64   # Mini-batch size
learning_rate = 0.01

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation function
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Derivatives
def relu_derivative(Z):
    return Z > 0

# Training with mini-batch gradient descent
for epoch in range(epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]
    
    for i in range(0, X_train.shape[0], batch_size):
        # Mini-batch data
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        
        # Forward pass
        Z1 = X_batch.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = softmax(Z2)
        
        # Loss calculation (cross-entropy)
        loss = -np.mean(np.sum(y_batch * np.log(A2 + 1e-8), axis=1))

        # Backward pass
        dZ2 = A2 - y_batch
        dW2 = A1.T.dot(dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X_batch.T.dot(dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Testing the model
def predict(X):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return np.argmax(A2, axis=1)

# Accuracy on test data
y_pred = predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
  