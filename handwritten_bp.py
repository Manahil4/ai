import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# # Softmax activation function for output layer (multi-class classification)
# def softmax(x):
#     exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#by gpt
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for each sample
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function: Cross-Entropy Loss for multi-class classification
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    log_p = - np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
    loss = np.sum(log_p) / n_samples
    return loss

# Derivative of Cross-Entropy with Softmax
def softmax_derivative(y_true, y_pred):
    return y_pred - y_true

# Training function using backpropagation
def train_network(X, y, learning_rate, epochs):
    np.random.seed(42)
    
    input_neurons = X.shape[1]   # 784 input features (28x28 pixels)
    hidden_neurons = 64          # You can choose the number of neurons in the hidden layer
    output_neurons = 10          # 10 output classes for digits (0-9)

    # Initialize weights and biases
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
    
    bias_hidden = np.random.uniform(size=(1, hidden_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))

    for epoch in range(epochs):
        # Forward Propagation
        hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = softmax(output_input)

        # Loss calculation
        loss = cross_entropy_loss(y, predicted_output)

        # Backpropagation
        d_output = softmax_derivative(y, predicted_output)
        d_hidden = np.dot(d_output, weights_hidden_output.T) * sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output -= np.dot(hidden_output.T, d_output) * learning_rate
        weights_input_hidden -= np.dot(X.T, d_hidden) * learning_rate
        
        bias_output -= np.sum(d_output, axis=0, keepdims=True) * learning_rate
        bias_hidden -= np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
        
        # Print the loss at certain intervals
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Testing function
def test_network(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = softmax(output_input)

    return np.argmax(predicted_output, axis=1)

# Load and preprocess the MNIST dataset
def preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize the images to values between 0 and 1
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255
    
    # One-hot encode the labels
    y_train =to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test

# Training parameters
learning_rate = 0.1
epochs = 1000

# Load and preprocess data
X_train, y_train, X_test, y_test = preprocess_data()

# Train the network
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train_network(X_train, y_train, learning_rate, epochs)

# Test the network
y_pred = test_network(X_test, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Calculate accuracy
accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred) * 100
print(f"Test Accuracy: {accuracy:.2f}%")
