import numpy as np
from tensorflow.keras.datasets import mnist  # Only used for importing the MNIST dataset

# Load the MNIST dataset and split it into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images by scaling pixel values to the range [0, 1] for numerical stability
x_train = x_train / 255.0
x_test = x_test / 255.0

# Function to convert integer labels to one-hot encoded vectors for classification
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((len(labels), num_classes))  # Initialize a matrix of zeros
    for i, label in enumerate(labels):  # Iterate over each label
        encoded[i, label] = 1  # Set the corresponding index to 1
    return encoded

# Apply one-hot encoding to training and test labels to prepare them for training
y_train_encoded = one_hot_encode(y_train, 10)
y_test_encoded = one_hot_encode(y_test, 10)

# Function to perform a 2D convolution operation on an image using a given kernel
def conv2d(image, kernel):
    h, w = image.shape  # Get the dimensions of the image
    kh, kw = kernel.shape  # Get the dimensions of the kernel
    output = np.zeros((h - kh + 1, w - kw + 1))  # Initialize the output matrix
    for i in range(h - kh + 1):  # Iterate over each row within bounds for convolution
        for j in range(w - kw + 1):  # Iterate over each column within bounds
            region = image[i:i + kh, j:j + kw]  # Extract a region of the image matching the kernel size
            output[i, j] = np.sum(region * kernel)  # Perform element-wise multiplication and summation
    return output  # Return the convolved output

# ReLU activation function: introduces non-linearity by zeroing out negative values
def relu(x):
    return np.maximum(0, x)  # Replace negative values with 0 to retain only positive values

# Function to perform max pooling on an image to reduce its dimensionality
def max_pooling(image, size=2, stride=2):
    h, w = image.shape  # Get the dimensions of the image
    output_h = (h - size) // stride + 1  # Calculate the output height
    output_w = (w - size) // stride + 1  # Calculate the output width
    output = np.zeros((output_h, output_w))  # Initialize the output matrix
    for i in range(0, h - size + 1, stride):  # Iterate with stride over rows
        for j in range(0, w - size + 1, stride):  # Iterate with stride over columns
            region = image[i:i + size, j:j + size]  # Extract the region for pooling
            output[i // stride, j // stride] = np.max(region)  # Store the max value in the region
    return output  # Return the pooled output

# Function to flatten a 2D matrix into a 1D array (used for transitioning between layers)
def flatten(matrix):
    return matrix.reshape(-1)  # Reshape the matrix into a vector

# Function to compute the output of a fully connected layer with ReLU activation
def dense_forward(x, weights, bias):
    return relu(np.dot(x, weights) + bias)  # Compute the linear combination and apply ReLU

# Softmax function to convert output scores to class probabilities
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Normalize by subtracting max for numerical stability
    return exp_x / exp_x.sum(axis=1, keepdims=True)  # Divide by the sum for each sample to get probabilities

# Function to calculate cross-entropy loss for classification tasks
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]  # Add a small value for numerical stability

# Function to perform backpropagation to compute gradients and update weights
def backward_pass(activations, weights, y_true, learning_rate=0.01):
    gradients_w = []  # List to store computed weight gradients
    gradients_b = []  # List to store computed bias gradients
    
    delta = activations[-1] - y_true  # Compute error at the output layer

    # Compute gradient for output layer weights and biases
    gradients_w.append(np.dot(activations[-2].T, delta))  # Derivative of loss w.r.t. weights
    gradients_b.append(np.sum(delta, axis=0))  # Derivative of loss w.r.t. biases
    
    # Backpropagate through the hidden layers
    for i in range(len(weights) - 2, -1, -1):
        delta = np.dot(delta, weights[i + 1].T) * (activations[i + 1] > 0)  # Compute the error term with ReLU derivative
        gradients_w.append(np.dot(activations[i].T, delta))  # Compute weight gradients
        gradients_b.append(np.sum(delta, axis=0))  # Compute bias gradients
    
    # Reverse the gradient lists to align with the network layer order
    gradients_w = gradients_w[::-1]
    gradients_b = gradients_b[::-1]
    
    # Update weights and biases with the computed gradients
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients_w[i]  # Update weights
        biases[i] -= learning_rate * gradients_b[i]  # Update biases

# Initialize network parameters
input_size = 784  # Flattened input size for 28x28 images
hidden_layer_size = 32  # Number of neurons in the hidden layer
output_size = 10  # Number of output classes (digits 0-9)

np.random.seed(42)  # Seed for reproducibility
weights = [
    np.random.randn(input_size, hidden_layer_size) * 0.01,  # Weights for input-to-hidden connections
    np.random.randn(hidden_layer_size, output_size) * 0.01  # Weights for hidden-to-output connections
]
biases = [
    np.zeros(hidden_layer_size),  # Bias for the hidden layer
    np.zeros(output_size)  # Bias for the output layer
]

# Training loop setup
epochs = 10  # Number of complete passes over the training data
batch_size = 32  # Number of samples per training batch

# Training process
for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):  # Iterate over mini-batches
        x_batch = x_train[i:i + batch_size].reshape(-1, input_size)  # Flatten the batch images
        y_batch = y_train_encoded[i:i + batch_size]  # Get the corresponding labels
        
        activations = [x_batch]  # Initialize the list of activations
        for w, b in zip(weights, biases):  # Forward pass through each layer
            activations.append(dense_forward(activations[-1], w, b))  # Append the activation to the list
        
        backward_pass(activations, weights, y_batch)  # Backpropagation to update parameters
    
    y_pred = softmax(activations[-1])  # Compute output probabilities after the epoch
    loss = cross_entropy_loss(y_batch, y_pred)  # Calculate loss for monitoring
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))  # Calculate training accuracy
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')  # Print progress

# Model evaluation on the test set
x_test_flattened = x_test.reshape(-1, input_size)  # Flatten test images
test_activations = [x_test_flattened]  # Initialize activations for the test set
for w, b in zip(weights, biases):  # Forward pass through the test set
    test_activations.append(dense_forward(test_activations[-1], w, b))  # Append test activations
y_test_pred = softmax(test_activations[-1])  # Compute test predictions

# Calculate and display test accuracy
test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test_encoded, axis=1))  # Test accuracy
print(f'Test Accuracy: {test_accuracy:.4f}')  # Print test accuracy
