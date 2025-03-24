import numpy as np 
import time, sys, cProfile
import matplotlib.pyplot as plt

import cProfile as profile
import cProfile

class Layer_Primitive:
    """
    Base class for neural network layers. 
    Provides the interface for forward and backward propagation methods.
    """
    def __init__(self):
        """
        Initializes the input and output attributes of the layer.
        """
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        Performs the forward pass through the layer.
        Must be implemented by subclasses.
        
        Parameters:
        input (np.array): Input data to the layer.
        """
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """
        Performs the backward pass through the layer.
        Must be implemented by subclasses.
        
        Parameters:
        output_error (np.array): Gradient of the loss with respect to the output.
        learning_rate (float): Learning rate for updating the layer's parameters.
        """
        raise NotImplementedError


class Affine_Layer(Layer_Primitive):
    """
    Fully connected (affine) layer performing linear transformation with bias.
    """
    def __init__(self, input_size, output_size):
        """
        Initializes weights and biases for the layer.
        
        Parameters:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        """
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        """
        Performs the forward pass by computing weighted input plus bias.
        
        Parameters:
        input_data (np.array): Input data to the layer.
        
        Returns:
        np.array: Output of the affine transformation.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_grad, learning_rate):
        """
        Performs the backward pass by computing gradients and updating weights and biases.
        
        Parameters:
        output_grad (np.array): Gradient of the loss with respect to the output.
        learning_rate (float): Learning rate for updating the parameters.
        
        Returns:
        np.array: Gradient of the loss with respect to the input.
        """
        input_error = np.dot(output_grad, self.weights.T)
        weights_error = np.dot(self.input.T, output_grad)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_grad, axis=0, keepdims=True)
        
        return input_error


class ActivationLayer(Layer_Primitive):
    """
    Layer applying an element-wise activation function.
    """
    def __init__(self, activation, activation_grad):
        """
        Initializes the activation function and its derivative.
        
        Parameters:
        activation (function): Activation function.
        activation_grad (function): Derivative of the activation function.
        """
        self.activation = activation
        self.activation_grad = activation_grad

    def forward_propagation(self, input_data):
        """
        Applies the activation function element-wise.
        
        Parameters:
        input_data (np.array): Input data to the layer.
        
        Returns:
        np.array: Activated output.
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_grad, learning_rate):
        """
        Computes the backward pass by applying the activation gradient.
        
        Parameters:
        output_grad (np.array): Gradient of the loss with respect to the output.
        learning_rate (float): Unused in activation layers.
        
        Returns:
        np.array: Gradient of the loss with respect to the input.
        """
        return self.activation_grad(self.input) * output_grad


# Activation functions and derivatives
def tanh(x):
    """
    Computes the hyperbolic tangent activation function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Output after applying tanh.
    """
    return np.tanh(x)


def tanh_grad(x):
    """
    Computes the derivative of the tanh function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Gradient of tanh.
    """
    return 1 - np.tanh(x) ** 2


def relu(x):
    """
    Computes the ReLU activation function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Output after applying ReLU.
    """
    return np.maximum(0, x)


def relu_grad(x):
    """
    Computes the gradient of ReLU.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Gradient of ReLU.
    """
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    """
    Computes the sigmoid activation function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Output after applying sigmoid.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    """
    Computes the derivative of the sigmoid function.
    
    Parameters:
    x (np.array): Input array.
    
    Returns:
    np.array: Gradient of sigmoid.
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


# Loss functions and gradients
def mse(y_true, y_pred):
    """
    Computes mean squared error loss.
    
    Parameters:
    y_true (np.array): Ground truth values.
    y_pred (np.array): Predicted values.
    
    Returns:
    float: Mean squared error.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_grad(y_true, y_pred):
    """
    Computes the gradient of the MSE loss.


    Parameters:
    y_true (np.array): Ground truth values.
    y_pred (np.array): Predicted values.

    Returns:
    float: mse grad.
    """
    return 2 * (y_pred - y_true) / y_true.size

class MyNetwork:
    """
    A simple feed-forward neural network composed of layers.
    """

    def __init__(self):
        """
        Initializes an empty network.
        """
        self.layers = []
        self.loss = None
        self.loss_grad = None

    def add(self, layer):
        """
        Adds a layer to the network.

        Parameters:
          layer : Layer_Primitive
              A layer to be added to the network.
        """
        self.layers.append(layer)

    def use_loss(self, loss, loss_grad):
        """
        Sets the loss function and its gradient for the network.

        Parameters:
          loss : function
              The loss function to be used.
          loss_grad : function
              The gradient of the loss function.
        """
        self.loss = loss
        self.loss_grad = loss_grad

    def fit(self, x_train, y_train, epochs, learning_rate):
        """
        Trains the network using gradient descent.

        Parameters:
          x_train : np.array
              Training data.
          y_train : np.array
              Training labels.
          epochs : int
              Number of training epochs.
          learning_rate : float
              Learning rate for updating the weights.
        """
        samples = len(x_train)
        print("Training on {:,} samples:".format(samples))
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                grad = self.loss_grad(y_train[j], output)
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, learning_rate)
            err /= samples
            print("Training epoch %d/%d   error=%f" % (i + 1, epochs, err))

    def fit_mini_batch(self, x_train, y_train, batch_size, epochs, learning_rate):
        """
        Trains the network using mini-batch gradient descent.

        Parameters:
          x_train : np.array
              Training data.
          y_train : np.array
              Training labels.
          batch_size : int
              Size of each mini-batch.
          epochs : int
              Number of training epochs.
          learning_rate : float
              Learning rate for updating the weights.
        """
        samples = len(x_train)
        print("Training on {:,} samples:".format(samples))

        for epoch in range(epochs):
            # Shuffle the dataset
            indices = np.random.permutation(samples)
            x = x_train[indices]
            y = y_train[indices]
            epoch_error = 0
            for i in range(0, samples, batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                batch_error = 0
                for j in range(len(x_batch)):
                    output = x_batch[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    batch_error += self.loss(y_batch[j], output)
                    grad = self.loss_grad(y_batch[j], output)
                    for layer in reversed(self.layers):
                        grad = layer.backward_propagation(grad, learning_rate)
                batch_error /= len(x_batch)
                epoch_error += batch_error
            epoch_error /= (samples / batch_size)
            print("Training epoch %d/%d   error=%f" % (epoch + 1, epochs, epoch_error))


    def prof(self, x_train, y_train, epochs=1, learning_rate=1):
        """
        Profiles the performance of the fit method.

        Parameters:
          x_train : np.array
              Training data.
          y_train : np.array
              Training labels.
          epochs : int, optional
              Number of training epochs (default is 1).
          learning_rate : float, optional
              Learning rate for updating the weights (default is 1).
        """
        profile.runctx(
            "self.fit(x_train, y_train, epochs, learning_rate)", globals(), locals()
        )

    def predict(self, x_test, y_test=np.array([])):
        """
        Predicts the output for given input.

        Parameters:
          x_test : np.array
              Test data.
          y_test : np.array, optional
              Test labels (default is an empty array).

        Returns:
          list
              Predicted output for each sample in x_test.
        """
        if y_test.size:
            assert len(x_test) == len(
                y_test
            )  # if Y is given
        samples = len(x_test)
        result = []
        loss = 0
        correct = 0
        for i in range(samples):
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
            if y_test.size:
                loss += self.loss(y_test[i], output)
                target = y_test[i]
                if np.equal(target.argmax(), output.argmax()):
                    correct += 1
        if y_test.size:
            mean_loss = loss / samples
            print(
                "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    mean_loss, correct, samples, 100.0 * correct / samples
                )
            )
        return result

from keras.datasets import mnist
from keras.utils import to_categorical
# load MNIST from server
# Using a standard library (keras.datasets) to load the mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# One-hot encoding of the output.
# Currently a number in range [0,9]; Change into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)
# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)
netMiniGD = MyNetwork()


netMiniGD.add(Affine_Layer(28*28, 128))
netMiniGD.add(ActivationLayer(tanh, tanh_grad))
netMiniGD.add(Affine_Layer(128, 64))
netMiniGD.add(ActivationLayer(tanh, tanh_grad))
netMiniGD.add(Affine_Layer(64, 10))
netMiniGD.add(ActivationLayer(tanh, tanh_grad))

netMiniGD.use_loss(mse, mse_grad)


epoch_num = 10
lr_sched = 0.05 # add a learning rate scheduler of your choice here
t2 = time.time()
netMiniGD.fit_mini_batch(x_train[:10000], y_train[:10000], batch_size=128, epochs=epoch_num, learning_rate=lr_sched)
print(f"Total process time: {round(time.time() - t2,3)}")
