import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class Dense_layer():
    def __init__(self, input_features, layer1_nodes, output_labels, learning_rate, iterations):
        self.weights_layer1 = np.random.rand(
            input_features, layer1_nodes)*0.1  # 8x8 dimension
        self.bias_layer1 = np.zeros((1, layer1_nodes))  # 1x8 dimension
        self.weights_layer2 = np.random.rand(
            layer1_nodes, output_labels)*0.1  # 8x3 dimension
        self.bias_layer2 = np.zeros((1, output_labels))  # 1x3 dimension
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, input):  # Calculates the sigmoid for a given input
        return 1/(1+np.exp(-input))

    # Calculates the sigmoid derivative for a given input
    def sigmoid_derivative(self, input):
        return np.exp(-input) / ((1 + np.exp(-input)) ** 2)

    def forward(self, input):
        # The below is the output of layer 1 (hidden layer ), first its passed through a linear layer and then sigmoid activated
        # a1 in the above equation
        self.output_layer1 = np.dot(
            input, self.weights_layer1)+self.bias_layer1
        self.activated_output_layer1 = self.sigmoid(
            self.output_layer1)  # z1 in the above equation

        # The below is the output of layer 2, first its passed through a linear layer and then sigmoid activated
        # a2 in the above equation
        self.output_layer2 = np.dot(
            self.activated_output_layer1, self.weights_layer2)+self.bias_layer2
        self.activated_output_layer2 = self.sigmoid(
            self.output_layer2)  # a2 in the above equation

    def calc_loss(self, predicted_label, actual_label):  # Function to calculate the loss
        len = predicted_label.shape[0]
        return (0.5*np.sum(np.square((predicted_label-actual_label))))/len

    def backward(self, input, y_pred, y_actual):  # Function to calculate the derivatives

        # Calculation of the derivatives of loss
        self.loss_derivative2 = np.multiply(
            (y_pred-y_actual), self.sigmoid_derivative(self.output_layer2))
        self.loss_derivative1 = np.dot(
            self.loss_derivative2, self.weights_layer2.T)*self.sigmoid_derivative(self.output_layer1)

        self.layer2_weight_derivative = np.dot(
            self.activated_output_layer1.T, self.loss_derivative2)
        self.layer2_bias_derivative = 1 / \
            input.shape[0] * \
            (np.sum(self.loss_derivative2, axis=0).reshape(1, 3))

        self.layer1_weight_derivative = np.dot(input.T, self.loss_derivative1)
        self.layer1_bias_derivative = 1 / \
            input.shape[0] * \
            (np.sum(self.loss_derivative1, axis=0).reshape(1, 8))

    # In this function we update the parameters
    # The weights are updates using Batch Gradient descent algorithm
    def update_parameters(self, dW1, db1, dW2, db2):
        self.weights_layer1 -= (self.learning_rate * dW1)
        self.bias_layer1 -= (self.learning_rate*db1)
        self.weights_layer2 -= (self.learning_rate * dW2)
        self.bias_layer2 -= (self.learning_rate*db2)

    def train(self, input, y_actual):  # Function to train the model
        self.forward(input)  # We first forward pass
        # Calculate and store the loss generated
        loss = self.calc_loss(self.activated_output_layer2, y_actual)
        # Then we calculate the derivatives from the Loss function
        self.backward(input, self.activated_output_layer2, y_actual)
        self.update_parameters(self.layer1_weight_derivative, self.layer1_bias_derivative, self.layer2_weight_derivative,
                               self.layer2_bias_derivative)  # Then we backpropagate and update the model parameters(weights and biases)

        return loss

    def predict(self, input):  # Predicts the label for a given input
        # We for ward pass the input into the neural network
        self.forward(input)
        # find the index with maximum value in the 1x3 numpy vector
        index = np.argmax(self.activated_output_layer2)
        if index == 0:
            return 'M'
        elif index == 1:
            return 'F'
        else:
            return 'I'

    # Used to calculate accuracy for any given dataset with input as dataset and y_actual as the output actual labels
    def calc_accuracy(self, input, y_actual):
        # We for ward pass the input into the neural network
        self.forward(input)
        # The last layer gives us the predicted values
        y_predicted = self.activated_output_layer2
        correct = 0
        # We calculate the labels identified correctly by our network
        for i in range(len(input)):
            if np.where(y_predicted[i] == max(y_predicted[i])) == np.where(y_actual[i] == max(y_actual[i])):
                correct += 1

        return (correct/len(input))
