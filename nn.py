import numpy as np
import pandas as pd
import math
from tqdm import tqdm


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

    def sigmoid(self, input):
        return 1/(1+np.exp(-input))

    def sigmoid_derivative(self, input):
        return np.exp(-input) / ((1 + np.exp(-input)) ** 2)

    def forward(self, input):
        # The below is the output of layer 1 (hidden layer ), first its passed through a linear layer and then sigmoid activated
        self.output_layer1 = np.dot(
            input, self.weights_layer1)+self.bias_layer1
        self.activated_output_layer1 = self.sigmoid(self.output_layer1)

        # The below is the output of layer 2, first its passed through a linear layer and then sigmoid activated
        self.output_layer2 = np.dot(
            self.activated_output_layer1, self.weights_layer2)+self.bias_layer2
        self.activated_output_layer2 = self.sigmoid(self.output_layer2)

        return np.argmax(self.activated_output_layer2)

    def calc_loss(self, predicted_label, actual_label):
        return np.sum(np.square((predicted_label-actual_label)))

    def backward(self, input, y_pred, y_actual):

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

    def update_parameters(self, dW1, db1, dW2, db2):
        self.weights_layer1 -= (self.learning_rate * dW1)
#         print(self.bias_layer1.shape)
#         print(db1.shape)
        self.bias_layer1 -= (self.learning_rate*db1)
        self.weights_layer2 -= (self.learning_rate * dW2)
        self.bias_layer2 -= (self.learning_rate*db2)
        # print("Weights are updated")

    def train(self, input, y_actual):
        self.forward(input)
        loss = self.calc_loss(self.activated_output_layer2, y_actual)
        self.backward(input, self.activated_output_layer2, y_actual)
        self.update_parameters(self.layer1_weight_derivative, self.layer1_bias_derivative,
                               self.layer2_weight_derivative, self.layer2_bias_derivative)

        return loss

    def check_output(self, input, y_actual):
        pred_output = self.forward(input)
        actual_output = np.argmax(y_actual)
        if actual_output == pred_output:
            return True
        else:
            return False
