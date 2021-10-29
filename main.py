from numpy.core.numeric import correlate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nn import Dense_layer
from create_dataset import split_dataset
from plots import scatter_plot, scatter_plot_normalized

'''
We plot the features with respect to the sex using scatter plot
'''
df = pd.read_csv('Snails.csv')

feature_index = {
    'length': 1,
    'diameter': 2,
    'height': 3,
    'whole-weight': 4,
    'shucked-weight': 5,
    'viscera-weight': 6,
    'shell-weight': 7,
    'rings': 8
}

# We give any two features we want to plot the scatter plot for
feature_1_column = feature_index['diameter']  # We select 'diameter'
feature_2_column = feature_index['rings']  # We then select 'rings'

scatter_plot(df, df['sex'], feature_1_column, feature_2_column)
scatter_plot_normalized(df, df['sex'], feature_1_column, feature_2_column)


'''
We define our model here

input_features=8
layer1_nodes=8, since there are 8 nodes in the hidden layer
output_labels=3, since there are 3 labels, M,F and I
learning_rate=0.01
iterations=500
'''
neural_net = Dense_layer(input_features=8, layer1_nodes=8,
                         output_labels=3, learning_rate=0.01, iterations=500)
train, Y_train, test, Y_test = split_dataset(df, ratio=0.8)

EPOCHS = 500
loss = []  # Store the loss after each epoch
accuracy_train = []  # Store the training accuracy after each epoch
accuracy_test = []  # Store the testing accuracy after each epoch

for j in range(EPOCHS):
    epoch_loss = neural_net.train(train, Y_train)
    training_accuracy = neural_net.calc_accuracy(train, Y_train)
    test_accuracy = neural_net.calc_accuracy(test, Y_test)

    loss.append(epoch_loss)
    accuracy_train.append(training_accuracy)
    accuracy_test.append(test_accuracy)


'''
Training accuracy
In this we pass the training dataset and calculate accuracy
'''
print("Training accuracy = ", neural_net.calc_accuracy(train, Y_train))


'''
Test accuracy
In this we pass the test dataset and calculate accuracy
'''
print("Test accuracy = ", neural_net.calc_accuracy(test, Y_test))


'''
We plot the
1. Loss vs Epoch
2. Training accuracy vs Epoch
3. Testing accuracy vs Epoch
'''
plt.figure(num=3)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch number/iteration")
plt.ylabel("Normalized Loss")
plt.plot(loss)

plt.figure(num=4)
plt.title("Training accuracy vs Epoch")
plt.xlabel("Epoch number/iteration")
plt.ylabel("Training accuracy")
plt.plot(accuracy_train)


plt.figure(num=5)
plt.title("Test accuracy vs Epoch")
plt.xlabel("Epoch number/iteration")
plt.ylabel("Test accuracy")
plt.plot(accuracy_test)
plt.show()
