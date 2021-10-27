from numpy.core.numeric import correlate
from nn import Dense_layer
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('Snails.csv')


def splitData(data, split_ratio=[0.50, 0.50]):
    train = data.sample(frac=split_ratio[0])
    val = data.drop(train.index)
    return (train, val)


train, test = splitData(df)
test.reset_index(inplace=True, drop=True)
train.reset_index(inplace=True, drop=True)
gender_dataset_train = train['sex']
gender_dataset_test = test['sex']

train = train.drop('sex', axis=1)
train = train.drop('Id', axis=1)
train = train.to_numpy()
train = (train - train.min(axis=0)) / (train.max(axis=0) - train.min(axis=0))


test = test.drop('sex', axis=1)
test = test.drop('Id', axis=1)
test = test.to_numpy()
test = (test - test.min(axis=0)) / (test.max(axis=0) - test.min(axis=0))

Y_train = np.zeros((len(train), 3))
for i in range(len(train)):
    if gender_dataset_train[i] == 'M':
        Y_train[i][0] = 1
    elif gender_dataset_train[i] == 'F':
        Y_train[i][1] = 1
    else:
        Y_train[i][2] = 1

Y_test = np.zeros((len(test), 3))
for i in range(len(test)):
    if gender_dataset_test[i] == 'M':
        Y_test[i][0] = 1
    elif gender_dataset_test[i] == 'F':
        Y_test[i][1] = 1
    else:
        Y_test[i][2] = 1

neural_net = Dense_layer(input_features=8, layer1_nodes=8,
                         output_labels=3, learning_rate=0.001, iterations=500)

for j in range(500):

    loss = neural_net.train(train, Y_train)
    loss = (0.5*loss)/train.shape[0]
    if (j+1) % 50 == 0:
        print(f'The loss after {j+1}th iteration is = {loss}')


neural_net.forward(train)
pred = neural_net.activated_output_layer2
correct = 0
for i in range(len(train)):
    if np.where(pred[i] == max(pred[i])) == np.where(Y_train[i] == max(Y_train[i])):
        correct += 1

print("The training accuracy is = ",correct/len(train))


neural_net.forward(test)
pred = neural_net.activated_output_layer2
correct = 0
for i in range(len(test)):
    if np.where(pred[i] == max(pred[i])) == np.where(Y_test[i] == max(Y_test[i])):
        correct += 1

print("The test accuracy is = ",correct/len(test))
