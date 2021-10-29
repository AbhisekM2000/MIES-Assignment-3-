import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# Function to split the dataset into a given ratio ( ratio = percentage of training set)
def split_dataset(df, ratio):
    # We split the data as per the given ratio
    train = df.sample(frac=ratio)
    test = df.drop(train.index)

    # Then we make the indexing of each in a sequential manner
    test.reset_index(inplace=True, drop=True)  # Make proper indexing
    train.reset_index(inplace=True, drop=True)  # Make proper indexing

    # We store the gender for the train and test set separately
    gender_dataset_train = train['sex']
    gender_dataset_test = test['sex']

    # We drop the gender/sex column and the Id column from both train and test dataset
    train = train.drop('sex', axis=1)
    train = train.drop('Id', axis=1)

    test = test.drop('sex', axis=1)
    test = test.drop('Id', axis=1)

    # Then we convert the train and test set to numpy array and then we normalize the whole dataset
    train = train.to_numpy()
    train = (train - train.min(axis=0)) / \
        (train.max(axis=0) - train.min(axis=0))
    test = test.to_numpy()
    test = (test - test.min(axis=0)) / (test.max(axis=0) - test.min(axis=0))

    # Finally we form the output vector(actual output labels) which is one-hot encoded that is 'M'=[1,0,0], 'F'=[0,1,0] and 'I'=[0,0,1]
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

    return train, Y_train, test, Y_test
