from numpy.core.numeric import correlate
from nn import Dense_layer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(df, gender_label, column1, column2):
    df_numpy = df.to_numpy()
    plt.figure(num=1)
    color = ['blue', 'red', 'yellow']
    for i in range(3):
        if i == 0:
            gender = 'M'
        elif i == 1:
            gender = 'F'
        elif i == 2:
            gender = 'I'
        x = []
        y = []
        for j in range(len(gender_label)):
            if gender_label[j] == gender:
                x.append(df_numpy[j][column1])
                y.append(df_numpy[j][column2])
        plt.scatter(x, y, c=color[i], label=gender)
    plt.legend(loc='best')
    plt.xlabel(df.columns[column1])
    plt.ylabel(df.columns[column2])
    plt.title("Scatter plot between " +
              df.columns[column1] + " and " + df.columns[column2])


def scatter_plot_normalized(df, gender_label, column1, column2):
    # We drop the 'sex' axis so that we can normalize the dataframe
    df_dropped = df.drop('sex', axis=1)
    df_numpy = df_dropped.to_numpy()  # We convert it to numpy array
    df_norm = (df_numpy - df_numpy.min(axis=0)) / (df_numpy.max(axis=0) -
                                                   df_numpy.min(axis=0))  # Now we normalize the datapoints

    plt.figure(num=2)
    # We take these three colours for 3 different genders
    color = ['blue', 'red', 'yellow']
    for i in range(3):
        if i == 0:
            gender = 'M'
        elif i == 1:
            gender = 'F'
        elif i == 2:
            gender = 'I'
            # We use
        x = []
        y = []
        for j in range(len(gender_label)):
            if gender_label[j] == gender:
                x.append(df_norm[j][column1])
                y.append(df_norm[j][column2])
        plt.scatter(x, y, c=color[i], label=gender)
    plt.legend(loc='best')
    plt.xlabel(df.columns[column1])
    plt.ylabel(df.columns[column2])
    plt.title("Normalized scatter plot between " +
              df.columns[column1] + " and " + df.columns[column2])
