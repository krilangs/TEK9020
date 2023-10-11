# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Classifiers import LeastSquares, MinErrorRate, NearestNeighbour


def plot_3d_ds2():
    """
    Plot the 3D feature space for data set 2.
    """
    filename = "data/ds-2.txt"
    data = pd.read_csv(filename, header=None, sep="\s+", engine="python")
    data = data.to_numpy()

    fig = plt.figure()
    ax =  fig.add_subplot(projection="3d")
    class1 = data[data[:, 0] == 1]
    class2 = data[data[:, 0] == 2]
    ax.scatter(class1[:, 1], class1[:, 2], class1[:, 3], label="Klasse 1")
    ax.scatter(class2[:, 1], class2[:, 2], class2[:, 3], label="Klasse 2")
    ax.set_xlabel("Egenskap 1", size=11)
    ax.set_ylabel("Egenskap 2", size=11)
    ax.set_zlabel("Egenskap 3", size=11)
    ax.set_title("Egenskapsrommet for datasett 2", size=14)
    plt.legend(loc="best")
    plt.show()

def train_test_split(data):
    """
    Split the data set into training and test data.
    For simplicity; training data are the even objects while test data are the
    odd objects (by Pythonic numbering).
    """
    train = data[::2].to_numpy()
    test = data[1::2].to_numpy()

    return train, test

def best_combination(err_rate, idxs):
    """
    Find the best feature combination according to smallest error rate.
    """
    best_idx = np.argmin(err_rate)
    best_comb_idx = idxs[best_idx]

    return err_rate[best_idx], best_comb_idx


if __name__=="__main__":
    """
    1) Compute the error rate using the Nearest Neighbour classifier to find
    the best combination of features for each dimension of features,
    per data set.
    2) For the best feature combinations in each feature dimension, find the
    best classifier between the Nearest Neighbour, Least Squares and
    Minimum Error Rate classifiers, per data set.
    """
    #plot_3d_ds2()

    # 1)
    folder = "data/"
    #filename = folder+"ds-1.txt"
    #filename = folder+"ds-2.txt"
    filename = folder+"ds-3.txt"
    data = pd.read_csv(filename, header=None, sep="\s+", engine="python")
    train, test = train_test_split(data)

    if train.shape[1] == 5:
        err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4, \
        idx_d1, idx_d2, idx_d3, idx_d4 = NearestNeighbour(train, test)

        error_rate = [err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4]
        index = [idx_d1, idx_d2, idx_d3, idx_d4]
    else:
        err_rate_d1, err_rate_d2, err_rate_d3, \
        idx_d1, idx_d2, idx_d3 = NearestNeighbour(train, test)

        error_rate = [err_rate_d1, err_rate_d2, err_rate_d3]
        index = [idx_d1, idx_d2, idx_d3]

    print("Dim 1:", err_rate_d1)
    print("Dim 2:", err_rate_d2)
    print("Dim 3:", err_rate_d3)
    if train.shape[1] == 5:
        print("Dim 4:", err_rate_d4)

    # 2)
    for err_rate, idx in zip(error_rate, index):
        NN_err_rate, best_comb = best_combination(err_rate, idx)
        minimum_err_rate = MinErrorRate(train, test, best_comb)
        ls_err_rate = LeastSquares(train, test, best_comb)

        print("-------------------------------------------")
        print(f"Best combination: \t {best_comb}")
        print(f"Nearest neighbor: \t {NN_err_rate:.3}")
        print(f"Minimum error rate: \t {minimum_err_rate:.3}")
        print(f"Least squares: \t \t {ls_err_rate:.3}")
        print("-------------------------------------------\n")