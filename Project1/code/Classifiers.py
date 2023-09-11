# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import comb

def g(x, a):
    """
    Discriminant function for linear 2-class classifier.
    """
    y = np.append(1, x)
    return a.T @ y.T

def g_q(x, W, w, w0):
    """
    Quadratic discriminant function.
    """
    return x@W@x.T + w.T@x.T + w0

#----------------------------------------------------
def MinErrorRate(train_data, test_data, feature_idx):
    """
    Compute the classification error rate for the best feature combination with
    the minimum error rate classifier assuming a normal distribution.
    """
    # Skip class dimension
    train_obj = train_data[:, 1:]
    test_obj = test_data[:, 1:]

    # Get features at the place of the best feature combinations
    train_obj = train_obj[:, feature_idx]
    test_obj = test_obj[:, feature_idx]

    # Differentiate between the two classes
    idx1 = train_data[:, 0] == 1
    idx2 = train_data[:, 0] == 2
    train1 = train_obj[idx1]
    train2 = train_obj[idx2]

    # Number of class objects in the training data
    n = train_data.shape[0]
    n1 = np.sum(idx1)
    n2 = np.sum(idx2)

    # A priori probability of the two classes
    P_omega1 = n1/n
    P_omega2 = n2/n

    # Maximum likelihood estimation expectation vector for the classes
    mu1 = 1/n1*np.sum(train1, axis=0)
    mu2 = 1/n2*np.sum(train2, axis=0)

    # Maximum likelihood estimation covariance matrix for the classes
    cov1 = 1/n1*(train1 - mu1).T@(train1 - mu1)
    cov2 = 1/n2*(train2 - mu2).T@(train2 - mu2)

    # Components in the two class discriminant function
    # dxd matrices
    W1 = -0.5*np.linalg.pinv(cov1)
    W2 = -0.5*np.linalg.pinv(cov2)

    # dx1 vectors
    w1 = np.linalg.pinv(cov1)@mu1.T
    w2 = np.linalg.pinv(cov2)@mu2.T

    # 1x1 scalars
    w01 = -0.5*mu1 @ np.linalg.pinv(cov1) @ mu1.T\
        -0.5*np.log(np.linalg.det(cov1)) + np.log(P_omega1)
    w02 = -0.5*mu2 @ np.linalg.pinv(cov2) @ mu2.T\
        -0.5*np.log(np.linalg.det(cov2)) + np.log(P_omega2)

    err_rate = 0
    for i in range(test_obj.shape[0]):
        g1 = g_q(test_obj[i], W1, w1, w01)
        g2 = g_q(test_obj[i], W2, w2, w02)
        g_diff = g1 - g2

        # Check if misclassification using discriminant function
        if g_diff >= 0:
            class_err = test_data[i, 0] != 1
        else:
            class_err = test_data[i, 0] != 2

        err_rate += class_err
    err_rate /= test_obj.shape[0]

    return err_rate

#-------------------------------------------------------------
def LeastSquares(train_data, test_data, feature_idx):
    """
    Compute the classification error rate for the best feature combination with
    the least squares method.
    """
    # Skip class dimension
    train_obj = train_data[:, 1:]
    test_obj = test_data[:, 1:]

    # Get features at the place of the best feature combinations
    train_obj = train_obj[:, feature_idx]
    test_obj = test_obj[:, feature_idx]

    # Differentiate between the two classes
    idx1 = train_data[:, 0] == 1
    idx2 = train_data[:, 0] == 2

    # Matrix containing expanded training feature vectors
    Y = np.c_[np.ones((train_obj.shape[0], 1)), train_obj]
    b = idx1 + (-1)*idx2

    # Weight vector
    a = np.linalg.pinv(Y.T @ Y) @ Y.T @ b

    err_rate = 0
    for i in range(test_obj.shape[0]):
        g_disc = g(test_obj[i], a)

        # Check if misclassification using discriminant function
        if g_disc >= 0:
            class_err = test_data[i, 0] != 1
        else:
            class_err = test_data[i, 0] != 2

        err_rate += class_err
    err_rate /= test_obj.shape[0]

    return err_rate

#-----------------------------------------------------
def NearestNeighbour(train_data, test_data):
    """
    Compute the classification error rate for all feature combinations with
    the nearest neighbour classifier.
    """
    # Skip class dimension
    train_obj = train_data[:, 1:]
    test_obj = test_data[:, 1:]

    # Number of features
    features = test_obj.shape[1]

    # Error rate for all combinations of features
    err_rate_d1 = np.zeros(features)
    err_rate_d2 = np.zeros(comb(features, 2, exact=True))
    err_rate_d3 = np.zeros(comb(features, 3, exact=True))

    if features == 4:
        err_rate_d4 = np.zeros(1)
        idx_d4 = []

    # Indices of feature combinations
    idx_d1 = []
    idx_d2 = []
    idx_d3 = []

    # Calculate the error rates for each combination of features for each dimension
    for test_idx in range(test_obj.shape[0]):
        test = test_obj[test_idx]
        d2_idx = 0
        d3_idx = 0

        for i in range(features): # 1D
            diff = np.abs(test[i] - train_obj[:, i])
            index = np.argmin(diff)
            class_err = test_data[test_idx, 0] != train_data[index, 0]
            err_rate_d1[i] += class_err
            idx_d1.append([i])

            for j in range(i+1, features): # 2D
                diff = np.linalg.norm(test[[i,j]] - train_obj[:, [i,j]], axis=1)
                index = np.argmin(diff)
                class_err = test_data[test_idx, 0] != train_data[index, 0]
                err_rate_d2[d2_idx] += class_err
                idx_d2.append([i,j])
                d2_idx += 1

                for k in range(j+1, features): # 3D
                    diff = np.linalg.norm(test[[i,j,k]] \
                                          - train_obj[:, [i,j,k]], axis=1)
                    index = np.argmin(diff)
                    class_err = test_data[test_idx, 0] != train_data[index, 0]
                    err_rate_d3[d3_idx] += class_err
                    idx_d3.append([i,j,k])
                    d3_idx += 1

                    for l in range(k+1, features): #4D
                        diff = np.linalg.norm(test[[i,j,k,l]] \
                                              - train_obj[:, [i,j,k,l]], axis=1)
                        index = np.argmin(diff)
                        class_err = test_data[test_idx, 0] != train_data[index, 0]
                        err_rate_d4[0] += class_err
                        idx_d4.append([i,j,k,l])

    n_objects = test_obj.shape[0]
    err_rate_d1 /= n_objects
    err_rate_d2 /= n_objects
    err_rate_d3 /= n_objects

    if features == 4:
        err_rate_d4 /= n_objects
        return err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4,\
            idx_d1, idx_d2, idx_d3, idx_d4

    return err_rate_d1, err_rate_d2, err_rate_d3, idx_d1, idx_d2, idx_d3
