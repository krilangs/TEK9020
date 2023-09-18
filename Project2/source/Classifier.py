# -*- coding: utf-8 -*-
import numpy as np

def g_q(x, W, w, w0):
    """
    Quadratic discriminant function.
    """
    return x@W@x.T + w.T@x.T + w0

def MinErrorRate(train1, train2, train3, test_data):
    """
    Use the minimum error rate classifier, assuming a normal distribution, to
    do a segmentation task on some given test image containing classes as
    trained on for some given training data.
    """
    # Number of features
    feat = train1.shape[2]

    # Number of class objects
    n = test_data.shape[0]*test_data.shape[1]
    n1 = train1.shape[0]*train1.shape[1]
    n2 = train2.shape[0]*train2.shape[1]
    n3 = train3.shape[0]*train3.shape[1]

    # A priori probability of the three classes
    P_omega1 = n1/n
    P_omega2 = n2/n
    P_omega3 = n3/n

    # Maximum likelihood estimation expectation vector for the classes
    mu1 = 1/n1*np.sum(np.sum(train1, axis=0), axis=0)
    mu2 = 1/n2*np.sum(np.sum(train2, axis=0), axis=0)
    mu3 = 1/n3*np.sum(np.sum(train3, axis=0), axis=0)

    # Maximum likelihood estimation covariance matrix for the classes
    cov1_diff = 1/n1*(train1 - mu1)
    cov2_diff = 1/n2*(train2 - mu2)
    cov3_diff = 1/n3*(train3 - mu3)
    cov1 = np.zeros((feat, feat))
    cov2 = np.zeros((feat, feat))
    cov3 = np.zeros((feat, feat))

    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            cov1 += cov1_diff[i, j].reshape(-1, 1) @ cov1_diff[i, j].reshape(-1, 1).T
    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            cov2 += cov2_diff[i, j].reshape(-1, 1) @ cov2_diff[i, j].reshape(-1, 1).T
    for i in range(train3.shape[0]):
        for j in range(train3.shape[1]):
            cov3 += cov3_diff[i, j].reshape(-1, 1) @ cov3_diff[i, j].reshape(-1, 1).T

    ## Components in the three class discriminant function
    # dxd matrices
    W1 = -0.5*np.linalg.pinv(cov1)
    W2 = -0.5*np.linalg.pinv(cov2)
    W3 = -0.5*np.linalg.pinv(cov3)

    # dx1 vectors
    w1 = np.linalg.pinv(cov1)@mu1.T
    w2 = np.linalg.pinv(cov2)@mu2.T
    w3 = np.linalg.pinv(cov3)@mu3.T

    # 1x1 scalars
    w01 = -0.5*mu1 @ np.linalg.pinv(cov1) @ mu1.T\
        -0.5*np.log(np.linalg.det(cov1)) + np.log(P_omega1)
    w02 = -0.5*mu2 @ np.linalg.pinv(cov2) @ mu2.T\
        -0.5*np.log(np.linalg.det(cov2)) + np.log(P_omega2)
    w03 = -0.5*mu3 @ np.linalg.pinv(cov3) @ mu3.T\
        -0.5*np.log(np.linalg.det(cov3)) + np.log(P_omega3)

    segmentation = np.zeros((test_data.shape[0], test_data.shape[1], 3))
    # Set class colors for segmentation
    classes = [[0.85, 0.1, 0.87], [0.9, 0.8, 0.3], [0.07, 0.6, 0.5]]
    for i in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            g1 = g_q(test_data[i, j], W1, w1, w01)
            g2 = g_q(test_data[i, j], W2, w2, w02)
            g3 = g_q(test_data[i, j], W3, w3, w03)
            g_arr = np.array([g1, g2, g3])
            idx = np.argmax(g_arr)
            segmentation[i, j] = classes[idx]

    return segmentation