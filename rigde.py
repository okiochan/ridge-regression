import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dataRidge
import pca

def RidgeRegression(X,y,C):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn linear MNK
    res = np.linalg.inv(X.T.dot(X) + np.eye(n+1) * C).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def solve_linreg(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def Quality(w, w0, X, y):
    res = 0
    l = X.shape[0]
    for i in range(l):
        res += (w.dot(X[i]) + w0 - y[i]) ** 2
    return res / l / 2

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

X, Y = dataRidge.DataBuilder().Build("fake")
X = pca.Normalize(X)
n = X.shape[1]
Cov = X.T.dot(X)
E, V = np.linalg.eigh(Cov)

print ("\ncov matrix")
print(Cov)
print("\n\neigen values for cov matrix")
print(E)

c = 0.001
Cov2 = Cov + c * np.eye(n)
e, v = np.linalg.eigh(Cov2)
print("\nnew eigen values")
print(e)

print("\nsolution for w after using Ridge")
w, w0 = RidgeRegression(X, Y, c)
print(w, w0)

pca.ShowPercentage(X)

X_hat = pca.GetComponents(X,2)
w_hat, w0_hat = RidgeRegression(X_hat, Y, c)

print("\nsolution for w after using PCA")
print(w_hat)
print(w0_hat)
# print(X_hat)

print("\n\nSSE for PCA")
print(Quality(w_hat, w0_hat,X_hat, Y))
print("\nSSE for Ridge")
print(Quality(w, w0, X, Y))