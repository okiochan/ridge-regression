import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dataRidge
import pca

def solve_linreg(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def Quality(w, X, y):
    res = 0
    l = X.shape[0]
    for i in range(l):
        res += (w.dot(X[i]) - y[i]) ** 2 / 2
    return res

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

X, Y = X, Y = dataRidge.DataBuilder().Build("fake")
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
w = np.linalg.inv(Cov2).dot(X.T).dot(Y)
print(w)

pca.ShowPercentage(X)

X_hat = pca.GetComponents(X,2)
w_hat = solve_linreg(X_hat, Y)

print("\nsolution for w after using PCA")
print(w_hat)
# print(X_hat)

print("\n\nSSE for PCA")
print(Quality(w_hat,X_hat, Y))
print("\nSSE for Ridge")
print(Quality(w, X, Y))