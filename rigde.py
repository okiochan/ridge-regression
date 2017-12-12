import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def RegressionInv(X):
    return X.T.dot(X)

def LSRegression(X,y):
    l = X.shape[0]
    n = X.shape[1]
    
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    res = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]

def GenerateSample():
    l = 50
    real = 2
    X = np.random.randn(l,real)

    a = np.random.randn(real)
    y = np.zeros(l)
    for i in range(l):
        y[i] = a.dot(X[i,:])

    # fake parameters
    F = 2 * X
    Xp = np.concatenate((X,F),axis=1)

    return Xp, y

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

X, Y = GenerateSample()
n = X.shape[1]
Cov = X.T.dot(X)
E, V = np.linalg.eigh(Cov)

# print ("\ncov matrix")
# print(Cov)
# print("\n\neigen values for cov matrix")
# print(E)
#print(V)

c = 0.001
Cov2 = Cov + c * np.eye(n)
e, v = np.linalg.eigh(Cov2)
print("\nnew eigen values")
print(e)
print("\nsolution for w")
print(np.linalg.inv(Cov2).dot(X.T).dot(Y))
