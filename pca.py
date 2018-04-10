import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

def Normalize(X):
    X = X-X.mean(axis=0)
    return X
    
def ShowPercentage(X):
    print("\n\nPercentage of saved information PCA")
    L, U = SingularDecomposition(X)
    L /= np.sum(L)
    for i in range(1,L.size,1):
        L[i] += L[i-1]
    print([L[i] for i in range(0,L.size,1)])

def GetComponents(X, m):
    X = Normalize(X)
    L, U = SingularDecomposition(X)
    G = X.dot(U)
    return G[:,np.arange(m)]

def SingularDecomposition(X):
    l = X.shape[0]
    n = X.shape[1]
    Xcov = np.dot(X.T,X)/l
    L, U = np.linalg.eigh(Xcov)
    reverse = n - 1 - np.arange(n)
    return L[reverse], U[:,reverse]

