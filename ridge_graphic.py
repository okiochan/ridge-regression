import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dataRidge

np.set_printoptions(formatter={'float':lambda x: '%.4f' % x})

X, Y = dataRidge.DataBuilder().Build("ski")
n_alphas = 100
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
Cov = X.T.dot(X)
for C in alphas:
    #get new cov matrix
    Cov2 = Cov + C * np.eye(X.shape[1])
    #solve regression
    w = np.linalg.inv(Cov2).dot(X.T).dot(Y)
    coefs.append(np.abs(w))

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
    
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
