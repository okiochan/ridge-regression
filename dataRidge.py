import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import math
from sklearn import datasets

class FakeData:
    def GenerateSample(self):
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

class SkikitData:
    def GenerateSample(self):
        print("hello")
        X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
        Y = np.ones(10)
        return X,Y

class DataBuilder:
    def Build(self, name):
        if name == "fake":
            x, y = FakeData().GenerateSample()
            return x, y
        elif name == "ski":
            print("sdf")
            x, y = SkikitData().GenerateSample()
            return x, y
        else:
            assert("Unknown data")