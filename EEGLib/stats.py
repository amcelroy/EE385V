import numpy as np


class Stats:
    def cross_correlation(self, dataset1=np.ndarray, dataset2=np.ndarray, var1=np.ndarray, var2=np.ndarray):
        t = dataset1 - dataset2
        t = t ** 2
        t = t ** .5
        t /= (var1 * var2)
        return t

    def squared_error(self, dataset1=np.ndarray, dataset2=np.ndarray):
        t = dataset1 - dataset2
        t = t ** 2
        t = t ** .5
        return t

    def error(self, dataset1=np.ndarray, dataset2=np.ndarray):
        t = dataset1 - dataset2
        return t
