import numpy as np
from numpy import linalg


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM(object):
    # 数组的flatten 和 ravel 方法将数组变为一个一维向量
    a = np.ravel()

    # 未完成svm
