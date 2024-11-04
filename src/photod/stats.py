import numpy as np


def pnorm(pdf, dx):
    return pdf / np.sum(pdf) / dx


def getMargDistr(arr2d, dX, dY):
    margX = np.sum(arr2d, axis=0)
    margY = np.sum(arr2d, axis=1)
    return pnorm(margX, dX), pnorm(margY, dY)


def getMargDistr3D(arr3d, dX, dY, dZ):
    margX = np.sum(arr3d, axis=(0, 2))
    margY = np.sum(arr3d, axis=(1, 2))
    margZ = np.sum(arr3d, axis=(0, 1))
    return pnorm(margX, dX), pnorm(margY, dY), pnorm(margZ, dZ)


def Entropy(p):
    pOK = p[p > 0]
    return -np.sum(pOK * np.log2(pOK))


def getStats(x, pdf):
    mean = np.sum(x * pdf) / np.sum(pdf)
    V = np.sum((x - mean) ** 2 * pdf) / np.sum(pdf)
    return mean, np.sqrt(V)
