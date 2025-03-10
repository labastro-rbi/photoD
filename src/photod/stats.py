import jax.numpy as jnp

def pnorm(pdf, dx):
    return pdf / jnp.sum(pdf) / dx


def getMargDistr(arr2d, dX, dY):
    margX = jnp.sum(arr2d, axis=0)
    margY = jnp.sum(arr2d, axis=1)
    return pnorm(margX, dX), pnorm(margY, dY)


def getMargDistr3D(arr3d, dX, dY, dZ):
    margX = jnp.sum(arr3d, axis=(0, 2))
    margY = jnp.sum(arr3d, axis=(1, 2))
    margZ = jnp.sum(arr3d, axis=(0, 1))
    return pnorm(margX, dX), pnorm(margY, dY), pnorm(margZ, dZ)


def Entropy(p):
    # Because we cannot filter non-concrete arrays, 1 because log is 0
    pOK = jnp.where(p > 0, p, 1)
    return -jnp.sum(pOK * jnp.log2(pOK))


def getStats(x, pdf):
    mean = jnp.sum(x * pdf) / jnp.sum(pdf)
    V = jnp.sum((x - mean) ** 2 * pdf) / jnp.sum(pdf)
    return mean, jnp.sqrt(V)


def getPosteriorQuantiles(x, pdf):
    cumsum = jnp.cumsum(pdf)
    cdf = (cumsum - 0.5 * pdf) / cumsum[-1]
    quantiles = jnp.array([0.14, 0.5, 0.86])
    return jnp.interp(quantiles, cdf, x)

def getQrQuantiles(postCube, QrGrid, QrIndices):
    margPdfOverFeH = jnp.sum(postCube, axis=(0)).T
    weightsQr = jnp.zeros_like(QrGrid)
    weightsQr = weightsQr.at[QrIndices].add(margPdfOverFeH)
    cumsum = jnp.cumsum(weightsQr)
    cdf = (cumsum - 0.5 * weightsQr) / cumsum[-1]
    quantiles = jnp.array([0.14, 0.5, 0.86])
    QrQuantiles = jnp.interp(quantiles, cdf, QrGrid)
    return QrQuantiles
