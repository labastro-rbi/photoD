import jax.numpy as jnp
import numpy as np

QUANTILES = (0.16, 0.5, 0.84)


def pnorm(pdf, dx):
    """Normalize a sampled distribution to unit integral."""
    return pdf / jnp.sum(pdf) / dx


def getMargDistr(arr2d, dX, dY):
    """Marginal distributions of a 2D map, along its second and first axis."""
    return pnorm(jnp.sum(arr2d, axis=0), dX), pnorm(jnp.sum(arr2d, axis=1), dY)


def getMargDistr3D(arr3d, dX, dY, dZ):
    """Marginal distributions of a (FeH, Mr, Ar) cube in Mr, FeH and Ar."""
    margX = jnp.sum(arr3d, axis=(0, 2))
    margY = jnp.sum(arr3d, axis=(1, 2))
    margZ = jnp.sum(arr3d, axis=(0, 1))
    return pnorm(margX, dX), pnorm(margY, dY), pnorm(margZ, dZ)


def Entropy(p):
    """Entropy (in bits) of a sampled distribution; empty bins contribute nothing."""
    pOK = jnp.where(p > 0, p, 1)
    return -jnp.sum(pOK * jnp.log2(pOK))


def entropies(pdfs):
    """Entropy of each distribution in a list, with a single logarithm over all of them."""
    p = jnp.concatenate(pdfs)
    p = jnp.where(p > 0, p, 1)
    terms = p * jnp.log2(p)
    edges = np.cumsum([0] + [pdf.size for pdf in pdfs])
    return [-jnp.sum(terms[a:b]) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def getStats(x, pdf):
    """Mean and standard deviation of a sampled distribution."""
    mean = jnp.sum(x * pdf) / jnp.sum(pdf)
    V = jnp.sum((x - mean) ** 2 * pdf) / jnp.sum(pdf)
    return mean, jnp.sqrt(V)


def getPosteriorQuantiles(x, pdf):
    """The 16th, 50th and 84th percentile of a distribution sampled at the sorted values x.

    The cumulative distribution is taken at the bin centers and interpolated linearly, as with jnp.interp.
    The interval is found by counting cdf values <= q, which for a non-decreasing cdf is what jnp.interp's
    searchsorted returns; written this way it compiles much faster inside large jitted functions.
    """
    cumsum = jnp.cumsum(pdf)
    cdf = (cumsum - 0.5 * pdf) / cumsum[-1]
    q = jnp.array(QUANTILES, dtype=cdf.dtype)
    i = jnp.clip(jnp.sum(cdf[None, :] <= q[:, None], axis=1), 1, cdf.size - 1)
    dx = cdf[i] - cdf[i - 1]
    flat = jnp.abs(dx) <= np.spacing(np.finfo(cdf.dtype).eps)
    f = jnp.where(flat, x[i - 1], x[i - 1] + (q - cdf[i - 1]) / jnp.where(flat, 1, dx) * (x[i] - x[i - 1]))
    f = jnp.where(q < cdf[0], x[0], f)
    return jnp.where(q > cdf[-1], x[-1], f)
