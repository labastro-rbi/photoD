from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

QUANTILES = (0.16, 0.5, 0.84)
# the multimodality is measured at this many cuts of the density, evenly spaced from zero to its peak, with
# the two ends left out: at zero every bin is above the cut, and at the peak only the peak is
MULTIMODALITY_LEVELS = 300
# cuts evaluated in one step, per distribution of a batch. A DP2 posterior needs 27 on average.
MULTIMODALITY_CUTS_PER_STEP = 8
# The Qr posterior is smoothed by a Gaussian of this width (mag) before its multimodality is measured, on a
# regular grid of this step. On the giant branch the tracks of neighbouring [Fe/H] lie 0.04-0.5 mag apart and
# one tLoc step moves Mr by up to 0.19 mag, so the posterior of a giant is a comb of about 150 spikes. Each
# gap between teeth counts as a valley, and without smoothing a correct giant gets a median multimodality of
# 0.16. Smoothing removes the comb, and also lets a broad posterior spread over both branches show its second
# bump. On the TRILEGAL wrong-branch test (|Mr_true - Mr| > 4 sigma, 1,971 of 261,844 stars) the AUC is 0.84
# raw, 0.945 at 0.1 mag, 0.953 at 0.2, 0.950 at 0.3 and 0.929 at 0.5, against 0.952 for Mr_std.
MULTIMODALITY_SMOOTHING = 0.2
MULTIMODALITY_STEP = 0.01


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


def binWidths(x):
    """Width of the bin each sorted value of x stands for, from the midpoint before it to the one after."""
    half = 0.5 * jnp.diff(x)
    return jnp.concatenate([half[:1], half]) + jnp.concatenate([half, half[-1:]])


def smoothingGrid(x, sigma, step=MULTIMODALITY_STEP):
    """What smoothPdfs needs to smooth distributions sampled at the sorted values x by a Gaussian of sigma.

    The weight of each value of x is shared between the two points of a regular grid of the given step on
    either side of it, in proportion to how near it is to each, and the grid is then convolved with the
    Gaussian sampled at the same step out to four widths and normalized. The grid reaches four widths past
    either end of x, so no weight is lost. With sigma = 0 nothing is smoothed and the grid is x itself.

    Returns the grid, the index in it of the point at or below each value of x, the share of its weight that
    goes to the point above, and the kernel.
    """
    x = np.asarray(x, dtype=float)
    if not sigma > 0:
        return x, np.arange(x.size), np.zeros(x.size), np.ones(1)
    half = int(np.ceil(4 * sigma / step))
    start = x[0] - half * step
    nBins = int(np.ceil((x[-1] - start) / step)) + half + 2
    grid = start + step * np.arange(nBins)
    position = (x - start) / step
    lower = np.clip(np.floor(position).astype(int), 0, nBins - 2)
    kernel = np.exp(-0.5 * (step * np.arange(-half, half + 1) / sigma) ** 2)
    return grid, lower, position - lower, kernel / kernel.sum()


def smoothPdfs(pdfs, lower, upper, kernel, nBins):
    """Rows of weights smoothed onto the grid of smoothingGrid, given its index, share and kernel.

    Weights rather than densities go in and come out: the grid is regular, so the two are the same shape.
    """
    pdfs = jnp.asarray(pdfs)
    grid = jnp.zeros((pdfs.shape[0], nBins), dtype=pdfs.dtype)
    grid = grid.at[:, lower].add(pdfs * (1 - upper)).at[:, lower + 1].add(pdfs * upper, mode="drop")
    convolve = partial(jnp.convolve, mode="same", precision=jax.lax.Precision.HIGHEST)
    return jax.vmap(lambda row: convolve(row, kernel.astype(pdfs.dtype)))(grid)


def multimodality(x, pdfs, nLevels=MULTIMODALITY_LEVELS):
    """How much of each distribution, a row of pdfs sampled at the sorted values x, lies in a second peak.

    Cut the density at a level and every run of bins above the cut is one peak, whose excess is the
    probability it holds above the cut. The second largest excess, at the cut where it is largest, is the
    excess mass of a second mode (Müller & Sawitzki 1991), which is twice Hartigan's dip statistic. It is zero
    for any distribution with a single peak, however skewed, and for two peaks it is a lower bound on the
    weight of the smaller one, which it approaches as they move apart. The cuts are nLevels - 2 levels evenly
    spaced between zero and the highest density. The density is the weight of a bin over its width, so x need
    not be evenly spaced. A distribution with no weight has no peaks to compare and gives NaN.

    Only the cuts at which a peak splits in two are evaluated. Between two of them every peak only shrinks as
    the cut rises, so the second largest excess can only fall, and its largest value is at one of them. A peak
    splits at the first cut above the floor of a valley: a stretch of bins that are above the same number of
    cuts, with bins above more of them on either side. That is 27 of the 298 cuts for a DP2 posterior on
    average and 164 at most, so the cuts of the whole batch go into one list of (distribution, cut) pairs,
    rather than every distribution waiting on the most ragged one.
    """
    width = binWidths(x)
    total = jnp.sum(pdfs, axis=1)
    density = pdfs / jnp.where(total > 0, total, 1)[:, None] / width
    levels = jnp.linspace(0, jnp.max(density, axis=1), nLevels, axis=1)[:, 1:-1]
    nRows, nBins = density.shape
    nCuts = levels.shape[1]
    position = jnp.arange(nBins)

    # how many cuts each bin is above, and for each stretch of bins at the same height the heights either side
    height = jax.vmap(partial(jnp.searchsorted, side="right"))(levels, density)
    step = height[:, 1:] != height[:, :-1]
    edge = jnp.ones((nRows, 1), bool)
    first = jax.lax.cummax(jnp.where(jnp.concatenate([edge, step], 1), position, 0), axis=1)
    last = jax.lax.cummin(
        jnp.where(jnp.concatenate([step, edge], 1), position, nBins - 1), axis=1, reverse=True
    )
    before = jnp.where(first > 0, jnp.take_along_axis(height, jnp.maximum(first - 1, 0), 1), -1)
    after = jnp.where(last < nBins - 1, jnp.take_along_axis(height, jnp.minimum(last + 1, nBins - 1), 1), -1)
    valley = (before > height) & (after > height)
    # a valley floor above h cuts splits its peak at cut h (counting from zero), and the lowest cut is where
    # the search starts
    splits = jnp.zeros((nRows, nCuts), bool)
    splits = splits.at[jnp.arange(nRows)[:, None], jnp.where(valley, height, nCuts)].set(True, mode="drop")
    splits = splits.at[:, 0].set(True)

    nPairs = jnp.sum(splits)
    chunk = MULTIMODALITY_CUTS_PER_STEP * nRows
    row, cut = jnp.nonzero(splits, size=splits.size + chunk, fill_value=0)

    def evaluate(state):
        i, best = state
        r = jax.lax.dynamic_slice(row, (i * chunk,), (chunk,))
        c = jax.lax.dynamic_slice(cut, (i * chunk,), (chunk,))
        second = _secondExcess(density[r], levels[r, c], width, position)
        second = jnp.where(i * chunk + jnp.arange(chunk) < nPairs, second, 0)
        return i + 1, best.at[r].max(second)

    best = jnp.zeros(nRows, density.dtype)
    _, best = jax.lax.while_loop(lambda state: state[0] * chunk < nPairs, evaluate, (0, best))
    return jnp.where(total > 0, best, jnp.nan)


def _secondExcess(density, cuts, width, position):
    """The excess of the second largest peak of each row of density, cut at the level given for that row."""
    above = density >= cuts[:, None]
    excess = jnp.where(above, (density - cuts[:, None]) * width, 0)
    # summed along each run of bins, starting again from zero at every gap, so the total of a run is the value
    # at its last bin
    running = jax.lax.associative_scan(_restartingSum, (~above, excess), axis=1)[1]
    ends = above & ((position == density.shape[1] - 1) | ~jnp.roll(above, -1, axis=1))
    runs = jnp.where(ends, running, 0)
    return jnp.max(jnp.where(position == jnp.argmax(runs, axis=1)[:, None], 0, runs), axis=1)


def _restartingSum(a, b):
    """Two stretches of a sum joined, where a set flag means the sum starts again from zero at that bin."""
    return a[0] | b[0], jnp.where(b[0], b[1], a[1] + b[1])
