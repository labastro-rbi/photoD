from functools import partial
from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import nested_pandas as npd
import numpy as np
import pandas as pd
from lsdb.catalog.map_catalog import MapCatalog
from lsdb.core.search.region_search import MOCSearch
from mocpy import MOC

from photod.column_map.base import mapper_from_glossary
from photod.parameters import GlobalParams
from photod.priors import getPriorMapIndex, initializePriorGrid
from photod.results import BayesResults
from photod.stats import entropies, getMargDistr3D, getPosteriorQuantiles, pnorm

cc = None


def set_column_mapping(variable_mapping: Path):
    """Set the module-global column map 'cc' from a glossary file."""
    global cc
    cc = mapper_from_glossary("CatalogColumnMap", "Reference catalog columns", variable_mapping)


set_column_mapping(Path(__file__).parent / "column_map" / "variables.yaml")

# Lengths of the A_r grid used when stars are grouped by the upper limit of their A_r prior. Each length is
# compiled once per process; grid values above a star's limit have zero prior and are left out.
AR_GRID_LENGTHS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
QUANTILE_NAMES = ("lo", "median", "hi")
# A batch holds a locus by A_r plane for each of its stars, so its memory is the batch size times the number
# of locus points times the length of the star's A_r grid. Bounding the bytes rather than that plane lets one
# batch size serve a locus of any size and a field of any extinction: bounding the plane alone leaves the
# locus to decide how much memory a run needs, and the full locus of DP2 on a deep A_r grid then asks for
# tens of gigabytes per batch.
# The budget is for one batch of one process, and a run has as many processes as it has workers, so it is
# set well below the memory of a machine rather than near it.
BATCH_BYTES = 2 << 30
BATCH_BYTES_PER_CELL = 12  # the posterior in float32, and the temporaries it is built from
# Bits of the flags column. A star with none of them set is one the model describes, on a single branch, with
# every colour measured and nothing of it against the edge of the grid.
FLAG_POOR_FIT = 1  # chi2 above CHI2_POOR, or no answer at all: the locus does not describe this star
FLAG_TWO_BRANCHES = 2  # the Mr posterior is lopsided, which is how a giant and a dwarf solution both survive
FLAG_FEH_EDGE = 4  # [Fe/H] is against the end of the model grid, so it is a limit rather than a measurement
FLAG_AR_EDGE = 8  # A_r is against the top of its grid, and the distance goes wrong with it
FLAG_COLOR_MISSING = 16  # at least one colour had no measurement and carried no weight in the fit
FLAG_NO_MAGNITUDE = 32  # no r magnitude, so neither a prior map nor a distance: the row carries no estimate
FLAG_NO_PRIOR = 64  # no prior map for this part of the sky, so the fit was never run: no estimate either
CHI2_POOR = 100.0
ASYMMETRY_POOR = 3.0
MISSING_COLOR_ERR = 1.0  # a colour whose error is above this is effectively unmeasured: the fit weighs it
# by 1 / err^2 like any other, which against a well measured colour is a ten-thousandth of the weight or less


def makeBayesEstimates3d(
    starsData: npd.NestedFrame,
    priorGrid: np.ndarray,
    globalParams: GlobalParams,
    batchSize: int = 100,
    returnPosteriors: bool = False,
    batchBytes: int = None,
):
    """Posterior statistics for all stars of a catalog partition.

    Parameters
    ----------
    starsData : DataFrame
        Fitted colors and their errors (<color>Err), observed r magnitude, object id and coordinates, and the
        dust-map A_r column when globalParams.ArMapColumn is set.
    priorGrid : array of shape (number of r bins, number of locus points)
        Prior maps interpolated onto the locus, from priors.initializePriorGrid.
    globalParams : GlobalParams
        Locus, color model, A_r grid and A_r prior settings.
    batchBytes : int, optional
        Memory one batch may take, BATCH_BYTES by default. It is the budget of one process, so a run of
        several workers needs that many times more.
    batchSize : int
        Number of stars computed together, as an upper bound: a batch is also kept within batchBytes, which
        for a large locus or a deep A_r grid is the limit that applies. 50-100 is a good choice on a CPU, a
        few hundred to a few thousand on a GPU.
    returnPosteriors : bool
        Also return the prior, likelihood and posterior cubes and their marginal distributions. Meant for
        inspecting a few stars.

    Returns
    -------
    estimatesDf : DataFrame
        chi2min, the 14th/50th/86th percentiles of Mr, [Fe/H], A_r and Qr (and of the true Mr when
        globalParams.computeMrTrue), and the entropy drop from prior to posterior for Mr, [Fe/H] and A_r.
    results : BayesResults
    """
    colors, colorsErr, priorIndices, arMax, rmag, curveIndex, arMap = getColorsAndPriorIndices(
        starsData, globalParams
    )
    nStars = colors.shape[0]
    if nStars == 0:
        meta = getEstimatesMeta(globalParams.computeMrTrue).reset_index(drop=True)
        return meta, BayesResults(meta[cc.chi_sq_min].to_numpy(), {})

    logPriorGrid, priorEntropy = _priorTables(priorGrid, globalParams)
    if returnPosteriors:
        gridLength = np.full(nStars, globalParams.Ar1d.size)
    else:
        gridLength = _arGridLengths(arMax, globalParams.Ar1d)

    # fixed-size batches, the last one padded with copies of a star, so that nothing is recompiled from one
    # partition to the next. The size must not depend on how many stars the partition holds: a survey has
    # partitions of every size, and letting each choose its own compiles a kernel for each of them, which
    # costs far more than padding a small partition up to a full batch ever saves.
    batchSize = int(max(1, batchSize))
    batches = []
    for nAr in np.unique(gridLength):
        stars = np.where(gridLength == nAr)[0]
        size = max(1, min(batchSize, _batchLimit(globalParams, int(nAr), batchBytes)))
        if returnPosteriors:
            # the cubes are as large as the batch, so a handful of stars must not pay for a full one
            size = min(size, stars.size)
        args = jax.device_put(globalParams.starArgs(int(nAr)))
        stars = np.concatenate([stars, np.full(-stars.size % size, stars[0])])
        for b in np.split(stars, stars.size // size):
            data = (
                colors[b],
                colorsErr[b],
                priorIndices[b],
                arMax[b],
                rmag[b],
                curveIndex[b],
                arMap[b],
            )
            out = _starBatch(
                data, logPriorGrid, priorEntropy, args, globalParams.computeMrTrue, returnPosteriors
            )
            # off the device as soon as it is done: holding every batch of a partition on the GPU until the
            # end keeps thousands of buffers alive, which exhausts it on a large partition
            batches.append((b, _toHost(out)))

    chi2min = _collect([(b, out[0]) for b, out in batches], nStars)
    statistics = {
        name: _collect([(b, out[1][name]) for b, out in batches], nStars) for name in batches[0][1][1]
    }
    # the distance modulus is r - (Mr + A_r), and Qr is the posterior of Mr + A_r, so its quantiles carry
    # over directly, the low end of one being the high end of the other
    for name, source in (("lo", "hi"), ("median", "median"), ("hi", "lo")):
        statistics[f"{cc.distance_modulus}_quantile_{name}"] = (
            rmag - statistics[f"{cc.abs_mag_ext_r}_quantile_{source}"]
        )
    # a star with no r magnitude has neither a prior map nor a distance modulus, and an answer read off the
    # brightest map would look like any other: it keeps its row, empty, and the flag says why
    noMagnitude = ~np.isfinite(rmag)
    if noMagnitude.any():
        chi2min[noMagnitude] = np.nan
        for value in statistics.values():
            value[noMagnitude] = np.nan
    statistics[cc.quality_flags] = _qualityFlags(
        chi2min, statistics, colorsErr, arMax, noMagnitude, globalParams
    )

    results = BayesResults(chi2min, statistics)
    if returnPosteriors:
        extra = {
            name: _collect([(b, out[2][name]) for b, out in batches], nStars) for name in batches[0][1][2]
        }
        results.priorCube, results.likeCube, results.postCube = extra["prior"], extra["like"], extra["post"]
        results.margpostMr = {k: extra[f"margMr{k}"] for k in range(3)}
        results.margpostFeH = {k: extra[f"margFeH{k}"] for k in range(3)}
        results.margpostAr = {k: extra[f"margAr{k}"] for k in range(3)}

    estimatesDf = pd.DataFrame(
        {
            cc.object_id: starsData[cc.object_id],
            cc.right_ascension: starsData[cc.right_ascension],
            cc.declination: starsData[cc.declination],
            cc.observed_mag_r: rmag,
            cc.chi_sq_min: chi2min,
            **statistics,
        }
    )
    # in the order and the types the meta declares, so that every partition of a run, and a partition that
    # holds no stars at all, describe themselves the same way
    meta = getEstimatesMeta(globalParams.computeMrTrue)
    return estimatesDf[list(meta.columns)].astype(meta.dtypes.to_dict()), results


def unfittedEstimates(starsData, globalParams, flag):
    """Rows for stars the fit could never be run on, with the flag that says why.

    Only their identity, position and magnitude are known, and the estimates are empty. They are kept because
    a catalog whose star count no longer matches the selection it came from cannot be used to count stars,
    which is most of what these catalogs are for.
    """
    meta = getEstimatesMeta(globalParams.computeMrTrue)
    columns = {name: np.full(len(starsData), np.nan) for name in meta.columns}
    for name in (cc.object_id, cc.right_ascension, cc.declination, cc.observed_mag_r):
        columns[name] = starsData[name].to_numpy()
    # the flags that describe the input rather than the answer are reported for these rows too, so that
    # counting the stars of a run by what was measured of them does not depend on which rows were fitted
    errors = starsData[[color + "Err" for color in globalParams.fitColors]].to_numpy(dtype=np.float64)
    flags = np.full(len(starsData), flag | FLAG_POOR_FIT, dtype=np.int32)
    rmag = starsData[cc.observed_mag_r].to_numpy(dtype=np.float64)
    flags |= np.where((errors > MISSING_COLOR_ERR).any(axis=1), FLAG_COLOR_MISSING, 0)
    flags |= np.where(~np.isfinite(rmag), FLAG_NO_MAGNITUDE, 0)
    columns[cc.quality_flags] = flags
    return pd.DataFrame(columns)[list(meta.columns)].astype(meta.dtypes.to_dict())


def makeBayesPosteriors3d(starsData: npd.NestedFrame, mapCatalog: MapCatalog, globalParams: GlobalParams):
    """Prior, likelihood and posterior cubes for a few stars, each with the prior map of its own sky pixel."""
    maxMapOrder = mapCatalog.hc_structure.pixel_tree.get_max_depth()
    results = []
    for index in range(len(starsData)):
        star = starsData.iloc[[index]]
        ra = star[cc.right_ascension].to_numpy() * u.deg
        dec = star[cc.declination].to_numpy() * u.deg
        mapMoc = MOC.from_lonlat(ra, dec, max_norder=maxMapOrder)
        mapPartitionDf = mapCatalog.search(MOCSearch(mapMoc, fine=False)).compute()
        priorGrid = jnp.array(list(initializePriorGrid(mapPartitionDf, globalParams).values()))
        _, starResults = makeBayesEstimates3d(star, priorGrid, globalParams, returnPosteriors=True)
        results.append(starResults)
    return results


def getColorsAndPriorIndices(catalog, params):
    """Per star: colors and errors, prior map index, the A_r prior limit, r, the dust curve and A_r(map)."""
    colors = catalog[list(params.fitColors)].to_numpy(dtype=np.float64)
    colorsErr = catalog[[color + "Err" for color in params.fitColors]].to_numpy(dtype=np.float64)
    if params.colorErrFloor > 0:
        colorsErr = np.sqrt(colorsErr**2 + params.colorErrFloor**2)
    priorIndices = np.asarray(getPriorMapIndex(catalog[cc.observed_mag_r]), dtype=np.int32)
    if params.ArMapColumn is None:
        arMap = np.zeros(len(catalog))
        arMax = np.full(len(catalog), np.inf)
    else:
        arMap = catalog[params.ArMapColumn].to_numpy(dtype=np.float64)
        arMax = params.ArPriorScale * arMap + params.ArPriorOffset
    rmag = catalog[cc.observed_mag_r].to_numpy(dtype=np.float64)
    if params.ArCurveIndexColumn is None:
        curveIndex = np.zeros(len(catalog), dtype=np.int32)
    else:
        curveIndex = catalog[params.ArCurveIndexColumn].to_numpy(dtype=np.int32)
    return colors, colorsErr, priorIndices, arMax, rmag, curveIndex, arMap


def getEstimatesMeta(computeMrTrue: bool = False):
    """Empty frame with the columns and types of the estimates, as lsdb meta."""
    names = [cc.abs_mag_r, cc.metallicity, cc.extinction_r, cc.abs_mag_ext_r, cc.distance_modulus] + (
        ["Mr_true"] if computeMrTrue else []
    )
    quantileCols = [f"{name}_quantile_{q}" for name in names for q in QUANTILE_NAMES]
    entropyCols = [cc.abs_mag_r_entropy_drop, cc.metallicity_entropy_drop, cc.extinction_r_entropy_drop]
    colNames = [
        cc.object_id,
        cc.right_ascension,
        cc.declination,
        cc.observed_mag_r,
        cc.chi_sq_min,
        cc.quality_flags,
        *sorted(quantileCols + entropyCols),
    ]
    # the positions and the magnitude are carried over from the catalog rather than fitted, and float32
    # would round a position by a tenth of an arcsecond
    dtypes = {
        cc.object_id: np.int64,
        cc.quality_flags: np.int32,
        cc.right_ascension: np.float64,
        cc.declination: np.float64,
        cc.observed_mag_r: np.float64,
    }
    meta = npd.NestedFrame.from_dict(
        {col: pd.Series([], dtype=dtypes.get(col, np.float32)) for col in colNames}
    )
    meta.index.name = "_healpix_29"
    return meta


def starPosterior(star, logPriorGrid, priorEntropy, args, computeMrTrue=False, returnPosteriors=False):
    """Posterior statistics for one star.

    For locus point i and extinction A the model colors are m_i + A R, so
        chi2(i, A) = sum_c w_c (d_ic - A R_c)^2 = c_i + s (A - A0_i)^2,
    with d = colors - m_i, w = 1 / err^2, s = sum_c w_c R_c^2, A0_i = sum_c w_c d_ic R_c / s and
    c_i = chi2(i, A0_i). The posterior is scaled to 1 at its maximum over the allowed cells, which keeps it
    representable however large chi2 is.
    """
    colors, colorsErr, priorIndex, arMax, rmag, curveIndex, arMap = star
    Ar1d, ArFull, FeH1d, Mr1d = args["Ar1d"], args["ArFull"], args["FeH1d"], args["Mr1d"]
    nFeH, nMr, nAr = FeH1d.size, Mr1d.size, Ar1d.size

    w = 1.0 / colorsErr**2
    R = args["reddVector"]
    d = colors - args["locusColors2d"]
    s = jnp.sum(w * R**2)
    A0 = jnp.where(s > 0, _dot(d, w * R) / jnp.where(s > 0, s, 1.0), 0.0)
    c = _dot(jnp.square(d - A0[:, None] * R), w)
    chi2min = jnp.min(_chi2GridMin(c, s, A0, ArFull, ArFull.size - 1))

    # flat A_r prior between 0 and arMax (no limit for a single fixed A_r)
    allowed = (Ar1d <= arMax) | (ArFull.size == 1)
    logPrior = logPriorGrid[priorIndex]
    if args["ArCurves"] is None:
        Astar, wDust = jnp.zeros_like(A0), jnp.zeros_like(A0)
    else:
        # A 3D dust map adds a second quadratic in A_r: locus point i puts the star at mu = r - MrTrue_i - A,
        # where the map has A*_i, so ln prior = -w_i (A - A*_i)^2 / 2. The chi2 is quadratic in A_r as well,
        # so the two combine into one quadratic and the fit stays a single pass over the (locus, A_r) grid.
        curve = args["ArCurves"][curveIndex] * arMap
        Astar = jnp.interp(args["MrTrueFlat"], (rmag - args["ArCurveMu"] - curve)[::-1], curve[::-1])
        wDust = jnp.where(
            curve[-1] > 0, 1.0 / ((args["ArCurveFrac"] * Astar) ** 2 + args["ArCurveFloor"] ** 2), 0.0
        )
    S = s + wDust
    Acomb = (s * A0 + wDust * Astar) / S
    extra = s * wDust / S * (A0 - Astar) ** 2
    # That Gaussian is a density in A_r whose width follows A*_i, so it has to be normalised along with it.
    # Dropping the normalisation weighs every locus point as if its A_r were equally well known, which
    # favours the points the map reddens the most, and those are the distant, luminous solutions.
    logDust = 0.5 * jnp.log(jnp.where(wDust > 0, wDust, 1.0))
    # the peak is taken over the A_r grid, not over the real line: without that, a star whose best A_r falls
    # outside the grid underflows everywhere and its quantiles come out as NaN
    logPeak = jnp.max(
        logPrior + logDust - 0.5 * _chi2GridMin(c + extra, S, Acomb, Ar1d, jnp.sum(allowed) - 1)
    )
    u = logPrior + logDust - 0.5 * (c + extra) - logPeak
    post = jnp.exp(u[:, None] - 0.5 * S[:, None] * (Ar1d - Acomb[:, None]) ** 2) * allowed

    # sums over the short trailing axes as products with vectors of ones: several times faster on CPUs
    ones = partial(jnp.ones, dtype=post.dtype)
    postFeHMr = _dot(post, ones(nAr)).reshape(nFeH, nMr)
    post = post.reshape(nFeH, nMr, nAr)
    postMrAr = jnp.sum(post, axis=0)
    margMr = pnorm(jnp.sum(postFeHMr, axis=0), args["dMr"])
    margFeH = pnorm(_dot(postFeHMr, ones(nMr)), args["dFeH"])
    margAr = pnorm(jnp.sum(postMrAr, axis=0), args["dAr"])
    QrWeights = (
        jnp.zeros_like(args["QrGrid"], dtype=post.dtype)
        .at[args["QrIdxIndep"]]
        .add(postMrAr[args["QrColsIndep"]])
        .at[args["QrIdxDep"]]
        .add(post[:, args["QrColsDep"], :])
    )

    statistics = {}
    pdfs = [
        (Mr1d, margMr, cc.abs_mag_r),
        (FeH1d, margFeH, cc.metallicity),
        (Ar1d, margAr, cc.extinction_r),
        (args["QrGrid"], QrWeights, cc.abs_mag_ext_r),
    ]
    if computeMrTrue:
        MrTrueWeights = (
            jnp.zeros_like(args["MrTrueGrid"], dtype=post.dtype).at[args["MrTrueIndices"]].add(postFeHMr)
        )
        pdfs.append((args["MrTrueGrid"], MrTrueWeights, "Mr_true"))
    for values, pdf, name in pdfs:
        for q, value in zip(QUANTILE_NAMES, getPosteriorQuantiles(values, pdf), strict=True):
            statistics[f"{name}_quantile_{q}"] = value
    # the entropies are of sampled densities, so each carries the width of its own bin and comes out in bits
    HMr, HFeH, HAr, HAr0 = entropies([margMr, margFeH, margAr, pnorm(allowed * 1.0, args["dAr"])])
    statistics[cc.abs_mag_r_entropy_drop] = HMr * args["dMr"] - priorEntropy[priorIndex, 0]
    statistics[cc.metallicity_entropy_drop] = HFeH * args["dFeH"] - priorEntropy[priorIndex, 1]
    statistics[cc.extinction_r_entropy_drop] = (HAr - HAr0) * args["dAr"]

    cubes = {}
    if returnPosteriors:
        prior = (
            jnp.exp(
                logPrior[:, None] + logDust[:, None] - 0.5 * wDust[:, None] * (Ar1d - Astar[:, None]) ** 2
            )
            * allowed
        ).reshape(nFeH, nMr, nAr)
        like = jnp.exp(-0.5 * (c[:, None] + s * (Ar1d - A0[:, None]) ** 2 - chi2min)).reshape(nFeH, nMr, nAr)
        cubes = {"prior": prior, "like": like, "post": post}
        for k, cube in enumerate((prior, like, post)):
            margs = getMargDistr3D(cube, args["dMr"], args["dFeH"], args["dAr"])
            cubes[f"margMr{k}"], cubes[f"margFeH{k}"], cubes[f"margAr{k}"] = margs
    return chi2min, statistics, cubes


@partial(jax.jit, static_argnames=("computeMrTrue", "returnPosteriors"))
def _starBatch(data, logPriorGrid, priorEntropy, args, computeMrTrue, returnPosteriors):
    star = partial(
        starPosterior,
        logPriorGrid=logPriorGrid,
        priorEntropy=priorEntropy,
        args=args,
        computeMrTrue=computeMrTrue,
        returnPosteriors=returnPosteriors,
    )
    return jax.vmap(star)(data)


def _dot(a, b):
    """Matrix product in full float precision (on recent GPUs the default for float32 is TF32)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _chi2GridMin(c, s, A0, grid, kLast):
    """Smallest c + s * (A - A0)^2 over grid[0..kLast] of a uniform grid, for every locus point.

    The function is convex in A, so the minimum is at one of the two grid values around A0.
    """
    if grid.size == 1:
        return c + s * (grid[0] - A0) ** 2
    k = jnp.floor((A0 - grid[0]) / (grid[1] - grid[0])).astype(jnp.int32)
    below = grid[jnp.clip(k, 0, kLast)]
    above = grid[jnp.clip(k + 1, 0, kLast)]
    return c + s * jnp.minimum((below - A0) ** 2, (above - A0) ** 2)


def _priorTables(priorGrid, params):
    """Log prior on the locus (plateau points excluded) and the entropy of its Mr and [Fe/H] marginals."""
    prior = jnp.asarray(priorGrid) * params.locusValid
    maps = prior.reshape(-1, params.FeH1d.size, params.Mr1d.size)
    columns = []
    for marg, step in ((maps.sum(axis=1), params.dMr), (maps.sum(axis=2), params.dFeH)):
        p = marg / marg.sum(axis=1, keepdims=True) / step
        p = jnp.where(p > 0, p, 1)
        columns.append(-jnp.sum(p * jnp.log2(p), axis=1) * step)
    return jnp.log(prior), jnp.stack(columns, axis=1)


def _arGridLengths(arMax, Ar1d):
    """Length of the A_r grid each star needs: all values allowed by its prior, the first value above the
    limit (so that the A_r percentiles are interpolated exactly as on the full grid) and one more for the
    float32 rounding of the limit."""
    need = np.minimum(np.searchsorted(Ar1d, arMax, side="right") + 2, Ar1d.size)
    lengths = np.array(sorted({n for n in AR_GRID_LENGTHS if n < Ar1d.size} | {Ar1d.size}))
    return lengths[np.searchsorted(lengths, need)]


def _qualityFlags(chi2min, statistics, colorsErr, arMax, noMagnitude, globalParams):
    """One bit per thing worth knowing about a star's fit, as described by the FLAG_ constants.

    Everything here is a property of the answer rather than of the star, and none of it removes a row: a
    catalog that quietly drops what it cannot fit is harder to use than one that says so.
    """
    flags = np.zeros(chi2min.size, dtype=np.int32)
    # the absolute magnitude itself when the locus is parametrised by tLoc, since that is where a giant and a
    # dwarf solution sit far apart
    name = "Mr_true" if "Mr_true_quantile_median" in statistics else cc.abs_mag_r
    low, median, high = (statistics[f"{name}_quantile_{q}"] for q in QUANTILE_NAMES)
    # a star the fit could not place at all must not come out looking like one it placed well
    flags |= np.where((chi2min > CHI2_POOR) | ~np.isfinite(median), FLAG_POOR_FIT, 0)

    upper, lower = high - median, median - low
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lower > 0, upper / lower, np.inf)
    lopsided = (ratio > ASYMMETRY_POOR) | (ratio < 1 / ASYMMETRY_POOR)
    flags |= np.where(lopsided & np.isfinite(median), FLAG_TWO_BRANCHES, 0)

    feH = statistics[f"{cc.metallicity}_quantile_median"]
    step = np.diff(globalParams.FeH1d)[0] if globalParams.FeH1d.size > 1 else 0.0
    flags |= np.where(
        (feH <= globalParams.FeH1d[0] + step) | (feH >= globalParams.FeH1d[-1] - step), FLAG_FEH_EDGE, 0
    )

    # A_r is bounded by the dust map as well as by the grid, and with a map it is the map that binds: a star
    # pinned against either has an extinction that is a limit, and a distance that goes wrong with it
    # A single A_r held fixed has no edge to be pinned against: every star sits on the one value there is
    if globalParams.Ar1d.size > 1:
        ar = statistics[f"{cc.extinction_r}_quantile_median"]
        limit = np.minimum(arMax, globalParams.Ar1d[-1]) - globalParams.dAr
        flags |= np.where(ar >= limit, FLAG_AR_EDGE, 0)

    flags |= np.where((colorsErr > MISSING_COLOR_ERR).any(axis=1), FLAG_COLOR_MISSING, 0)
    flags |= np.where(noMagnitude, FLAG_NO_MAGNITUDE, 0)
    return flags


def _batchLimit(globalParams, nAr, batchBytes=None):
    """How many stars of this A_r grid length fit a batch into the memory budget."""
    cells = BATCH_BYTES_PER_CELL * globalParams.FeH1d.size * globalParams.Mr1d.size * nAr
    return max(1, int((BATCH_BYTES if batchBytes is None else batchBytes) // cells))


def _toHost(out):
    """A batch result as plain arrays, releasing the device buffers it was computed into."""
    return tuple(
        (
            {name: np.asarray(value) for name, value in item.items()}
            if isinstance(item, dict)
            else np.asarray(item)
        )
        for item in out
    )


def _collect(parts, nStars):
    """Assemble per-batch results into arrays over all stars (padded entries repeat a real star)."""
    first = np.asarray(parts[0][1])
    out = np.empty((nStars,) + first.shape[1:], dtype=first.dtype)
    for b, values in parts:
        out[b] = np.asarray(values)
    return out
