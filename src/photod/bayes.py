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
from photod.quality import MIN_COLOURS, QC_CLASSES, claimedClass, colourPattern, latitudeClass, poorFit
from photod.results import BayesResults
from photod.selection import beyondRecipe, failsRecipe
from photod.stats import getMargDistr3D, getPosteriorQuantiles, getStats, multimodality, pnorm

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
# The posterior split into the classes of locus.CLASS_SEGMENTS: the mass on each, the mass on the helium
# white dwarfs, and the log odds of white dwarf against the rest, after the fit and in the prior alone. Every
# other estimate is of the posterior within one class of star, since one median over two classes that lie
# eight magnitudes apart describes neither. By default that is the most probable class of star. A QSO is not
# one of them: it has no distance, so the stellar estimates are of the most probable stellar class whatever
# p_QSO is, and where the QSO class is the most probable of all the distance modulus is left empty and
# FLAG_EXTRAGALACTIC set. The white dwarf distance is also given on its own wherever the class holds more than
# WD_DISTANCE_MIN_P of the posterior, and the QSO redshift (QSO_REDSHIFT, the median along the track) wherever
# that class does. With GlobalParams.estimateClass "MS" the stellar estimates are of the stars for every
# object, the distance modulus is written whatever class wins (FLAG_EXTRAGALACTIC still says where the QSO
# class does) and the QSO redshift is written for every object.
CLASS_COLUMNS = ("p_MS", "p_WD", "p_He", "p_QSO", "lnodds_WD", "lnprior_odds_WD")
# the most probable of p_MS, p_WD and p_QSO, as its index in quality.QC_CLASSES (0, 1, 2); -1 with no answer
CLASS_COLUMN = "class"
# chi2min within each class: the smallest chi2 over the class's locus points that have prior weight, with A_r
# inside each point's own allowed range (the box, or the band of GlobalParams.ArCurveMode). It is what a star
# given that class is tested on (photod.quality); chi2min itself is over every point and the whole A_r grid.
CHI2_CLASS_COLUMNS = tuple(f"chi2min{name}" for name in QC_CLASSES)
STELLAR_CLASSES = ("MS", "WD")
WD_DISTANCE_MIN_P = 0.05
WD_DISTANCE = "DM_WD"
QSO_REDSHIFT = "photoz"
# On a locus whose white dwarf rows are a mass axis (GlobalParams.wdMasses), the median of the white dwarf
# posterior along it, as a mass in Msun, where the class is as likely as for WD_DISTANCE. Photometry barely
# constrains it: on DP2 it follows the prior on mass and correlates at 0.15-0.20 with spectroscopic masses.
WD_MASS = "WD_mass"
# A batch holds a locus by A_r plane for each of its stars, so its memory is the batch size times the number
# of locus points times the length of the star's A_r grid. Bounding the bytes rather than that plane lets one
# batch size serve a locus of any size and a field of any extinction: bounding the plane alone leaves the
# locus to decide how much memory a run needs, and the full locus of DP2 on a deep A_r grid then asks for
# tens of gigabytes per batch.
# The budget is for one batch of one process, and a run has as many processes as it has workers. It is a
# bound worked out at BATCH_BYTES_PER_CELL, and the fit holds far less than that: at 16 GB a DP2 process
# peaked at 6.2 GB of GPU memory, and two of them shared an 80 GB H100 at 18 GB. Large batches are what make
# the fit fast, the stars on a deep A_r grid most of all, so a GPU with less memory, or one shared with other
# work, is the case for giving a run less rather than the default.
BATCH_BYTES = 16 << 30
BATCH_BYTES_PER_CELL = 12  # the posterior in float32, and the temporaries it is built from
# Bits of the fit_flags column. A star with none of them set is one the model describes, on a single branch,
# with every colour measured and nothing of it against the edge of the grid.
FLAG_POOR_FIT = 1  # chi2 above CHI2_POOR, or no answer at all: the locus does not describe this star
FLAG_TWO_BRANCHES = 2  # the Mr posterior is lopsided, which is how a giant and a dwarf solution both survive
FLAG_FEH_EDGE = 4  # [Fe/H] is against the end of the model grid, so it is a limit rather than a measurement
# (on a white dwarf mass axis, for a star of that class, a mass within a row of 0.40 or 1.30 Msun)
FLAG_AR_EDGE = 8  # A_r is against the top of its grid, and the distance goes wrong with it
FLAG_COLOR_MISSING = 16  # at least one colour had no measurement and carried no weight in the fit
FLAG_NO_MAGNITUDE = 32  # no r magnitude, so neither a prior map nor a distance: the row carries no estimate
FLAG_NO_PRIOR = 64  # no prior map for this part of the sky, so the fit was never run: no estimate either
FLAG_COLLAPSED = 128  # the posterior fell into a single grid cell: precise-looking and not to be trusted
FLAG_BEYOND_RECIPE = 256  # past the r limit of the selection recipe (photod.selection), reported apart
FLAG_EXTRAGALACTIC = 512  # the QSO class is the most probable: the distance modulus is empty, unless the
# estimates are of the stars on every row (GlobalParams.estimateClass "MS")
FLAG_POOR_FIT_P = 1024  # chi2min of the class the star is given is above the model's own 99th percentile for
# its colours and |b| (photod.quality; set only when the fit has a QC table)
FLAG_TOO_FEW_COLOURS = 2048  # fewer than quality.MIN_COLOURS colours measured: no white dwarf call
FLAG_FAILS_RECIPE = 4096  # fails the S/N cuts of the selection recipe (photod.selection), fitted all the same
# because the run kept every row (run_dp2 --selection flag): its class probabilities and calls are computed
# like any other row's, and the bit is how they are left out
CHI2_POOR = 100.0
ASYMMETRY_POOR = 3.0
COLLAPSED_WIDTH = 0.1  # half-width of the Qr interval below which an answer is collapsed, not precise
MISSING_COLOR_ERR = 1.0  # a colour whose error is above this, or is not a number, is not measured: the fit
# gives it no weight at all, whatever value the catalogue holds for it, and nColors does not count it


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
        The columns and types of getEstimatesMeta: object id, position and r magnitude, chi2min, the flags,
        the number of measured colours, the multimodality, the class columns (CLASS_COLUMNS), and the
        16th/50th/84th percentiles and standard deviation of the locus coordinate, [Fe/H], A_r and the
        distance modulus (and of the true Mr when globalParams.computeMrTrue), all of these within the most
        probable class of star, the white dwarf distance modulus (WD_DISTANCE) and mass (WD_MASS) where that
        class is likely enough, and the QSO redshift (QSO_REDSHIFT) where the QSO class is.
    results : BayesResults
        chi2min and every statistic computed, which also holds Qr, written to no column.
    """
    colors, colorsErr, priorIndices, arMax, rmag, curveIndex, arMap = getColorsAndPriorIndices(
        starsData, globalParams
    )
    nStars = colors.shape[0]
    if nStars == 0:
        meta = getEstimatesMeta(globalParams.computeMrTrue).reset_index(drop=True)
        return meta, BayesResults(meta[cc.chi_sq_min].to_numpy(), {})

    logPriorGrid = _logPrior(priorGrid, globalParams)
    lnPriorOdds = _priorClassOdds(priorGrid, globalParams)
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
            out = _starBatch(data, logPriorGrid, args, globalParams.computeMrTrue, returnPosteriors)
            # off the device as soon as it is done: holding every batch of a partition on the GPU until the
            # end keeps thousands of buffers alive, which exhausts it on a large partition
            batches.append((b, _toHost(out)))

    chi2min = _collect([(b, out[0]) for b, out in batches], nStars)
    statistics = {
        name: _collect([(b, out[1][name]) for b, out in batches], nStars) for name in batches[0][1][1]
    }
    # the distance modulus is r - (Mr + A_r), and Qr is the posterior of Mr + A_r, so its quantiles carry
    # over directly, the low end of one being the high end of the other
    # r is measured, not known, and DM = r - Qr, so the distance modulus is less certain than Qr by that
    # error. It is symmetric, so the median does not move and each end moves out in quadrature. On DP2 this
    # is worth a few thousandths of a magnitude for a bright star and about 5 % of the interval at r = 23.
    rmagErr = magnitudeError(starsData)
    for dm, qr in ((cc.distance_modulus, cc.abs_mag_ext_r), (WD_DISTANCE, f"{cc.abs_mag_ext_r}_WD")):
        for name, source in (("lo", "hi"), ("median", "median"), ("hi", "lo")):
            statistics[f"{dm}_quantile_{name}"] = rmag - statistics[f"{qr}_quantile_{source}"]
        if np.any(rmagErr > 0):
            median = statistics[f"{dm}_quantile_median"]
            for name, direction in (("lo", -1.0), ("hi", 1.0)):
                half = np.abs(statistics[f"{dm}_quantile_{name}"] - median)
                statistics[f"{dm}_quantile_{name}"] = median + direction * np.hypot(half, rmagErr)
    statistics[f"{cc.distance_modulus}_std"] = np.hypot(statistics[f"{cc.abs_mag_ext_r}_std"], rmagErr)
    for name in QUANTILE_NAMES:
        del statistics[f"{cc.abs_mag_ext_r}_WD_quantile_{name}"]
    likelyWD = statistics["p_WD"] > WD_DISTANCE_MIN_P
    for name in QUANTILE_NAMES:
        statistics[f"{WD_DISTANCE}_quantile_{name}"] = np.where(
            likelyWD, statistics[f"{WD_DISTANCE}_quantile_{name}"], np.nan
        )
    feHWD = statistics.pop(f"{cc.metallicity}_WD_quantile_median")
    statistics[WD_MASS] = (
        np.where(likelyWD, np.interp(feHWD, globalParams.FeH1d, globalParams.wdMasses), np.nan)
        if globalParams.wdMasses is not None
        else np.full(nStars, np.nan)
    )
    extragalactic = statistics["p_QSO"] > np.maximum(statistics["p_MS"], statistics["p_WD"])
    if globalParams.estimateClass is None:
        statistics[QSO_REDSHIFT] = np.where(
            statistics["p_QSO"] > WD_DISTANCE_MIN_P, statistics[QSO_REDSHIFT], np.nan
        )
        modulus = cc.distance_modulus
        for name in [f"{modulus}_quantile_{q}" for q in QUANTILE_NAMES] + [f"{modulus}_std"]:
            statistics[name] = np.where(extragalactic, np.nan, statistics[name])
    statistics["lnprior_odds_WD"] = lnPriorOdds[priorIndices].astype(np.float32)
    # a class with no point of prior weight has no chi2 to give, and is written as empty rather than infinite
    for name in CHI2_CLASS_COLUMNS:
        statistics[name] = np.where(np.isfinite(statistics[name]), statistics[name], np.nan)
    # a star with no r magnitude has neither a prior map nor a distance modulus, and an answer read off the
    # brightest map would look like any other: it keeps its row, empty, and the flag says why
    noMagnitude = ~np.isfinite(rmag)
    if noMagnitude.any():
        chi2min[noMagnitude] = np.nan
        for value in statistics.values():
            value[noMagnitude] = np.nan
    nColors = np.sum(np.isfinite(colorsErr), axis=1).astype(np.int32)
    flags = _qualityFlags(chi2min, statistics, colorsErr, arMax, noMagnitude, globalParams, rmag)
    flags |= np.where(extragalactic, FLAG_EXTRAGALACTIC, 0).astype(np.int32)
    flags |= np.where(nColors < MIN_COLOURS, FLAG_TOO_FEW_COLOURS, 0).astype(np.int32)
    flags |= np.where(failsRecipe(starsData), FLAG_FAILS_RECIPE, 0).astype(np.int32)
    if globalParams.qcThresholds is not None:
        latitude = latitudeClass(starsData[cc.right_ascension], starsData[cc.declination])
        poor = poorFit(globalParams.qcThresholds, statistics, colourPattern(colorsErr), latitude)
        flags |= np.where(poor, FLAG_POOR_FIT_P, 0).astype(np.int32)
    statistics[cc.quality_flags] = flags
    statistics["nColors"] = nColors
    statistics[CLASS_COLUMN] = claimedClass(statistics).astype(np.int8)

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
    colors = starsData[list(globalParams.fitColors)].to_numpy(dtype=np.float64)
    errors = starsData[[color + "Err" for color in globalParams.fitColors]].to_numpy(dtype=np.float64)
    flags = np.full(len(starsData), flag | FLAG_POOR_FIT, dtype=np.int32)
    rmag = starsData[cc.observed_mag_r].to_numpy(dtype=np.float64)
    # measured as getColorsAndPriorIndices has it: an error of at most MISSING_COLOR_ERR and a value
    measured = (errors <= MISSING_COLOR_ERR) & np.isfinite(colors)
    flags |= np.where((~measured).any(axis=1), FLAG_COLOR_MISSING, 0)
    flags |= np.where(~np.isfinite(rmag), FLAG_NO_MAGNITUDE, 0)
    flags |= np.where(beyondRecipe(rmag), FLAG_BEYOND_RECIPE, 0)
    flags |= np.where(measured.sum(axis=1) < MIN_COLOURS, FLAG_TOO_FEW_COLOURS, 0)
    flags |= np.where(failsRecipe(starsData), FLAG_FAILS_RECIPE, 0)
    columns[cc.quality_flags] = flags
    columns["nColors"] = measured.sum(axis=1)
    columns[CLASS_COLUMN] = np.full(len(starsData), -1)
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
    # A colour that was not measured carries no weight at all: an infinite error, whose weight 1 / err^2 is
    # exactly zero, and a value of zero, which it then multiplies. Catalogues write such a colour as 0 with an
    # error of 9.99, and 0 is a white dwarf's g-r, so the small weight a finite placeholder keeps pulls the
    # faintest objects, the ones with most colours missing, towards the white dwarf track. A colour or an
    # error that is not a number is not measured either; passed on as it is, it would make every estimate of
    # the star NaN.
    missing = ~(colorsErr <= MISSING_COLOR_ERR) | ~np.isfinite(colors)
    colors = np.where(missing, 0.0, colors)
    colorsErr = np.where(missing, np.inf, colorsErr)
    if params.colorErrFloor > 0:
        colorsErr = np.sqrt(colorsErr**2 + params.colorErrFloor**2)
    priorIndices = np.asarray(getPriorMapIndex(catalog[cc.observed_mag_r]), dtype=np.int32)
    if params.ArMapColumn is None:
        arMap = np.zeros(len(catalog))
        arMax = np.full(len(catalog), np.inf)
    else:
        # a star the dust map has no value for is not bounded by it, as in a run without a map: a limit of
        # NaN leaves no A_r allowed at all, and the posterior empty
        arMap = catalog[params.ArMapColumn].to_numpy(dtype=np.float64)
        mapped = np.isfinite(arMap)
        arMax = np.where(mapped, params.ArPriorScale * arMap + params.ArPriorOffset, np.inf)
        arMap = np.where(mapped, arMap, 0.0)
    rmag = catalog[cc.observed_mag_r].to_numpy(dtype=np.float64)
    if params.ArCurveIndexColumn is None:
        curveIndex = np.zeros(len(catalog), dtype=np.int32)
    else:
        curveIndex = catalog[params.ArCurveIndexColumn].to_numpy(dtype=np.int32)
    return colors, colorsErr, priorIndices, arMax, rmag, curveIndex, arMap


def magnitudeError(catalog):
    """The error on r, where the catalog carries one: the distance modulus is r - Qr, so it belongs there.

    A catalog without the column is answered with zeros, which is what the fit did before it was read: the
    distance modulus then carries only the width of the Qr posterior, as if r were exact.
    """
    name = f"{cc.observed_mag_r}Err"
    if name not in getattr(catalog, "columns", ()):
        return np.zeros(len(catalog))
    error = catalog[name].to_numpy(dtype=np.float64)
    return np.where(np.isfinite(error) & (error > 0), error, 0.0)


def getEstimatesMeta(computeMrTrue: bool = False):
    """Empty frame with the columns and types of the estimates, as lsdb meta."""
    # Qr is computed but not written: it is Mr + A_r, which the distance modulus and r already give
    names = [cc.abs_mag_r, cc.metallicity, cc.extinction_r, cc.distance_modulus] + (
        ["Mr"] if computeMrTrue else []
    )
    quantileCols = [f"{name}_quantile_{q}" for name in names for q in QUANTILE_NAMES]
    stdCols = [f"{name}_std" for name in names]
    colNames = [
        cc.object_id,
        cc.right_ascension,
        cc.declination,
        cc.observed_mag_r,
        cc.chi_sq_min,
        *CHI2_CLASS_COLUMNS,
        cc.quality_flags,
        "nColors",
        cc.multimodality,
        *CLASS_COLUMNS,
        CLASS_COLUMN,
        *sorted(quantileCols + stdCols),
        *(f"{WD_DISTANCE}_quantile_{q}" for q in QUANTILE_NAMES),
        WD_MASS,
        QSO_REDSHIFT,
    ]
    # the positions and the magnitude are carried over from the catalog rather than fitted, and float32
    # would round a position by a tenth of an arcsecond
    dtypes = {
        cc.object_id: np.int64,
        cc.quality_flags: np.int32,
        "nColors": np.int32,
        CLASS_COLUMN: np.int8,
        cc.right_ascension: np.float64,
        cc.declination: np.float64,
        cc.observed_mag_r: np.float64,
    }
    meta = npd.NestedFrame.from_dict(
        {col: pd.Series([], dtype=dtypes.get(col, np.float32)) for col in colNames}
    )
    meta.index.name = "_healpix_29"
    return meta


def starPosterior(star, logPriorGrid, args, computeMrTrue=False, returnPosteriors=False):
    """Posterior statistics for one star, and the Qr posterior they are partly read from.

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
    kFirst, kLast, allowedPoint, logWidth = 0, jnp.sum(allowed) - 1, allowed[None, :], 0.0
    band = args["ArBandRows"] is not None
    if band:
        # each point's own stretch of the A_r grid inside the box, with a flat density over its cells
        # (GlobalParams.ArCurveMode), so that a point whose band is narrower is not given more prior for it.
        # A star the dust map has no value for (arMax infinite) has no column to bound a band by: its arMap
        # of 0 only stands in for the missing value, so every point keeps the box, the QSO points too.
        Alo, Ahi = arBandEdges(
            args["MrTrueFlat"],
            rmag,
            args["ArCurves"][curveIndex],
            args["ArCurveMu"],
            arMap,
            args["ArPriorScale"],
            args["ArPriorOffset"],
            args["ArBandRows"] & jnp.isfinite(arMax),
            args["extragalactic"],
        )
        dA = args["dAr"]
        kTop = jnp.floor((jnp.minimum(Ahi, 1e6) - Ar1d[0]) / dA + 1e-9).astype(jnp.int32)
        kLast = jnp.clip(jnp.minimum(kLast, kTop), 0, None)
        kFirst = jnp.clip(jnp.ceil((Alo - Ar1d[0]) / dA - 1e-9).astype(jnp.int32), 0, kLast)
        cells = jnp.arange(nAr)[None, :]
        allowedPoint = (cells >= kFirst[:, None]) & (cells <= kLast[:, None])
        logWidth = -jnp.log((kLast - kFirst + 1) * dA)
    if args["ArCurves"] is None or band:
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
    # S is 0 for a star with no colour measured and no dust curve: nothing then constrains A_r, and the
    # posterior is the prior
    Acomb = jnp.where(S > 0, (s * A0 + wDust * Astar) / jnp.where(S > 0, S, 1.0), 0.0)
    extra = s * wDust / jnp.where(S > 0, S, 1.0) * (A0 - Astar) ** 2
    # That Gaussian is a density in A_r whose width follows A*_i, so it has to be normalised along with it.
    # Dropping the normalisation weighs every locus point as if its A_r were equally well known, which
    # favours the points the map reddens the most, and those are the distant, luminous solutions.
    logDust = 0.5 * jnp.log(jnp.where(wDust > 0, wDust, 1.0))
    # the peak is taken over the A_r grid, not over the real line: without that, a star whose best A_r falls
    # outside the grid underflows everywhere and its quantiles come out as NaN
    logBase = logPrior + logDust + logWidth
    peaks = logBase - 0.5 * _chi2GridMin(c + extra, S, Acomb, Ar1d, kLast, kFirst)
    names = list(args["classAxisMasks"])
    perClass = args["estimateMask"] is not None
    if perClass:
        # each class scaled to its own peak (GlobalParams.estimateClass), and a class no point of which is
        # allowed to 1, where its posterior is zero anyway
        # (each point's class by its place in names, which inside jit are in the sorted order of a pytree)
        classAxis = jnp.argmax(jnp.stack([args["classAxisMasks"][name] for name in names]), axis=0)
        classFlat = jnp.broadcast_to(classAxis, (nFeH, nMr)).ravel()
        classPeak = jax.ops.segment_max(peaks, classFlat, num_segments=len(names))
        reached = jnp.isfinite(classPeak)
        # the peaks against the highest, for the class masses: a peak is near -chi2 / 2, and a float32 of 2e4
        # is good to 2e-3 only, which taken as it is would leave the probabilities summing to 1 within that
        classShift = jnp.where(reached, classPeak - jnp.max(jnp.where(reached, classPeak, -jnp.inf)), 0.0)
        classPeak = jnp.where(reached, classPeak, 0.0)
        logPeak = classPeak[classFlat]
    else:
        logPeak = jnp.max(peaks)
    u = logBase - 0.5 * (c + extra) - logPeak
    post = jnp.exp(u[:, None] - 0.5 * S[:, None] * (Ar1d - Acomb[:, None]) ** 2) * allowedPoint

    # sums over the short trailing axes as products with vectors of ones: several times faster on CPUs
    ones = partial(jnp.ones, dtype=post.dtype)
    postFeHMr = _dot(post, ones(nAr)).reshape(nFeH, nMr)
    post = post.reshape(nFeH, nMr, nAr)
    # Held as an array of its own. Left to the compiler, this sum is fused into what is read from it, and on a
    # GPU that made the standard deviation of A_r alone take 16 of the 26 ms of a batch of 400 DP2 stars.
    postMrAr = jax.lax.optimization_barrier(jnp.sum(post, axis=0))

    # The mass of each class is the posterior summed over the stretch of the axis it occupies; every class
    # lies on its own stretch at every [Fe/H] (locus.segmentAxis), so the masks are along the axis alone.
    # The sums are of weights scaled to 1 at the peak, so the largest is at least 1 and the log odds stay
    # finite down to the float underflow of the smaller (about e^-87).
    statistics = {}
    postT = jnp.sum(postFeHMr, axis=0)
    masks = jnp.stack([args["classAxisMasks"][name] for name in names])
    masses = jnp.sum(jnp.where(masks, postT[None, :], 0.0), axis=1)
    iWD = names.index("WD")
    tiny = jnp.finfo(post.dtype).tiny
    helium = jnp.sum(jnp.where(args["heliumAxisMask"], postT, 0.0))
    if perClass:
        # each mass is of its own class's scale, so they are put on one in logarithms
        logMass = jnp.log(masses) + classShift
        logTotal = jax.nn.logsumexp(logMass)
        for name, value in zip(names, logMass, strict=True):
            statistics[f"p_{name}"] = jnp.exp(value - logTotal)
        statistics["p_He"] = jnp.exp(jnp.log(helium) + classShift[iWD] - logTotal)
        logRest = jax.nn.logsumexp(jnp.where(jnp.arange(len(names)) == iWD, -jnp.inf, logMass))
        statistics["lnodds_WD"] = jnp.maximum(logMass[iWD] - logTotal, jnp.log(tiny)) - jnp.maximum(
            logRest - logTotal, jnp.log(tiny)
        )
    else:
        total = jnp.sum(masses)
        for name, mass in zip(names, masses, strict=True):
            statistics[f"p_{name}"] = mass / total
        statistics["p_He"] = helium / total
        # the rest as a sum of its own classes rather than the total less the white dwarfs, which in float32
        # loses everything below a part in 1e7 of the total and caps the odds of a sure white dwarf near e^16
        rest = jnp.sum(jnp.where(jnp.arange(len(names)) == iWD, 0.0, masses))
        statistics["lnodds_WD"] = jnp.log(jnp.maximum(masses[iWD], tiny)) - jnp.log(jnp.maximum(rest, tiny))
    chi2Class = _chi2GridMin(c, s, A0, Ar1d, kLast, kFirst).reshape(nFeH, nMr)
    reachable = jnp.isfinite(logPrior).reshape(nFeH, nMr)
    for name in names:
        inClass = reachable & args["classAxisMasks"][name][None, :]
        statistics[f"chi2min{name}"] = jnp.min(jnp.where(inClass, chi2Class, jnp.inf))
    # the white dwarf posterior along [Fe/H], which on a mass axis is the mass (WD_MASS)
    wd = args["classAxisMasks"]["WD"].astype(post.dtype)
    statistics[f"{cc.metallicity}_WD_quantile_median"] = getPosteriorQuantiles(
        FeH1d, _dot(postFeHMr * wd[None, :], ones(nMr))
    )[1]
    # the QSO class: its median redshift along the track, on the track's own points, so that the interval
    # to the next class on the axis never enters the interpolation
    if args["qsoAxisIndex"].size:
        _, statistics[QSO_REDSHIFT], _ = getPosteriorQuantiles(
            args["qsoRedshift1d"], postT[args["qsoAxisIndex"]]
        )
    else:
        statistics[QSO_REDSHIFT] = jnp.asarray(jnp.nan, dtype=post.dtype)
    # every estimate below is of the posterior within one class of star: the one GlobalParams.estimateClass
    # names, or else the most probable of the stellar classes. In the second case a QSO that no star comes
    # near has no stellar posterior left above the float underflow, and its stellar estimates are empty.
    if perClass:
        keep = args["estimateMask"].astype(post.dtype)
    else:
        stellar = jnp.array([names.index(name) for name in STELLAR_CLASSES])
        keep = masks[stellar[jnp.argmax(masses[stellar])]].astype(post.dtype)
    postFeHMr = postFeHMr * keep[None, :]
    QrWD = (
        jnp.zeros_like(args["QrGrid"], dtype=post.dtype)
        .at[args["QrIdxIndep"]]
        .add(postMrAr[args["QrColsIndep"]] * wd[args["QrColsIndep"], None])
        .at[args["QrIdxDepWD"]]
        .add(post[:, args["QrColsDepWD"], :])
    )
    postMrAr = postMrAr * keep[:, None]
    margMr = pnorm(postT * keep, args["dMr"])
    margFeH = pnorm(_dot(postFeHMr, ones(nMr)), args["dFeH"])
    margAr = pnorm(jnp.sum(postMrAr, axis=0), args["dAr"])
    QrWeights = (
        jnp.zeros_like(args["QrGrid"], dtype=post.dtype)
        .at[args["QrIdxIndep"]]
        .add(postMrAr[args["QrColsIndep"]])
        .at[args["QrIdxDep"]]
        .add(post[:, args["QrColsDep"], :] * keep[args["QrColsDep"]][None, :, None])
    )
    for q, value in zip(QUANTILE_NAMES, getPosteriorQuantiles(args["QrGrid"], QrWD), strict=True):
        statistics[f"{cc.abs_mag_ext_r}_WD_quantile_{q}"] = value

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
        pdfs.append((args["MrTrueGrid"], MrTrueWeights, "Mr"))
    for values, pdf, name in pdfs:
        for q, value in zip(QUANTILE_NAMES, getPosteriorQuantiles(values, pdf), strict=True):
            statistics[f"{name}_quantile_{q}"] = value
        # the second moment of the same sampled density. It says something the quantiles do not for a
        # posterior with two branches, where the spread is wide while the 16-84 interval can be narrow.
        statistics[f"{name}_std"] = getStats(values, pdf)[1]

    cubes = {}
    if returnPosteriors:
        logPriorA = jnp.broadcast_to(logPrior + logDust + logWidth, logPrior.shape)
        prior = (
            jnp.exp(logPriorA[:, None] - 0.5 * wDust[:, None] * (Ar1d - Astar[:, None]) ** 2) * allowedPoint
        ).reshape(nFeH, nMr, nAr)
        like = jnp.exp(-0.5 * (c[:, None] + s * (Ar1d - A0[:, None]) ** 2 - chi2min)).reshape(nFeH, nMr, nAr)
        cubes = {"prior": prior, "like": like, "post": post}
        for k, cube in enumerate((prior, like, post)):
            margs = getMargDistr3D(cube, args["dMr"], args["dFeH"], args["dAr"])
            cubes[f"margMr{k}"], cubes[f"margFeH{k}"], cubes[f"margAr{k}"] = margs
    return chi2min, statistics, cubes, QrWeights


@partial(jax.jit, static_argnames=("computeMrTrue", "returnPosteriors"))
def _starBatch(data, logPriorGrid, args, computeMrTrue, returnPosteriors):
    star = partial(
        starPosterior,
        logPriorGrid=logPriorGrid,
        args=args,
        computeMrTrue=computeMrTrue,
        returnPosteriors=returnPosteriors,
    )
    chi2min, statistics, cubes, QrWeights = jax.vmap(star)(data)
    # Of the distance modulus, which is the Qr posterior turned round: the measure does not depend on which
    # way the axis runs. It is taken before the error on r is folded in, which moves both peaks together, and
    # on the whole batch at once, which is several times faster than star by star (see stats.multimodality).
    statistics[cc.multimodality] = multimodality(args["QrGrid"], QrWeights)
    return chi2min, statistics, cubes


def _dot(a, b):
    """Matrix product in full float precision (on recent GPUs the default for float32 is TF32)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _chi2GridMin(c, s, A0, grid, kLast, kFirst=0):
    """Smallest c + s * (A - A0)^2 over grid[kFirst..kLast] of a uniform grid, for every locus point.

    The function is convex in A, so the minimum is at one of the two grid values around A0.
    """
    if grid.size == 1:
        return c + s * (grid[0] - A0) ** 2
    k = jnp.floor((A0 - grid[0]) / (grid[1] - grid[0])).astype(jnp.int32)
    below = grid[jnp.clip(k, kFirst, kLast)]
    above = grid[jnp.clip(k + 1, kFirst, kLast)]
    return c + s * jnp.minimum((below - A0) ** 2, (above - A0) ** 2)


def arBandEdges(MrTrue, rmag, curve, mu, arMap, scale, offset, band, extragalactic):
    """The A_r band of each locus point (GlobalParams.ArCurveMode "band"), as its lower and upper edge.

    The column in front of a star at distance modulus mu is C(mu) = curve(mu) A_r(map), and the band is
    C / scale - offset <= A <= scale C + offset, the box's factors both ways. Point i at extinction A sits at
    mu = r - MrTrue_i - A, so an edge e is met where A = e(r - MrTrue_i - A); r - mu - e(mu) is monotone in
    mu, which makes that an interpolation. A point of an extragalactic segment lies behind all of the dust,
    so its column is the map's. A point off the band rows, or on a sightline whose curve ends at zero, keeps
    the box: 0 to infinity here, which the caller cuts to the box.
    """
    column = curve * arMap
    edges = []
    for edge, far in (
        (column / scale - offset, arMap / scale - offset),
        (scale * column + offset, scale * arMap + offset),
    ):
        A = jnp.interp(MrTrue, (rmag - mu - edge)[::-1], edge[::-1])
        edges.append(jnp.where(extragalactic, far, A))
    inBand = band & ((column[-1] > 0) | extragalactic)
    return jnp.where(inBand, jnp.maximum(edges[0], 0.0), 0.0), jnp.where(inBand, edges[1], jnp.inf)


def _logPrior(priorGrid, params):
    """Log prior on the locus, the points that only pad an isochrone to the grid excluded.

    The prior of a point is its density times the area of its cell (GlobalParams.priorWeights).
    """
    return jnp.log(jnp.asarray(priorGrid) * params.locusValid * params.priorWeights)


def _priorClassOdds(priorGrid, params):
    """Log prior odds of white dwarf against the rest, per r bin, with the cell weights the fit uses."""
    prior = np.asarray(priorGrid, dtype=np.float64) * params.locusValid * params.priorWeights
    isWD = np.broadcast_to(params.classAxisMasks["WD"], (params.FeH1d.size, params.Mr1d.size)).ravel()
    massWD, rest = prior[:, isWD].sum(axis=1), prior[:, ~isWD].sum(axis=1)
    tiny = np.finfo(np.float64).tiny
    return np.log(np.maximum(massWD, tiny)) - np.log(np.maximum(rest, tiny))


def _arGridLengths(arMax, Ar1d):
    """Length of the A_r grid each star needs: all values allowed by its prior, the first value above the
    limit (so that the A_r percentiles are interpolated exactly as on the full grid) and one more for the
    float32 rounding of the limit."""
    need = np.minimum(np.searchsorted(Ar1d, arMax, side="right") + 2, Ar1d.size)
    lengths = np.array(sorted({n for n in AR_GRID_LENGTHS if n < Ar1d.size} | {Ar1d.size}))
    return lengths[np.searchsorted(lengths, need)]


def _qualityFlags(chi2min, statistics, colorsErr, arMax, noMagnitude, globalParams, rmag=None):
    """One bit per thing worth knowing about a star's fit, as described by the FLAG_ constants.

    Everything here is a property of the answer rather than of the star, and none of it removes a row: a
    catalog that quietly drops what it cannot fit is harder to use than one that says so.
    """
    flags = np.zeros(chi2min.size, dtype=np.int32)
    # the absolute magnitude itself when the locus is parametrised by tLoc, since that is where a giant and a
    # dwarf solution sit far apart
    name = "Mr" if "Mr_quantile_median" in statistics else cc.abs_mag_r
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

    # A posterior narrower than this is not a precise answer but a collapsed one: the weight has landed in a
    # single cell of the grid. Measured on the DP2 run against Gaia parallaxes, over unflagged stars with a
    # parallax good to ten per cent, the half-width where this happens is sharp. Above 0.1 mag the answers
    # behave: 0.3 to 5 per cent have [Fe/H] pinned at an end of its grid and the median distance is off by
    # 0.41 to 0.47 mag. Below it they do not: 40 per cent are pinned, and between 0.02 and 0.05 mag the
    # median distance is off by 9.5 magnitudes. That is 1.3 per cent of unflagged stars, and the row says so
    # rather than looking like the most precise answer in the catalog.
    width = 0.5 * (
        statistics[f"{cc.abs_mag_ext_r}_quantile_hi"] - statistics[f"{cc.abs_mag_ext_r}_quantile_lo"]
    )
    flags |= np.where(np.isfinite(width) & (width < COLLAPSED_WIDTH), FLAG_COLLAPSED, 0)

    flags |= np.where((colorsErr > MISSING_COLOR_ERR).any(axis=1), FLAG_COLOR_MISSING, 0)
    flags |= np.where(noMagnitude, FLAG_NO_MAGNITUDE, 0)
    if rmag is not None:
        flags |= np.where(beyondRecipe(rmag), FLAG_BEYOND_RECIPE, 0)
    return flags


def _batchLimit(globalParams, nAr, batchBytes=None):
    """How many stars of this A_r grid length fit a batch into the memory budget."""
    cells = BATCH_BYTES_PER_CELL * globalParams.FeH1d.size * globalParams.Mr1d.size * nAr
    return max(1, int((BATCH_BYTES if batchBytes is None else batchBytes) // cells))


def _toHost(out):
    """A batch result as plain arrays, releasing the device buffers it was computed into.

    All of them in one transfer: copied one at a time, each waits on the device by itself, and with the 25 or
    so arrays of a batch that took a sixth of the time of a run.
    """
    return jax.device_get(out)


def _collect(parts, nStars):
    """Assemble per-batch results into arrays over all stars (padded entries repeat a real star)."""
    first = np.asarray(parts[0][1])
    out = np.empty((nStars,) + first.shape[1:], dtype=first.dtype)
    for b, values in parts:
        out[b] = np.asarray(values)
    return out
