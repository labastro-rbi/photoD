from functools import partial

import astropy.units as u
import jax
import jax.numpy as jnp
import nested_pandas as npd
import numpy as np
import pandas as pd
from lsdb.catalog.map_catalog import MapCatalog
from lsdb.core.search.moc_search import MOCSearch
from mocpy import MOC

from photod.parameters import GlobalParams
from photod.priors import getPriorMapIndex, initializePriorGrid, make3Dprior
from photod.results import BayesResults
from photod.stats import Entropy, getMargDistr3D, getPosteriorQuantiles, getQrQuantiles

from photod.column_map.catalog import m as cc

def makeBayesEstimates3d(
    starsData: npd.NestedFrame,
    priorGrid: np.ndarray,
    globalParams: GlobalParams,
    batchSize: int = 100,
    returnPosteriors: bool = False,
):
    """Compute the bayes statistics for all the stars in a catalog partition.

    Used for fast, large-scale processing. It leverages parallelization with JAX.
    """
    colorsAndIndices = getColorsAndPriorIndices(starsData, globalParams)
    # Use `jax.lax.map` to batch computations with scan/vmap and use memory efficiently.
    # The BayesResult object is populated with the chi2min and statistics for each star.
    # If `returnAllInfo` is True, the prior and posterior arrays will be included in the results.
    func = partial(
        loopOverEachStar,
        priorGrid=priorGrid,
        globalParams=globalParams.getArgs(),
        returnPosteriors=returnPosteriors,
    )
    results = BayesResults(*jax.lax.map(func, colorsAndIndices, batch_size=batchSize))
    # Create the DataFrame with the expectation values and uncertainties
    estimatesDf = pd.DataFrame(
        {
            "glon": starsData["glon"],
            "glat": starsData["glat"],
            "chi2min": results.chi2min,
            **results.statistics,
        }
    )
    return estimatesDf, results


def makeBayesPosteriors3d(
    starsData: npd.NestedFrame,
    mapCatalog: MapCatalog,
    globalParams: GlobalParams,
):
    """Compute the bayes posteriors for all the stars in a DataFrame.

    The posterior arrays are extremely large in memory and, therefore, this method should
    only be used with a handful set of stars.

    It does not use JAX for parallelization.
    """
    maxMapOrder = mapCatalog.hc_structure.pixel_tree.get_max_depth()
    results = []
    for index in range(len(starsData)):
        star = starsData.iloc[[index]]
        ra = star["ra"].to_numpy() * u.deg
        dec = star["dec"].to_numpy() * u.deg
        mapMoc = MOC.from_lonlat(ra, dec, max_norder=maxMapOrder)
        # Find the map partition corresponding to each star
        mapPartitionDf = mapCatalog.search(MOCSearch(mapMoc, fine=False)).compute()
        priorGrid = initializePriorGrid(mapPartitionDf, globalParams)
        priorGrid = jax.numpy.array(list(priorGrid.values()))
        _, starResults = makeBayesEstimates3d(star, priorGrid, globalParams, returnPosteriors=True)
        results.append(starResults)
    return results


def getColorsAndPriorIndices(catalog, params):
    colors = catalog[list(params.fitColors)].to_numpy(dtype=np.float64)
    colorErrNames = [color + "Err" for color in params.fitColors]
    colorsErr = catalog[colorErrNames].to_numpy(dtype=np.float64)
    priorIndices = jnp.array(getPriorMapIndex(catalog["rmag"]))
    return colors, colorsErr, priorIndices


@partial(jax.jit, static_argnames="returnPosteriors")
def loopOverEachStar(starData, priorGrid, globalParams, returnPosteriors):
    """Internal method with the logic to be run for each star."""
    colors, colorsErr, priorIndices = starData
    locusColors, Ar1d, FeH1d, Mr1d, dFeH, dMr, QrGrid, QrIndices = globalParams
    chi2map = calculateChi2(colors, colorsErr, locusColors)
    dAr, likeCube, priorCube, chi2min = likeAndPrior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices)
    postCube = priorCube * likeCube
    margPost = getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr)
    statistics = postProcess(Ar1d, FeH1d, Mr1d, postCube, QrGrid, QrIndices, *margPost)
    otherInfo = [likeCube, priorCube, postCube, *margPost] if returnPosteriors else []
    return chi2min, statistics, *otherInfo


def calculateChi2(colors, colorsErr, locusColors):
    """Compute chi2 map using provided 3D model locus."""
    # Remove the last axis (color), hence axis=-1
    return jnp.sum(jnp.square((colors - locusColors) / colorsErr), axis=-1)


def likeAndPrior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices):
    """Compute the likelihood map, the 3D (Mr, FeH, Ar) prior from 2D (Mr, FeH) prior
    using uniform prior for Ar, and the chi2min."""
    likeGrid = jnp.exp(-0.5 * chi2map)
    likeCube = likeGrid.reshape(FeH1d.size, Mr1d.size, Ar1d.size)
    dAr = Ar1d[1] - Ar1d[0] if Ar1d.size > 1 else 0.01
    ## generate 3D (Mr, FeH, Ar) prior from 2D (Mr, FeH) prior using uniform prior for Ar
    prior2d = priorGrid[priorIndices].reshape(FeH1d.size, Mr1d.size)
    priorCube = make3Dprior(prior2d, Ar1d.size)
    return dAr, likeCube, priorCube, jnp.min(chi2map)


def getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr):
    """Get posterior information"""
    margpostMr, margpostFeH, margpostAr = {}, {}, {}
    for idx, cube in enumerate([priorCube, likeCube, postCube]):
        distrMr, distrFeH, distrAr = getMargDistr3D(cube, dMr, dFeH, dAr)
        margpostMr[idx] = distrMr
        margpostFeH[idx] = distrFeH
        margpostAr[idx] = distrAr
    return margpostMr, margpostFeH, margpostAr


def postProcess(Ar1d, FeH1d, Mr1d, postCube, QrGrid, QrIndices, margpostMr, margpostFeH, margpostAr):
    """Get expectation values and uncertainties marginalize and get statistics."""

    MrQuantiles = getPosteriorQuantiles(Mr1d, margpostMr[2])
    FeHQuantiles = getPosteriorQuantiles(FeH1d, margpostFeH[2])
    ArQuantiles = getPosteriorQuantiles(Ar1d, margpostAr[2])
    QrQuantiles = getQrQuantiles(postCube, QrGrid, QrIndices)

    # Calculate the quantiles for Mr, FeH and Ar and add them as columns to the result
    quantile_names = ["lo", "median", "hi"]
    posteriorsDict = {
        f"{statisticsName}_quantile_{quantile_names[i]}": quantile
        for quantiles, statisticsName in zip(
            [MrQuantiles, FeHQuantiles, ArQuantiles, QrQuantiles], ["Mr", "FeH", "Ar", "Qr"]
        )
        for i, quantile in enumerate(quantiles)
    }

    return {
        **posteriorsDict,
        "MrdS": Entropy(margpostMr[2]) - Entropy(margpostMr[0]),
        "FeHdS": Entropy(margpostFeH[2]) - Entropy(margpostFeH[0]),
        "ArdS": Entropy(margpostAr[2]) - Entropy(margpostAr[0]),
    }


def getEstimatesMeta():
    """Creates an empty pd.DataFrame with the meta for the results"""
    quantileCols = [
        f"{statisticsName}_quantile_{quantile}"
        for statisticsName in ["Mr", "FeH", "Ar", "Qr"]
        for quantile in ["lo", "median", "hi"]
    ]
    estimateCols = sorted([*quantileCols, "MrdS", "FeHdS", "ArdS"])
    colNames = ["glon", "glat", "chi2min", *estimateCols]
    meta = npd.NestedFrame.from_dict({col: pd.Series([], dtype=np.float32) for col in colNames})
    meta.index.name = "_healpix_29"
    return meta
