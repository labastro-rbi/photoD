from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from photod.parameters import GlobalParams
from photod.plotting import plotStar
from photod.priors import getPriorMapIndex, make3Dprior
from photod.results import BayesResults
from photod.stats import Entropy, getMargDistr3D, getPosteriorQuantiles, getQrQuantiles


def makeBayesEstimates3d(
    starsData: pd.DataFrame,
    priorGrid: np.ndarray,
    globalParams: GlobalParams,
    iStart: int = 0,
    iEnd: int = -1,
    batchSize: int = 10000,
    returnAllInfo: bool = False,
):
    """Compute the Bayes Estimates for stars in `data`"""
    if iEnd < iStart:
        iStart = 0
        iEnd = len(starsData)
    # Preselect stars according to the provided indices
    selectedStars, colors, colorsErr, priorIndices = selectStarsInRange(starsData, globalParams, iStart, iEnd)
    # Use `jax.lax.map` to batch computations with scan/vmap and use memory efficiently.
    # The BayesResult object is populated with the chi2min and statistics for each star.
    # If `returnAllInfo` is True, the prior and posterior arrays will be included in the results.
    func = partial(
        loopOverEachStar,
        priorGrid=priorGrid,
        globalParams=globalParams.getArgs(),
        returnAllInfo=returnAllInfo,
    )
    results = BayesResults(*jax.lax.map(func, (colors, colorsErr, priorIndices), batch_size=batchSize))
    # Create the DataFrame with the expectation values and uncertainties
    estimatesDf = getEstimates(selectedStars, results)
    return estimatesDf, results


def selectStarsInRange(catalog, params, iStart, iEnd):
    """Selects the stars whose indices fall in [iStart, iEnd[, and respective priors."""
    # Select colors for stars with indices in [iStart, iEnd[
    selectedStars = catalog.iloc[iStart:iEnd]
    colors = selectedStars[list(params.fitColors)].to_numpy(dtype=np.float64)
    colorErrNames = [color + "Err" for color in params.fitColors]
    colorsErr = selectedStars[colorErrNames].to_numpy(dtype=np.float64)
    # Select priors for stars in [iStart, iEnd[
    priorIndices = jnp.array(getPriorMapIndex(selectedStars["rmag"]))
    return selectedStars, colors, colorsErr, priorIndices


@partial(jax.jit, static_argnames="returnAllInfo")
def loopOverEachStar(starData, priorGrid, globalParams, returnAllInfo):
    """Internal method with the logic to be run for each star."""
    colors, colorsErr, priorIndices = starData
    locusColors, Ar1d, FeH1d, Mr1d, dFeH, dMr, QrGrid, QrIndices = globalParams
    # Calculate the likelihoods in log-space
    chi2map = calculateChi2(colors, colorsErr, locusColors)
    dAr, likeCube, priorCube, chi2min = likeAndPrior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices)
    postCube = priorCube * likeCube
    # Calculate the marginal posteriors
    margPost = getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr)
    statistics = postProcess(Ar1d, FeH1d, Mr1d, postCube, QrGrid, QrIndices, *margPost)
    otherInfo = [likeCube, priorCube, postCube, *margPost] if returnAllInfo else []
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


def getEstimates(starsData, results):
    """Construct the Bayes estimates Pandas DataFrame."""
    estimatesDf = pd.DataFrame(
        {
            "glon": starsData["glon"],
            "glat": starsData["glat"],
            "chi2min": results.chi2min,
            **results.statistics,
        }
    )
    return estimatesDf


def plotStars(
    starsData,
    margpostAr,
    margpostMr,
    margpostFeH,
    likeCube,
    priorCube,
    postCube,
    mdLocus,
    xLabel,
    yLabel,
    Mr1d,
    FeH1d,
    Ar1d,
    starIndices,
):
    """Create the plots for the specified stars."""

    def getValueForStar(statDict, index):
        return {key: value[index] for key, value in statDict.items()}

    # Iterate over indices of stars in the results
    for i in starIndices:
        if i not in starsData.index:
            print(f"Results for star {i} were not found. Skipping...")
            continue

        print(f"Plotting star {i}...")
        QrEst, QrEstUnc = plotStar(
            starsData.loc[i],
            getValueForStar(margpostAr, i),
            likeCube[i],
            priorCube[i],
            postCube[i],
            mdLocus,
            xLabel,
            yLabel,
            Mr1d,
            getValueForStar(margpostMr, i),
            FeH1d,
            getValueForStar(margpostFeH, i),
            Ar1d,
        )
        print(QrEst, QrEstUnc)
