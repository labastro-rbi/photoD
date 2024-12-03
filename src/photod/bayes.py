from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from photod.parameters import GlobalParams
from photod.plotting import plotStar
from photod.priors import getPriorMapIndex, make3Dprior
from photod.results import BayesResults
from photod.stats import Entropy, getMargDistr3D, getStats


def makeBayesEstimates3d(
    starsData: pd.DataFrame,
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
    func = partial(loopOverEachStar, globalParams=globalParams.getArgs(), returnAllInfo=returnAllInfo)
    results = BayesResults(*jax.lax.map(func, (colors, colorsErr, priorIndices), batch_size=batchSize))
    # Create the DataFrame with the expectation values and uncertainties
    estimatesDf = getEstimates(selectedStars, results.chi2min, results.statistics, do3D=True)
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
def loopOverEachStar(starData, globalParams, returnAllInfo):
    """Internal method with the logic to be run for each star."""
    colors, colorsErr, priorIndices = starData
    locusColors, Ar1d, FeH1d, Mr1d, priorGrid, dFeH, dMr = globalParams
    chi2map = calculateChi2(colors, colorsErr, locusColors)
    dAr, likeCube, priorCube, chi2min = likeAndPrior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices)
    postCube = priorCube * likeCube
    margpostMr, margpostFeH, margpostAr = getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr)
    statistics = postProcess(Ar1d, FeH1d, Mr1d, margpostMr, margpostFeH, margpostAr)
    otherInfo = [likeCube, priorCube, postCube, margpostMr, margpostFeH, margpostAr] if returnAllInfo else []
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
    margpostMr = {}
    margpostFeH = {}
    margpostAr = {}
    margpostMr[0], margpostFeH[0], margpostAr[0] = getMargDistr3D(priorCube, dMr, dFeH, dAr)
    margpostMr[1], margpostFeH[1], margpostAr[1] = getMargDistr3D(likeCube, dMr, dFeH, dAr)
    margpostMr[2], margpostFeH[2], margpostAr[2] = getMargDistr3D(postCube, dMr, dFeH, dAr)
    return margpostMr, margpostFeH, margpostAr


def postProcess(Ar1d, FeH1d, Mr1d, margpostMr, margpostFeH, margpostAr):
    """Get expectation values and uncertainties marginalize and get statistics."""
    MrEst, MrEstUnc = getStats(Mr1d, margpostMr[2])
    FeHEst, FeHEstUnc = getStats(FeH1d, margpostFeH[2])
    ArEst, ArEstUnc = getStats(Ar1d, margpostAr[2])
    return {
        "FeHEst": FeHEst,
        "FeHEstUnc": FeHEstUnc,
        "MrEst": MrEst,
        "MrEstUnc": MrEstUnc,
        "ArEst": ArEst,
        "ArEstUnc": ArEstUnc,
        "MrdS": Entropy(margpostMr[2]) - Entropy(margpostMr[0]),
        "FeHdS": Entropy(margpostFeH[2]) - Entropy(margpostFeH[0]),
        "ArdS": Entropy(margpostAr[2]) - Entropy(margpostAr[0]),
    }


def getEstimates(starsData, chi2min, statistics, do3D=False):
    """Construct the Bayes estimates Pandas DataFrame."""
    estimatesDf = pd.DataFrame(
        {
            "glon": starsData["glon"],
            "glat": starsData["glat"],
            "FeHEst": statistics["FeHEst"],
            "FeHUnc": statistics["FeHEstUnc"],
            "MrEst": statistics["MrEst"],
            "MrUnc": statistics["MrEstUnc"],
            "chi2min": chi2min,
            "MrdS": statistics["MrdS"],
            "FeHdS": statistics["FeHdS"],
        }
    )
    if do3D:
        estimatesDf["ArEst"] = statistics["ArEst"]
        estimatesDf["ArUnc"] = statistics["ArEstUnc"]
        estimatesDf["ArdS"] = statistics["ArdS"]
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
