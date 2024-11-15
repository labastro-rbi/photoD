import jax
import jax.numpy as jnp
import nested_pandas as nd
import numpy as np
import pandas as pd

from photod.parameters import GlobalParams
from photod.plotting import plot_star
from photod.priors import getPriorMapIndex, make3Dprior
from photod.results import BayesResults
from photod.stats import Entropy, getMargDistr3D, getStats


def make_bayes_estimates_3d(catalog, params: GlobalParams, iStart=0, iEnd=-1):
    if iEnd < iStart:
        iStart = 0
        iEnd = len(catalog)
    colors, colorsErr, priorIndices = select_stars_in_range(catalog, params, iStart, iEnd)
    results = run_bayes_with_jax(colors, colorsErr, priorIndices, *params.get_args())
    return results, get_estimates_df(catalog, results.chi2min, results.statistics, iStart, iEnd, do3D=True)


def select_stars_in_range(catalog, params, iStart, iEnd):
    """Selects the stars whose indices fall in [iStart, iEnd[, and respective priors."""
    # Select colors for stars with indices in [iStart, iEnd[
    colors = catalog[list(params.fitColors)].to_numpy(dtype=np.float64)[iStart:iEnd]
    colorErrNames = [color + "Err" for color in params.fitColors]
    colorsErr = catalog[colorErrNames].to_numpy(dtype=np.float64)[iStart:iEnd]
    # Select priors for stars in [iStart, iEnd[
    priorIndices = getPriorMapIndex(catalog["rmag"])
    priorIndices = jnp.array(priorIndices)[iStart:iEnd]
    return colors, colorsErr, priorIndices


def run_bayes_with_jax(colors, colorsErr, priorIndices, *global_params):
    """Use JAX's vmap to iterate over each star. The iterable arguments are `colors`, `colorsErr`, `priorIndices`."""
    jax_func = jax.jit(jax.vmap(loop_over_each_star, in_axes=(0,) * 3 + (None,) * 7))
    return BayesResults(*jax_func(colors, colorsErr, priorIndices, *global_params))


def loop_over_each_star(
    colors, colorsErr, priorIndices, locusColors, Ar1d, FeH1d, Mr1d, priorGrid, dFeH, dMr
):
    """Internal method with the logic to be run for each star."""
    chi2map = calculate_chi2(colors, colorsErr, locusColors)
    dAr, likeCube, priorCube, chi2min = like_and_prior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices)
    postCube = priorCube * likeCube
    margpostMr, margpostFeH, margpostAr, statistics = post_process(
        Ar1d, FeH1d, Mr1d, dAr, dFeH, dMr, likeCube, postCube, priorCube
    )
    return likeCube, priorCube, chi2min, postCube, margpostMr, margpostFeH, margpostAr, statistics


def calculate_chi2(colors, colorsErr, locusColors):
    """Compute chi2 map using provided 3D model locus."""
    # Remove the last axis (color), hence axis=-1
    return jnp.sum(jnp.square((colors - locusColors) / colorsErr), axis=-1)


def like_and_prior(Ar1d, FeH1d, Mr1d, chi2map, priorGrid, priorIndices):
    """Compute the likelihood map, the 3D (Mr, FeH, Ar) prior from 2D (Mr, FeH) prior
    using uniform prior for Ar, and the chi2min."""
    likeGrid = jnp.exp(-0.5 * chi2map)
    likeCube = likeGrid.reshape(FeH1d.size, Mr1d.size, Ar1d.size)
    dAr = Ar1d[1] - Ar1d[0] if Ar1d.size > 1 else 0.01
    ## generate 3D (Mr, FeH, Ar) prior from 2D (Mr, FeH) prior using uniform prior for Ar
    prior2d = priorGrid[priorIndices].reshape(FeH1d.size, Mr1d.size)
    priorCube = make3Dprior(prior2d, Ar1d.size)
    return dAr, likeCube, priorCube, jnp.min(chi2map)


def post_process(Ar1d, FeH1d, Mr1d, dAr, dFeH, dMr, likeCube, postCube, priorCube):
    """Get expectation values and uncertainties marginalize and get statistics."""
    margpostMr = {}
    margpostFeH = {}
    margpostAr = {}

    margpostMr[0], margpostFeH[0], margpostAr[0] = getMargDistr3D(priorCube, dMr, dFeH, dAr)
    margpostMr[1], margpostFeH[1], margpostAr[1] = getMargDistr3D(likeCube, dMr, dFeH, dAr)
    margpostMr[2], margpostFeH[2], margpostAr[2] = getMargDistr3D(postCube, dMr, dFeH, dAr)

    MrEst, MrEstUnc = getStats(Mr1d, margpostMr[2])
    FeHEst, FeHEstUnc = getStats(FeH1d, margpostFeH[2])
    ArEst, ArEstUnc = getStats(Ar1d, margpostAr[2])

    statistics = {
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

    return margpostMr, margpostFeH, margpostAr, statistics


def get_estimates_df(catalog, chi2min, statistics, iStart, iEnd, do3D=False):
    """Construct the Bayes estimates Pandas DataFrame."""

    def slice_statistics_arrays(statistics, iStart, iEnd):
        """Slice arrays in the statistics dictionary"""
        return {key: value[iStart:iEnd] for key, value in statistics.items()}

    catalog_slice = catalog.iloc[iStart:iEnd]
    chi2min_slice = chi2min[iStart:iEnd]
    statistics_slice = slice_statistics_arrays(statistics, iStart, iEnd)
    estimates_df = pd.DataFrame(
        {
            "glon": catalog_slice["glon"],
            "glat": catalog_slice["glat"],
            "FeHEst": statistics_slice["FeHEst"],
            "FeHUnc": statistics_slice["FeHEstUnc"],
            "MrEst": statistics_slice["MrEst"],
            "MrUnc": statistics_slice["MrEstUnc"],
            "chi2min": chi2min_slice,
            "MrdS": statistics_slice["MrdS"],
            "FeHdS": statistics_slice["FeHdS"],
        }
    )
    if do3D:
        estimates_df["ArEst"] = statistics_slice["ArEst"]
        estimates_df["ArUnc"] = statistics_slice["ArEstUnc"]
        estimates_df["ArdS"] = statistics_slice["ArdS"]
    return estimates_df


def plot_stars(
    catalog,
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
    plotStars,
):
    """Create all plots for the specified stars."""

    def get_value_for_star(stat_dict, index):
        return {key: value[index] for key, value in stat_dict.items()}

    for i in plotStars:
        print(f"Plotting star {i}")
        QrEst, QrEstUnc = plot_star(
            catalog.iloc[i],
            get_value_for_star(margpostAr, i),
            likeCube[i],
            priorCube[i],
            postCube[i],
            mdLocus,
            xLabel,
            yLabel,
            Mr1d,
            get_value_for_star(margpostMr, i),
            FeH1d,
            get_value_for_star(margpostFeH, i),
            Ar1d,
        )
        print(QrEst, QrEstUnc)
