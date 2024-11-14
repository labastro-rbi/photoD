import jax
import jax.numpy as jnp
import nested_pandas as nd
import numpy as np
from astropy.table import Table

import photod.locus as lt
from photod.plotting import plot_star
from photod.priors import getBayesConstants, getPriorMapIndex, make3Dprior, readPriors
from photod.stats import Entropy, getMargDistr3D, getStats


def makeBayesEstimates3D(
    catalog: nd.NestedFrame,
    fitColors: tuple,
    locusData: Table,
    locus3DList: dict[str, np.ndarray],
    ArGridList: dict[int, np.ndarray],
    priorsRootName: str,
    outfile: str,
    iStart: int = 0,
    iEnd: int = -1,
    plotStars: list = [],
    MrColumn: str = "Mr",
):
    if iEnd < iStart:
        iStart = 0
        iEnd = len(catalog)

    (
        FeH1d,
        Mr1d,
        dFeH,
        dMr,
        mdLocus,
        priorGrid,
        priorIndices,
        xLabel,
        yLabel,
    ) = _setup_bayes3d(catalog, locusData, priorsRootName, MrColumn)

    colors, colorsErr, Ar1d, locusColors, priorGrid, priorIndices, plotStars = prepare_arguments_for_jax(
        catalog,
        fitColors,
        ArGridList,
        locus3DList,
        priorGrid,
        priorIndices,
        plotStars,
        iStart,
        iEnd,
    )

    likeCube, priorCube, chi2min, postCube, margpostMr, margpostFeH, margpostAr, statistics = (
        run_bayes_with_jax(
            colors, colorsErr, priorIndices, locusColors, Ar1d, FeH1d, Mr1d, dFeH, dMr, priorGrid
        )
    )

    write_bayes_estimates(catalog, chi2min, statistics, outfile, iStart, iEnd, do3D=True)

    plot_stars(
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
    )


def _setup_bayes3d(catalog, locusData, priorsRootName, MrColumn):
    # read maps with priors (and interpolate on the Mr-FeH grid given by locusData, which is same for all stars)
    priorGrid = readPriors(rootname=priorsRootName, locusData=locusData, MrColumn=MrColumn)
    # get prior map indices using observed r band mags
    priorIndices = getPriorMapIndex(catalog["rmag"])
    # Mr and FeH 1-D grid properties extracted from locus data (same for all stars)
    xLabel = "FeH"
    yLabel = "Mr"
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    # grid step
    dFeH = FeH1d[1] - FeH1d[0]
    dMr = Mr1d[1] - Mr1d[0]
    # metadata for plotting likelihood maps below
    FeHmin = np.min(FeH1d)
    FeHmax = np.max(FeH1d)
    FeHNpts = FeH1d.size
    MrFaint = np.max(Mr1d)
    MrBright = np.min(Mr1d)
    MrNpts = Mr1d.size
    mdLocus = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
    print("Mr1d=", np.min(Mr1d), np.max(Mr1d), len(Mr1d))
    print("MrBright, MrFaint=", MrBright, MrFaint)
    return (
        FeH1d,
        Mr1d,
        dFeH,
        dMr,
        mdLocus,
        priorGrid,
        priorIndices,
        xLabel,
        yLabel,
    )


def prepare_arguments_for_jax(
    catalog, fitColors, ArGridList, locus3DList, priorGrid, priorIndices, plotStars, iStart, iEnd
):
    """Load the bayes method arguments and selects the stars whose indices fall in [iStart, iEnd[."""
    # TODO: Create object to hold these arguments?
    Ar1d, locusColors3d = ArGridList["ArLarge"], locus3DList["ArLarge"]  # We fixed this
    # Stack all locus data by colors
    locusColors = np.stack([locusColors3d[color] for color in fitColors], axis=-1)
    # Select colors for stars with indices in [iStart, iEnd[
    colors = catalog[list(fitColors)].to_numpy(dtype=np.float64)[iStart:iEnd]
    colorErrNames = [color + "Err" for color in fitColors]
    colorsErr = catalog[colorErrNames].to_numpy(dtype=np.float64)[iStart:iEnd]
    # Select priors for stars in [iStart, iEnd[
    priorGrid = jnp.array(list(priorGrid.values()))
    priorIndices = jnp.array(priorIndices)[iStart:iEnd]
    # Filter stars to plot to make sure they are within [iStart, iEnd[
    plotStars = [star for star in plotStars if star >= iStart and star < iEnd]
    print(f"Plotting stars in [{iStart}, {iEnd}[: ", plotStars)
    return colors, colorsErr, Ar1d, locusColors, priorGrid, priorIndices, plotStars


def run_bayes_with_jax(colors, colorsErr, priorIndices, locusColors, Ar1d, FeH1d, Mr1d, dFeH, dMr, priorGrid):
    """Use JAX's vmap to iterate over each star. The iterable arguments are `colors`, `colorsErr`, `priorIndices`."""
    jax_func = jax.jit(jax.vmap(loop_over_each_star, in_axes=(0, 0, 0, None, None, None, None, None, None, None)))
    return jax_func(colors, colorsErr, priorIndices, locusColors, Ar1d, FeH1d, Mr1d, priorGrid, dFeH, dMr)


def loop_over_each_star(colors, colorsErr, priorIndices, locusColors, Ar1d, FeH1d, Mr1d, priorGrid, dFeH, dMr):
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


def write_bayes_estimates(catalog, chi2min, statistics, outfile, iStart, iEnd, do3D=False):
    """Output the bayes estimates to a file."""
    fout = open(outfile, "w")
    if do3D:
        fout.write(
            "      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  ArEst  ArUnc  chi2min    MrdS     FeHdS      ArdS \n"
        )
    else:
        fout.write("      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  chi2min    MrdS     FeHdS \n")
    for i in range(iStart, iEnd):
        r1 = catalog["glon"][i]
        r2 = catalog["glat"][i]
        r3 = statistics["FeHEst"][i]
        r4 = statistics["FeHEstUnc"][i]
        r5 = statistics["MrEst"][i]
        r6 = statistics["MrEstUnc"][i]
        r7 = chi2min[i]
        r8 = statistics["MrdS"][i]
        r9 = statistics["FeHdS"][i]
        s = str("%12.8f " % r1) + str("%12.8f  " % r2) + str("%6.2f  " % r3) + str("%5.2f  " % r4)
        s = s + str("%6.2f  " % r5) + str("%5.2f  " % r6)
        if do3D:
            r15 = statistics["ArEst"][i]
            r16 = statistics["ArEstUnc"][i]
            r19 = statistics["ArdS"][i]
            s = s + str("%6.2f  " % r15) + str("%5.2f  " % r16) + str("%5.2f  " % r7) + str("%8.1f  " % r8)
            s = s + str("%8.1f  " % r9) + str("%8.1f  " % r19) + str("%8.0f" % i) + "\n"
        else:
            s = s + str("%5.2f  " % r7) + str("%8.1f  " % r8) + str("%8.1f  " % r9) + "\n"
        fout.write(s)
    fout.close()


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
