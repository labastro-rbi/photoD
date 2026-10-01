from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from photod.stats import getMargDistr, getMargDistr3D, getStats

# What the plots take for "this is not known for this star": a catalog of real stars carries no true [Fe/H],
# absolute magnitude or extinction, and nothing is marked for the quantities it does not carry.
NO_TRUTH = -99.0
PLOT_DIR = Path("plots")


def saveFigure(name):
    """Write the current figure to PLOT_DIR, which is created if it does not exist yet."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"{name}.png")


def starValue(star, column):
    """The value a star's row holds for a column, or NO_TRUTH when the catalog does not carry it."""
    value = star.get(column, np.nan)
    return float(value) if np.isfinite(value) else NO_TRUTH


def markTrue(ax, x0, height, style="--k"):
    """A vertical line at the true value of a 1D marginal, for a star whose true value is known."""
    if x0 > NO_TRUTH:
        ax.plot([x0, x0], [0, height], style, lw=1)


def oneImage(
    ax,
    image,
    extent,
    title="",
    showTrue=False,
    x0=NO_TRUTH,
    y0=NO_TRUTH,
    origin="upper",
    logScale=True,
    minFactor=100,
    cmap="Blues",
    markerAlpha=1.0,
):
    """One map panel: image[x, y] scaled to its maximum, with the true value marked.

    origin says which end of the extent the first row of the image belongs at, and has to follow the extent:
    "upper" when the extent runs from the largest value of y to the smallest, "lower" when it runs the other
    way. Both ways round the image is then drawn where its own y values are, as the true value is.
    """
    peak = np.max(image)
    im = np.asarray(image) / peak if peak > 0 else np.asarray(image)
    # a map that is zero everywhere has no range to put on a log scale, and LogNorm without one raises
    norm = LogNorm(im.max() / minFactor, vmax=im.max()) if logScale and peak > 0 else None
    mappable = ax.imshow(im.T, origin=origin, aspect="auto", extent=extent, cmap=cmap, norm=norm)
    ax.set_title(title)
    if showTrue:
        ax.scatter(x0, y0, s=150, c="red", alpha=markerAlpha)
        ax.scatter(x0, y0, s=40, c="yellow", alpha=markerAlpha)
    return mappable


def showMargPosteriors3D(
    x1d1,
    margp1,
    xLab1,
    yLab1,
    x1d2,
    margp2,
    xLab2,
    yLab2,
    x1d3,
    margp3,
    xLab3,
    yLab3,
    trueX1,
    trueX2,
    trueX3,
    saveFig=False,
):
    """Marginal prior (blue), likelihood (green) and posterior (red) of the three fitted parameters."""

    fig, axs = plt.subplots(1, 3, figsize=(12.7, 4))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    # plot
    panels = (
        (x1d1, margp1, xLab1, yLab1, trueX1),
        (x1d2, margp2, xLab2, yLab2, trueX2),
        (x1d3, margp3, xLab3, yLab3, trueX3),
    )
    for ax, (x1d, margp, xLab, yLab, trueX) in zip(axs, panels, strict=True):
        ax.plot(x1d, margp[2], "r", lw=3)
        ax.plot(x1d, margp[1], "g")
        ax.plot(x1d, margp[0], "b")

        ax.set(xlabel=xLab, ylabel=yLab)
        height = 1.05 * np.max([margp[0], margp[2]])
        markTrue(ax, trueX, height, style="k")
        meanX = getStats(x1d, margp[2])[0]
        ax.plot([meanX, meanX], [0, height], "--r")

    if saveFig:
        saveFigure("margPosteriors3D")
    plt.show()


def showCornerPlot3(
    postCube,
    Mr1d,
    FeH1d,
    Ar1d,
    md,
    xLab,
    yLab,
    x0=NO_TRUTH,
    y0=NO_TRUTH,
    z0=NO_TRUTH,
    logScale=False,
    cmap="Blues",
    saveFig=False,
):
    """Corner plot of a (FeH, Mr, Ar) posterior cube: 2D and 1D marginal distributions."""

    # unpack metadata
    xMin = md[0]  # FeH
    xMax = md[1]
    yMin = md[3]  # Mr
    yMax = md[4]
    zMin = Ar1d[0]  # Ar
    zMax = Ar1d[-1]

    #### make 3 marginal (summed) 2-D distributions and 3 1-D marginal distributions
    # grid steps
    dFeH = FeH1d[1] - FeH1d[0]
    dMr = Mr1d[1] - Mr1d[0]
    dAr = Ar1d[1] - Ar1d[0] if Ar1d.size > 1 else 0.01

    # 1-D marginal distributions
    margMr, margFeH, margAr = getMargDistr3D(postCube, dMr, dFeH, dAr)

    # 2-D marginal distributions
    # Mr vs. FeH
    im1 = np.sum(postCube, axis=(2))
    # Ar vs. FeH
    im2 = np.sum(postCube, axis=(1))
    # Ar vs. Mr
    im3 = np.sum(postCube, axis=(0))

    showTrue = False
    if (x0 > NO_TRUTH) & (y0 > NO_TRUTH):
        showTrue = True

    ### plot
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)
    panel = dict(showTrue=showTrue, logScale=logScale, cmap=cmap, markerAlpha=0.3)

    # row 1: marginal FeH
    axs[0, 0].plot(FeH1d, margFeH, "r", lw=3)
    markTrue(axs[0, 0], x0, 1.1 * np.max(margFeH))
    axs[0, 0].set(xlabel=xLab, ylabel=f"p({xLab})")
    axs[0, 1].set_axis_off()
    axs[0, 2].set_axis_off()

    # row 2: im1 and marginal Mr. The Mr extent runs from the faintest magnitude to the brightest, so the
    # first row of the image, which is the brightest, goes at the upper end of it.
    myExtent = [xMin, xMax, yMin, yMax]
    oneImage(axs[1, 0], im1, myExtent, x0=x0, y0=y0, origin="upper", **panel)
    axs[1, 0].set(xlabel=xLab, ylabel=yLab)
    axs[1, 1].plot(Mr1d, margMr, "r", lw=3)
    markTrue(axs[1, 1], y0, 1.1 * np.max(margMr))
    axs[1, 1].set(xlabel=yLab, ylabel=f"p({yLab})")
    axs[1, 2].set_axis_off()

    # row 3: im2, im3, and marginal Ar. A_r grows upwards here, so the first row of the image, which is the
    # smallest A_r, goes at the lower end of the extent.
    myExtent = [xMin, xMax, zMin, zMax]
    oneImage(axs[2, 0], im2, myExtent, x0=x0, y0=z0, origin="lower", **panel)
    axs[2, 0].set(xlabel=xLab, ylabel="Ar")
    myExtent = [yMax, yMin, zMin, zMax]
    oneImage(axs[2, 1], im3, myExtent, x0=y0, y0=z0, origin="lower", **panel)
    axs[2, 1].set(xlabel=yLab, ylabel="Ar")
    axs[2, 2].plot(Ar1d, margAr, "r", lw=3)
    markTrue(axs[2, 2], z0, 1.1 * np.max(margAr))
    axs[2, 2].set(xlabel="Ar", ylabel="p(Ar)")

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    if saveFig:
        saveFigure("cornerPlot3")
    plt.show()


def showQrCornerPlot(
    postCube,
    Mr1d,
    FeH1d,
    Ar1d,
    x0=NO_TRUTH,
    y0=NO_TRUTH,
    z0=NO_TRUTH,
    logScale=False,
    cmap="Blues",
    saveFig=False,
    MrTrue=None,
):
    """Posterior in the Qr = Mr + Ar vs. FeH plane and its marginal distributions; returns the Qr marginal.

    y0 and z0 are the true absolute magnitude and extinction of the star, so the true Qr is their sum.
    """

    # 2-D distribution in the Qr vs. FeH plane
    Qmap, Qr1d = getQmap(postCube, FeH1d, Mr1d, Ar1d, MrTrue)

    # 1-D marginal distribution for Qr. Summing the map over [Fe/H] leaves a density in Qr, so each marginal
    # is normalized with the step of its own axis.
    dFeH = FeH1d[1] - FeH1d[0]
    dQr = Qr1d[1] - Qr1d[0]
    margQr, margFeH = getMargDistr(Qmap, dQr, dFeH)

    # map plotting limits
    xMin = np.min(FeH1d)
    xMax = np.max(FeH1d)
    yMin = np.min(Qr1d)
    yMax = np.max(Qr1d)

    QrTrue = y0 + z0 if (y0 > NO_TRUTH) & (z0 > NO_TRUTH) else NO_TRUTH
    showTrue = False
    if (x0 > NO_TRUTH) & (QrTrue > NO_TRUTH):
        showTrue = True

    ### plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    myExtent = [xMin, xMax, yMax, yMin]
    oneImage(
        axs[0],
        Qmap,
        myExtent,
        showTrue=showTrue,
        x0=x0,
        y0=QrTrue,
        origin="upper",
        logScale=logScale,
        cmap=cmap,
        markerAlpha=0.3,
    )
    axs[0].set(xlabel="FeH", ylabel="Qr = Mr + Ar")
    axs[1].plot(Qr1d, margQr, "r", lw=3)
    markTrue(axs[1], QrTrue, 1.1 * np.max(margQr))
    axs[1].set(xlabel="Qr", ylabel="p(Qr)")
    axs[2].plot(FeH1d, margFeH, "r", lw=3)
    markTrue(axs[2], x0, 1.1 * np.max(margFeH))
    axs[2].set(xlabel="FeH", ylabel="p(FeH)")

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    if saveFig:
        saveFigure("QrCornerPlot")
    plt.show()
    return Qr1d, margQr


def show3Flat2Dmaps(
    Z1,
    Z2,
    Z3,
    md,
    xLab,
    yLab,
    x0=NO_TRUTH,
    y0=NO_TRUTH,
    logScale=False,
    minFac=1000,
    cmap="Blues",
    file_ext=None,
    saveFig=False,
):
    """Prior, likelihood and posterior maps side by side (md: locus metadata from getPlottingArgs)."""
    xMin, xMax, yMin, yMax = md[0], md[1], md[3], md[4]
    myExtent = [xMin, xMax, yMin, yMax]
    shape = (int(md[2]), int(md[5]))
    im1 = np.reshape(Z1, shape)
    im2 = np.reshape(Z2, shape)
    im3 = np.reshape(Z3, shape)

    showTrue = False
    if (x0 > NO_TRUTH) & (y0 > NO_TRUTH):
        showTrue = True

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    panel = dict(showTrue=showTrue, x0=x0, y0=y0, logScale=logScale, cmap=cmap, minFactor=minFac)
    for ax, image, title in zip(axs, (im1, im2, im3), ("Prior", "Likelihood", "Posterior"), strict=True):
        fig.colorbar(oneImage(ax, image, myExtent, title, origin="upper", **panel), ax=ax)

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()

    for ax in axs.flat:
        ax.set(xlabel=xLab, ylabel=yLab)

    if saveFig:
        saveFigure(f"bayesPanels{file_ext if file_ext is not None else ''}")
    plt.show()


def getQmap(cube, FeH1d, Mr1d, Ar1d, MrTrue=None):
    """Project a (FeH, Mr, Ar) posterior cube onto a Qr = Mr + Ar vs. FeH map.

    Qr is built from the true absolute magnitude of each grid point, as bayes.starPosterior builds it: with
    the locus parametrised by tLoc the second axis of the cube is not a magnitude, and MrTrue, the true Mr on
    the (FeH, Mr) grid from GlobalParams.getPlottingArgs, is what Qr is made of. Without it the Mr axis is
    taken to be the magnitude itself.

    Every cell of the cube is added to the Qr bin nearest its own Qr, so the map holds all of the posterior
    weight. Its step is the coarser of the Mr and the A_r step, which is as finely as the cube resolves Qr.
    """
    if MrTrue is None:
        MrTrue = np.broadcast_to(Mr1d, (np.size(FeH1d), np.size(Mr1d)))
    MrTrue = np.asarray(MrTrue, dtype=float)
    dQr = Mr1d[1] - Mr1d[0]
    if np.size(Ar1d) > 1:
        dQr = max(dQr, Ar1d[1] - Ar1d[0])
    QrMin = MrTrue.min() + np.min(Ar1d)
    nQr = int(np.rint((MrTrue.max() + np.max(Ar1d) - QrMin) / dQr)) + 1
    Qr1d = QrMin + dQr * np.arange(nQr)
    Qmap = np.zeros((np.size(FeH1d), nQr))
    for i in range(np.size(FeH1d)):
        k = np.clip(np.rint((MrTrue[i][:, None] + Ar1d - QrMin) / dQr).astype(int), 0, nQr - 1)
        Qmap[i] = np.bincount(k.ravel(), np.asarray(cube[i], dtype=float).ravel(), minlength=nQr)
    return Qmap, Qr1d


def plotStar(
    star,
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
    MrTrue=None,
):
    """All diagnostic plots for one star; returns the mean and standard deviation of its Qr posterior.

    The true values are read from the columns the catalog carries for the fitted quantities: xLabel and yLabel
    for the two axes of the maps, "Mr" for the absolute magnitude Qr is measured from, and "Ar". A catalog of
    real stars carries none of them, and then nothing is marked as true.
    """
    trueX = starValue(star, xLabel)
    trueY = starValue(star, yLabel)
    trueMr = starValue(star, "Mr")
    trueAr = starValue(star, "Ar")
    indA = np.argmax(margpostAr[2])
    show3Flat2Dmaps(
        priorCube[:, :, indA],
        likeCube[:, :, indA],
        postCube[:, :, indA],
        mdLocus,
        xLabel,
        yLabel,
        logScale=True,
        x0=trueX,
        y0=trueY,
    )
    showMargPosteriors3D(
        Mr1d,
        margpostMr,
        yLabel,
        f"p({yLabel})",
        FeH1d,
        margpostFeH,
        xLabel,
        f"p({xLabel})",
        Ar1d,
        margpostAr,
        "Ar",
        "p(Ar)",
        trueY,
        trueX,
        trueAr,
    )
    # these show marginal 2D and 1D distributions (aka "corner plot")
    showCornerPlot3(
        postCube,
        Mr1d,
        FeH1d,
        Ar1d,
        mdLocus,
        xLabel,
        yLabel,
        logScale=True,
        x0=trueX,
        y0=trueY,
        z0=trueAr,
    )
    # Qr vs. FeH posterior and marginal 1D distributions for Qr and FeH
    Qr1d, margpostQr = showQrCornerPlot(
        postCube, Mr1d, FeH1d, Ar1d, x0=trueX, y0=trueMr, z0=trueAr, logScale=True, MrTrue=MrTrue
    )
    QrEst, QrEstUnc = getStats(Qr1d, margpostQr)
    return QrEst, QrEstUnc


def plotStars(starsData, bayesResults, *plottingArgs):
    """Diagnostic plots for the stars of makeBayesPosteriors3d; returns (QrEst, QrEstUnc) for each star."""

    def getValueForStar(statDict, index):
        return {key: value[index] for key, value in statDict.items()}

    # Drop the _healpix_29 index
    stars = starsData.reset_index(drop=True)

    if len(stars) != len(bayesResults):
        raise ValueError("Stars data and results have different size")

    # each result holds arrays for a single star
    estimates = []
    for i, result in enumerate(bayesResults):
        QrEst, QrEstUnc = plotStar(
            stars.iloc[i],
            getValueForStar(result.margpostAr, 0),
            getValueForStar(result.margpostMr, 0),
            getValueForStar(result.margpostFeH, 0),
            result.likeCube[0],
            result.priorCube[0],
            result.postCube[0],
            *plottingArgs,
        )
        estimates.append((QrEst, QrEstUnc))
    return estimates


def show2Dmap(Xgrid, Ygrid, Z, metadata, xLabel, yLabel, logScale=False):
    """Show a prior map as written by priors.get2Dmap."""
    xMin, xMax, yMin, yMax = metadata[0], metadata[1], metadata[3], metadata[4]
    plt.subplots(1, 1, figsize=(6, 4.5))
    # the floor of the log scale follows the map, which a map of a sparsely populated sky pixel can leave
    # far below 0.001; a fixed floor above the maximum leaves the whole map in one color
    peak = np.max(Z)
    norm = LogNorm(min(0.001, peak / 1000), vmax=peak) if logScale and peak > 0 else None
    plt.imshow(
        Z.reshape(Xgrid.shape),
        origin="lower",
        aspect="auto",
        extent=[xMin, xMax, yMin, yMax],
        cmap="Blues",
        norm=norm,
    )
    plt.colorbar().set_label("density on log scale" if logScale else "density on lin scale")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()
