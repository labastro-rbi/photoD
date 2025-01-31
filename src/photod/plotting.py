import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from photod.stats import getMargDistr, getMargDistr3D, getStats


def showMargPosteriors3D(
    x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, x1d3, margp3, xLab3, yLab3, trueX1, trueX2, trueX3
):

    fig, axs = plt.subplots(1, 3, figsize=(12.7, 4))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    # plot
    axs[0].plot(x1d1, margp1[2], "r", lw=3)
    axs[0].plot(x1d1, margp1[1], "g")
    axs[0].plot(x1d1, margp1[0], "b")

    axs[0].set(xlabel=xLab1, ylabel=yLab1)
    axs[0].plot([trueX1, trueX1], [0, 1.05 * np.max([margp1[0], margp1[2]])], "k", lw=1)
    meanX1, sigX1 = getStats(x1d1, margp1[2])
    axs[0].plot([meanX1, meanX1], [0, 1.05 * np.max([margp1[0], margp1[2]])], "--r")

    axs[1].plot(x1d2, margp2[2], "r", lw=3)
    axs[1].plot(x1d2, margp2[1], "g")
    axs[1].plot(x1d2, margp2[0], "b")

    axs[1].set(xlabel=xLab2, ylabel=yLab2)
    axs[1].plot([trueX2, trueX2], [0, 1.05 * np.max([margp2[0], margp2[2]])], "k", lw=1)
    meanX2, sigX2 = getStats(x1d2, margp2[2])
    axs[1].plot([meanX2, meanX2], [0, 1.05 * np.max([margp2[0], margp2[2]])], "--r")

    axs[2].plot(x1d3, margp3[2], "r", lw=3)
    axs[2].plot(x1d3, margp3[1], "g")
    axs[2].plot(x1d3, margp3[0], "b")

    axs[2].set(xlabel=xLab3, ylabel=yLab3)
    axs[2].plot([trueX3, trueX3], [0, 1.05 * np.max([margp3[0], margp3[2]])], "k", lw=1)
    meanX3, sigX3 = getStats(x1d3, margp3[2])
    axs[2].plot([meanX3, meanX3], [0, 1.05 * np.max([margp3[0], margp3[2]])], "--r")

    plt.savefig("plots/margPosteriors3D.png")
    plt.show()


def showCornerPlot3(
    postCube, Mr1d, FeH1d, Ar1d, md, xLab, yLab, x0=-99, y0=-99, z0=-99, logScale=False, cmap="Blues"
):

    def oneImage(ax, image, extent, title, showTrue, x0, y0, origin, logScale=True, cmap="Blues"):
        im = image / image.max()
        if logScale:
            cmap = ax.imshow(
                im.T,
                origin=origin,
                aspect="auto",
                extent=extent,
                cmap=cmap,
                norm=LogNorm(im.max() / 100, vmax=im.max()),
            )
        else:
            cmap = ax.imshow(im.T, origin="upper", aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        if showTrue:
            ax.scatter(x0, y0, s=150, c="red", alpha=0.3)
            ax.scatter(x0, y0, s=40, c="yellow", alpha=0.3)
        return cmap

    # unpack metadata
    xMin = md[0]  # FeH
    xMax = md[1]
    yMin = md[3]  # Mr
    yMax = md[4]
    zMin = 0  # Ar
    zMin = Ar1d[0]
    zMax = Ar1d[-1]

    #### make 3 marginal (summed) 2-D distributions and 3 1-D marginal distributions
    # grid steps
    dFeH = FeH1d[1] - FeH1d[0]
    dMr = Mr1d[1] - Mr1d[0]
    if Ar1d.size > 1:
        dAr = Ar1d[1] - Ar1d[0]
    else:
        dAr = 0.01

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
    if (x0 > -99) & (y0 > -99):
        showTrue = True

    ### plot
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    # row 1: marginal FeH
    myExtent = [xMin, xMax, yMin, yMax]
    axs[0, 0].plot(FeH1d, margFeH, "r", lw=3)
    axs[0, 0].plot([x0, x0], [0, 1.1 * np.max(margFeH)], "--k", lw=1)
    axs[0, 0].set(xlabel="FeH", ylabel="p(FeH)")
    axs[0, 1].set_axis_off()
    axs[0, 2].set_axis_off()

    # row 2: im1 and marginal Mr
    myExtent = [xMin, xMax, yMin, yMax]
    cmap = oneImage(axs[1, 0], im1, myExtent, "", showTrue, x0, y0, origin="upper", logScale=logScale)
    axs[1, 0].set(xlabel="FeH", ylabel="Mr")
    axs[1, 1].plot(Mr1d, margMr, "r", lw=3)
    axs[1, 1].plot([y0, y0], [0, 1.1 * np.max(margMr)], "--k", lw=1)
    axs[1, 1].set(xlabel="Mr", ylabel="p(Mr)")
    axs[1, 2].set_axis_off()

    # row 3: im2, im3, and marginal Ar
    myExtent = [xMin, xMax, zMin, zMax]
    cmap = oneImage(axs[2, 0], im2, myExtent, "", showTrue, x0, z0, origin="lower", logScale=logScale)
    axs[2, 0].set(xlabel="FeH", ylabel="Ar")
    myExtent = [yMax, yMin, zMin, zMax]
    cmap = oneImage(axs[2, 1], im3, myExtent, "", showTrue, y0, z0, origin="lower", logScale=logScale)
    axs[2, 1].set(xlabel="Mr", ylabel="Ar")
    axs[2, 2].plot(Ar1d, margAr, "r", lw=3)
    axs[2, 2].plot([z0, z0], [0, 1.1 * np.max(margAr)], "--k", lw=1)
    axs[2, 2].set(xlabel="Ar", ylabel="p(Ar)")

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    # cb = fig.colorbar(cmap, ax=cax)
    # if (logScale):
    # cb.set_label("density on log scale")
    # else:
    # cb.set_label("density on linear scale")

    # for ax in axs.flat:
    # ax.set(xlabel=xLab, ylabel=yLab)
    # print('pero')

    plt.savefig("plots/cornerPlot3.png")
    plt.show()


def showQrCornerPlot(postCube, Mr1d, FeH1d, Ar1d, x0=-99, y0=-99, z0=-99, logScale=False, cmap="Blues"):

    def oneImage(ax, image, extent, title, showTrue, x0, y0, origin, logScale=True, cmap="Blues"):
        im = image / image.max()
        if logScale:
            cmap = ax.imshow(
                im.T,
                origin=origin,
                aspect="auto",
                extent=extent,
                cmap=cmap,
                norm=LogNorm(im.max() / 100, vmax=im.max()),
            )
        else:
            cmap = ax.imshow(im.T, origin="upper", aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        if showTrue:
            ax.scatter(x0, y0, s=150, c="red", alpha=0.3)
            ax.scatter(x0, y0, s=40, c="yellow", alpha=0.3)
        return cmap

    # 2-D distribution in the Qr vs. FeH plane
    Qmap, Qr1d = getQmap(postCube, FeH1d, Mr1d, Ar1d)

    # 1-D marginal distribution for Qr
    dFeH = FeH1d[1] - FeH1d[0]
    dQr = Qr1d[1] - Qr1d[0]
    margQr, margFeH = getMargDistr(Qmap, dFeH, dQr)

    # map plotting limits
    xMin = np.min(FeH1d)
    xMax = np.max(FeH1d)
    yMin = np.min(Qr1d)
    yMax = np.max(Qr1d)

    showTrue = False
    if (x0 > -99) & (y0 > -99):
        showTrue = True

    ### plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    myExtent = [xMin, xMax, yMax, yMin]
    cmap = oneImage(axs[0], Qmap, myExtent, "", showTrue, x0, y0 + z0, origin="upper", logScale=logScale)
    axs[0].set(xlabel="FeH", ylabel="Qr = Mr + Ar")
    axs[1].plot(Qr1d, margQr, "r", lw=3)
    axs[1].plot([y0 + z0, y0 + z0], [0, 1.1 * np.max(margQr)], "--k", lw=1)
    axs[1].set(xlabel="Qr", ylabel="p(Qr)")
    axs[2].plot(FeH1d, margFeH, "r", lw=3)
    axs[2].plot([x0, x0], [0, 1.1 * np.max(margFeH)], "--k", lw=1)
    axs[2].set(xlabel="FeH", ylabel="p(FeH)")

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    # cb = fig.colorbar(cmap, ax=cax)
    # if (logScale):
    # cb.set_label("density on log scale")
    # else:
    # cb.set_label("density on linear scale")

    # for ax in axs.flat:
    # ax.set(xlabel=xLab, ylabel=yLab)
    # print('pero')

    plt.savefig("plots/QrCornerPlot.png")
    plt.show()
    return Qr1d, margQr


def show3Flat2Dmaps(Z1, Z2, Z3, md, xLab, yLab, x0=-99, y0=-99, logScale=False, minFac=1000, cmap="Blues"):

    # unpack metadata
    xMin = md[0]
    xMax = md[1]
    nXbin = md[2]
    yMin = md[3]
    yMax = md[4]
    nYbin = md[5]
    # set local variables and
    myExtent = [xMin, xMax, yMin, yMax]
    Xpts = nXbin.astype(int)
    Ypts = nYbin.astype(int)
    # reshape flattened input arrays to get "images"
    im1 = Z1.reshape((Xpts, Ypts))
    im2 = Z2.reshape((Xpts, Ypts))
    im3 = Z3.reshape((Xpts, Ypts))
    print("pts:", Xpts, Ypts)

    showTrue = False
    if (x0 > -99) & (y0 > -99):
        showTrue = True

    def oneImage(ax, image, extent, minFactor, title, showTrue, x0, y0, logScale=True, cmap="Blues"):
        im = image / image.max()
        ImMin = im.max() / minFactor
        if logScale:
            cmap = ax.imshow(
                im.T,
                origin="upper",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                norm=LogNorm(ImMin, vmax=im.max()),
            )
            ax.set_title(title)
        else:
            cmap = ax.imshow(im.T, origin="upper", aspect="auto", extent=extent, cmap=cmap)
            ax.set_title(title)
        if showTrue:
            ax.scatter(x0, y0, s=150, c="red")
            ax.scatter(x0, y0, s=40, c="yellow")
        return cmap

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # plot
    from matplotlib.colors import LogNorm

    cmap = oneImage(axs[0], im1, myExtent, minFac, "Prior", showTrue, x0, y0, logScale=logScale)
    fig.colorbar(cmap, ax=axs[0])
    cmap = oneImage(axs[1], im2, myExtent, minFac, "Likelihood", showTrue, x0, y0, logScale=logScale)
    fig.colorbar(cmap, ax=axs[1])
    cmap = oneImage(axs[2], im3, myExtent, minFac, "Posterior", showTrue, x0, y0, logScale=logScale)
    fig.colorbar(cmap, ax=axs[2])

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()

    for ax in axs.flat:
        ax.set(xlabel=xLab, ylabel=yLab)

    plt.savefig("plots/bayesPanels.png")
    plt.show()


def getQmap(cube, FeH1d, Mr1d, Ar1d):
    # interpolate 3D cube(FeH, Mr, Ar) onto Qr=Mr+Ar vs. FeH 2D grid
    Qmap = np.zeros((len(FeH1d), len(Qr1d)))
    # Q grid, same size as Mr1d array
    Qr1d = np.linspace(np.min(Mr1d), np.max(Ar1d) + np.max(Mr1d), np.size(Mr1d))
    # Compute possible Mr values for all (j, k) pairs
    Mr_values = Qr1d[:, None] - Ar1d
    # Find nearest index for Mr in Mr1d using searchsorted
    jk_indices = np.searchsorted(Mr1d, Mr_values) - 1
    # Ensure indices are within bounds
    valid_mask = (jk_indices >= 0) & (jk_indices < len(Mr1d))
    jk_indices = np.clip(jk_indices, 0, len(Mr1d) - 1)
    for i in range(len(FeH1d)):
        Ssum = np.zeros(len(Qr1d))
        Ssum += np.where(valid_mask, cube[i, jk_indices, np.arange(len(Ar1d))], 0).sum(axis=1)
        Qmap[i, :] = Ssum
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
):
    # for testing and illustration
    FeHStar = star["FeH"]
    MrStar = star["Mr"]
    ArStar = star["Ar"]
    indA = np.argmax(margpostAr[2])
    show3Flat2Dmaps(
        priorCube[:, :, indA],
        likeCube[:, :, indA],
        postCube[:, :, indA],
        mdLocus,
        xLabel,
        yLabel,
        logScale=True,
        x0=FeHStar,
        y0=MrStar,
    )
    showMargPosteriors3D(
        Mr1d,
        margpostMr,
        "Mr",
        "p(Mr)",
        FeH1d,
        margpostFeH,
        "FeH",
        "p(FeH)",
        Ar1d,
        margpostAr,
        "Ar",
        "p(Ar)",
        MrStar,
        FeHStar,
        ArStar,
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
        x0=FeHStar,
        y0=MrStar,
        z0=ArStar,
    )
    # Qr vs. FeH posterior and marginal 1D distributions for Qr and FeH
    Qr1d, margpostQr = showQrCornerPlot(
        postCube, Mr1d, FeH1d, Ar1d, x0=FeHStar, y0=MrStar, z0=ArStar, logScale=True
    )
    QrEst, QrEstUnc = getStats(Qr1d, margpostQr)
    return QrEst, QrEstUnc


def plotStars(starsData, bayesResults, *plottingArgs):
    """Create the plots for the specified stars."""

    def getValueForStar(statDict, index):
        return {key: value[index] for key, value in statDict.items()}

    # Drop the _healpix_29 index
    stars = starsData.reset_index(drop=True)

    if len(stars) != len(bayesResults):
        raise ValueError("Stars data and results have different size")

    # Iterate over each star in the results. These results are arrays
    # of an element each because they were packed with JAX, and that is
    # why we need to get the first elements of these arrays.
    for i, result in enumerate(bayesResults):
        print(f"Plotting star {i}...")
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
        print(QrEst, QrEstUnc)
