import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from photod.column_map.locus_data import map as ldc


def getBayesConstants():
    """Grid of the prior maps and the r-band magnitude bins they are computed for."""
    return {
        # [Fe/H] and Mr (or tLoc) grid of the maps
        "FeHmin": -2.5,
        "FeHmax": 1.0,
        "FeHNpts": 36,
        "MrFaint": 17.0,
        "MrBright": -2.0,
        "MrNpts": 381,
        # maps for r = rmagMin ... rmagMax, each from the stars within +-rmagBinWidth
        "rmagMin": 14,
        "rmagMax": 27,
        "rmagNsteps": 27,
        "rmagBinWidth": 0.5,
    }


def initializePriorGrid(mapPartition, globalParams):
    """Prior maps of one sky pixel interpolated onto the locus grid, one per r bin of getBayesConstants().

    mapPartition has one row per map with columns rmag, kde, xGrid and yGrid (float64 arrays stored as bytes).
    Each r bin uses the map with the nearest rmag.
    """
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    rmag = mapPartition["rmag"].to_numpy()
    points = (
        np.asarray(globalParams.locusData["FeH"]),
        np.asarray(globalParams.locusData[globalParams.MrColumn]),
    )
    priorGrid = {}
    for rind, r in enumerate(rGrid):
        row = mapPartition.iloc[int(np.argmin(np.abs(rmag - r)))]
        X = np.frombuffer(row["xGrid"], dtype=np.float64)
        nX = np.unique(X).size
        X = X.reshape(-1, nX)
        Y = np.frombuffer(row["yGrid"], dtype=np.float64).reshape(X.shape)
        Z = np.frombuffer(row["kde"], dtype=np.float64).reshape(X.shape)
        priorGrid[rind] = interpolateMap(X[0], Y[:, 0], Z, *points)
    return priorGrid


def priorGridFromMaps(maps, rmag, xGrid, yGrid, globalParams):
    """The same, from the maps of one sky pixel held as an array rather than as a table of rows.

    maps is one density per entry of rmag on the grid xGrid by yGrid, and each r bin of getBayesConstants()
    takes the map whose rmag is nearest to it.
    """
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    rmag = np.asarray(rmag, dtype=float)
    points = (
        np.asarray(globalParams.locusData["FeH"]),
        np.asarray(globalParams.locusData[globalParams.MrColumn]),
    )
    return {
        rind: interpolateMap(
            xGrid, yGrid, np.asarray(maps[int(np.argmin(np.abs(rmag - r)))], dtype=float), *points
        )
        for rind, r in enumerate(rGrid)
    }


def interpolateMap(xgrid, ygrid, Z, x, y):
    """Bilinear interpolation of a map Z[y, x] on a regular grid, zero outside of it."""
    if ygrid[0] > ygrid[-1]:
        ygrid, Z = ygrid[::-1], Z[::-1]
    interp = RegularGridInterpolator((ygrid, xgrid), Z, method="linear", bounds_error=False, fill_value=0.0)
    return interp(np.column_stack([y, x]))


def readPriors(rootname, locusData, yColumn=ldc.abs_mag_r):
    """Prior maps written by dumpPriorMaps_testing (rootname-NN.npz), interpolated onto the locus grid."""
    bc = getBayesConstants()
    priorGrid = {}
    for rind in range(bc["rmagNsteps"]):
        m = np.load(f"{rootname}-{rind:02d}.npz")
        if m["metadata"][13] != bc["rmagBinWidth"]:
            raise ValueError(
                f"map {rind} was made with rmagBinWidth={m['metadata'][13]}, not {bc['rmagBinWidth']}"
            )
        X, Y = m["xGrid"], m["yGrid"]
        priorGrid[rind] = interpolateMap(
            X[0], Y[:, 0], m["kde"], locusData[ldc.metallicity], locusData[yColumn]
        )
    return priorGrid


def getPriorMapIndex(rObs):
    """Index of the prior map (r bin) nearest to each observed r magnitude.

    A star without a magnitude is given the first map so that the arrays keep their shape; it cannot be
    fitted, and makeBayesEstimates3d empties its row and flags it rather than reporting that map's answer.
    """
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    index = np.interp(rObs, rGrid, np.arange(bc["rmagNsteps"])) + 0.5
    return np.where(np.isfinite(index), index, 0.0).astype(int)


def get2Dmap(sample, labels, metadata, bandwidthFactor=1.0):
    """Kernel density estimate of the columns labels[0], labels[1] of sample on the grid given by metadata.

    Gaussian kernel with Scott's bandwidth along each axis, multiplied by bandwidthFactor. The density is
    evaluated by linear binning of the sample onto the grid followed by Gaussian smoothing, which is accurate
    when the grid step is small compared with the bandwidth and takes a fraction of a second for millions of
    stars. Returns the grid (X, Y) and the density, flattened in the order of X.ravel().
    """
    x = np.asarray(sample[labels[0]], dtype=float)
    y = np.asarray(sample[labels[1]], dtype=float)
    xgrid = np.linspace(metadata[0], metadata[1], int(metadata[2]))
    ygrid = np.linspace(metadata[3], metadata[4], int(metadata[5]))
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    ascending = ygrid[0] < ygrid[-1]
    ya = ygrid if ascending else ygrid[::-1]
    dx, dy = xgrid[1] - xgrid[0], ya[1] - ya[0]
    fx, fy = (x - xgrid[0]) / dx, (y - ya[0]) / dy
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    # a star is spread over the four grid points around it, so it needs a point on each side; one sitting
    # exactly on the last point of an axis has none beyond it and is left out. The grids of
    # getBayesConstants() run wider than the locus in both parameters, so nothing reaches those edges.
    inside = (ix >= 0) & (ix < xgrid.size - 1) & (iy >= 0) & (iy < ya.size - 1)
    ix, iy, wx, wy = ix[inside], iy[inside], (fx - np.floor(fx))[inside], (fy - np.floor(fy))[inside]
    counts = np.zeros((ya.size, xgrid.size))
    for ddy, ddx, weight in (
        (0, 0, (1 - wy) * (1 - wx)),
        (0, 1, (1 - wy) * wx),
        (1, 0, wy * (1 - wx)),
        (1, 1, wy * wx),
    ):
        np.add.at(counts, (iy + ddy, ix + ddx), weight)

    scott = x.size ** (-1.0 / 6.0) * bandwidthFactor
    sigma = (scott * np.std(y, ddof=1) / dy, scott * np.std(x, ddof=1) / dx)
    Z = gaussian_filter(counts, sigma=sigma, mode="constant") / (x.size * dx * dy)
    if not ascending:
        Z = Z[::-1]
    return Xgrid, Ygrid, Z.ravel()


def dumpPriorMaps_testing(
    sample,
    fileRootname,
    pix,
    show2Dmap=False,
    verbose=True,
    NrowMax=None,
    labels=("FeH", "Mr", "rmag"),
    bandwidthFactor=1.0,
    seed=0,
):
    """Prior maps of labels[0] vs labels[1] for the r bins of getBayesConstants(), one file per bin.

    sample is the TRILEGAL catalog of one sky pixel, with the columns in labels and glon, glat, Av, label,
    logage and comp. Each map is written to fileRootname-NN.npz; bins with fewer than 3 stars are skipped.
    A table of sample statistics per bin is written to fileRootname-SummaryStats.txt. NrowMax, if given, caps
    the number of stars used for each map, drawn with seed so that a rebuild gives the same maps.
    """
    bc = getBayesConstants()
    metadata = np.array(
        [
            bc["FeHmin"],
            bc["FeHmax"],
            bc["FeHNpts"],
            bc["MrFaint"],
            bc["MrBright"],
            bc["MrNpts"],
            np.mean(sample["glon"]),
            np.mean(sample["glat"]),
            pix.order,
            pix.pixel,
        ]
    )
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    mag = sample[labels[2]]
    rows = []
    for rind, r in enumerate(rGrid):
        rMin, rMax = r - bc["rmagBinWidth"], r + bc["rmagBinWidth"]
        tS = sample[(mag > rMin) & (mag < rMax)]
        if verbose:
            print(f"r = {rMin:.1f} to {rMax:.1f}: {len(tS)} of {len(sample)} stars")
        if len(tS) < 3:
            continue
        tSmap = tS.sample(n=NrowMax, random_state=seed) if NrowMax and len(tS) > NrowMax else tS
        xGrid, yGrid, Z = get2Dmap(tSmap, labels, metadata, bandwidthFactor)
        if show2Dmap:
            from photod.plotting import show2Dmap as plotMap

            plotMap(xGrid, yGrid, Z, metadata, labels[0], labels[1], logScale=True)
        mdExt = np.concatenate(
            (metadata, [bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"], bc["rmagBinWidth"], np.size(tS), r])
        )
        np.savez(
            f"{fileRootname}-{rind:02d}.npz",
            xGrid=xGrid,
            yGrid=yGrid,
            kde=Z.reshape(xGrid.shape),
            metadata=mdExt,
            labels=list(labels),
        )
        rows.append((rMin, rMax, tS))

    with open(fileRootname + "-SummaryStats.txt", "w") as f:
        f.write(
            " rMin    rMax   Ntotal     Amin   Amed    Amax     pMS      ppMS     pAGB      pWD        "
            "pA1       pA2      pA3      pA4       ptnD     ptkD       pH        pB        pMC \n"
        )
        for rMin, rMax, tS in rows:
            label, logage, comp = tS["label"], tS["logage"], tS["comp"]
            phases = np.array(
                [
                    (label == 1).sum(),
                    ((label > 1) & (label < 7)).sum(),
                    ((label > 6) & (label < 9)).sum(),
                    (label == 9).sum(),
                ],
                dtype=float,
            )
            phases = phases / phases.sum() if phases.sum() > 0 else np.full(4, -1.0)
            ages = [np.mean(logage < a) for a in (7, 8, 9, 10)]
            components = [np.mean(comp == k) for k in range(1, 6)]
            f.write(
                f"{rMin:5.1f}  {rMax:5.1f}  {len(tS):10.0f}  "
                f"{np.min(tS['Av']):6.2f} {np.median(tS['Av']):6.2f} {np.max(tS['Av']):6.2f} "
                + "".join(f"{v:8.3e} " for v in [*phases, *ages, *components])
                + "\n"
            )
