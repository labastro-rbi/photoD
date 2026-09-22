"""Stellar locus tables and the 3D (FeH, Mr or tLoc, Ar) color model built from them."""

from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.spatial import KDTree

DEFAULT_LOCUS_FILE = Path(__file__).resolve().parents[2] / "data" / "MSandRGBcolors_v1.3.txt"

# For LSSTlocus_10Gyr_fix.txt: which TRILEGAL evolutionary labels (0 PMS, 1 MS, 2 SGB, 3 RGB, 4-6 CHeB,
# 7 EAGB, 8 TPAGB, 9 PAGB/WD) belong to each monotonic segment of Mr(tLoc) below the turn-off, per [Fe/H] row.
# Segments are numbered from the smallest tLoc.
_ALL_LABELS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
_EVOLVED = {3, 4, 5, 6, 7, 9}
SEGMENT_LABEL_MAP = {
    **{round(feh, 1): {0: _ALL_LABELS} for feh in np.arange(-2.5, -0.75, 0.1)},
    -0.7: {2: {0, 1, 2}, 1: _EVOLVED, 0: {8}},
    -0.6: {2: {0, 1, 2}, 1: _EVOLVED, 0: {8}},
    -0.5: {2: {0, 1, 2}, 1: _EVOLVED, 0: {8}},
    -0.4: {3: {0, 1}, 2: {2}, 1: _EVOLVED, 0: {8}},
    -0.3: {3: {0, 1}, 2: {2}, 1: _EVOLVED, 0: {8}},
    -0.2: {4: {0, 1}, 3: {2}, 2: _EVOLVED, 1: {8}},
    -0.1: {3: {0, 1, 2}, 2: _EVOLVED, 1: {8}},
    0.0: {4: {0, 1}, 3: {2}, 2: _EVOLVED, 1: {8}},
    0.1: {4: {0, 1}, 3: {2}, 2: _EVOLVED, 1: {8}},
    0.2: {3: {0, 1}, 2: {2} | _EVOLVED, 1: {8}},
    0.3: {5: {0, 1}, 4: {2}, 3: _EVOLVED, 2: {8}},
    0.4: {5: {0, 1}, 4: {2}, 3: _EVOLVED, 2: {8}},
    0.5: {5: {0, 1}, 4: {2}, 3: _EVOLVED, 2: {8}},
}


def LSSTsimsLocus(fixForStripe82=False, datafile=None, colnames=("Mr", "FeH", "ug", "gr", "ri", "iz", "zy")):
    """Read a stellar locus table: colors on a regular grid of [Fe/H] and Mr (or tLoc).

    Parameters
    ----------
    fixForStripe82 : bool
        Apply the empirical u-g and i-z corrections that bring the SDSS-based locus MSandRGBcolors_v1.3 into
        agreement with the SDSS Stripe 82 standard-star catalog (v4.2). Only meant for that locus: loci
        calibrated to LSST, such as LSSTlocus_10Gyr_fix.txt, must be read without it.
    datafile : str or Path, optional
        Locus table; by default data/MSandRGBcolors_v1.3.txt of this repository.
    colnames : sequence of str
        Column names, in the order of the file columns.
    """
    path = Path(datafile or DEFAULT_LOCUS_FILE)
    if not path.exists():
        # the tables live in data/ of the repository rather than inside the package, so the default only
        # resolves in a checkout; from an installed copy the table has to be named
        raise FileNotFoundError(
            f"no locus table at {path}; pass datafile=, the tables are in data/ of the photoD repository"
        )
    locus = Table.read(path, format="ascii", names=list(colnames))
    locus["gi"] = locus["gr"] + locus["ri"]
    if fixForStripe82:
        ugFix = locus["ug"] + 0.02 * (2 + locus["FeH"]) * locus["gi"]
        ugMax = 2.53 + 0.13 * (1 + locus["FeH"])
        locus["ug"] = np.where(locus["gi"] > 1.8, ugMax, ugFix)
        ri = locus["ri"]
        locus["iz"] += 0.08 * ri - 0.09 * ri**2 + 0.008 * ri**5 + 0.01 * (2.5 + locus["FeH"])
    return locus


def subsampleLocusData(locusData, kMr, kFeH, xLabel="FeH", yLabel="Mr", verbose=False):
    """Keep every kFeH-th [Fe/H] and every kMr-th Mr (or tLoc) value of a locus on a regular grid.

    The last value of each axis is kept whatever the step leaves over. Dropping it takes the end off the
    model: with the DP2 locus and every twentieth tLoc that is the reddest dwarfs, and with every third
    [Fe/H] the metal-rich end, neither of which the fit could then place a star on.
    """
    nFeH = np.unique(locusData[xLabel]).size
    nMr = np.unique(locusData[yLabel]).size
    keep = lambda n, k: np.unique(np.append(np.arange(0, n, k), n - 1))  # noqa: E731
    feHrows, mrRows = keep(nFeH, kFeH), keep(nMr, kMr)
    rows = (feHrows[:, None] * nMr + mrRows[None, :]).ravel()
    if verbose:
        print(
            f"subsampled locus grid from {nFeH} x {nMr} to {feHrows.size} x {mrRows.size} "
            f"([Fe/H] x {yLabel})"
        )
    return locusData[rows]


def get3DmodelList(locusData, fitColors, agressive=False, DSED=False, xLabel="FeH", yLabel="Mr", ArFixed=0.2):
    """3D color models for the standard A_r grids ("ArSmall", "ArMedium", "ArLarge" and "ArFixed").

    DSED is kept so that older calls still work; it is not needed any more because the colors are reddened by
    name for any column order of the locus table.
    """
    if agressive:
        grids = {
            "ArSmall": np.linspace(0, 0.5, 101),
            "ArMedium": np.linspace(0, 2.0, 201),
            "ArLarge": np.linspace(0, 5.0, 251),
        }
    else:
        grids = {
            "ArSmall": np.linspace(0, 0.3, 31),
            "ArMedium": np.linspace(0, 0.8, 81),
            "ArLarge": np.linspace(0, 2.5, 126),
        }
    grids["ArFixed"] = np.array([ArFixed])
    models = make3DlocusList(locusData, fitColors, list(grids.values()), xLabel=xLabel, yLabel=yLabel)
    return grids, dict(zip(grids, models, strict=True))


def make3DlocusList(locusData, fitColors, ArGridList, DSED=False, xLabel="FeH", yLabel="Mr"):
    """For each A_r grid, the locus as a (FeH, Mr, Ar) structured array with the fitted colors reddened.

    DSED is not used (see get3DmodelList).
    """
    C = extcoeff()
    nFeH = np.unique(locusData[xLabel]).size
    nMr = np.unique(locusData[yLabel]).size
    table = locusData.copy(copy_data=False)
    table["Ar"] = np.zeros(len(table))
    locus2D = np.array(table).reshape(nFeH, nMr)

    locus3DList = []
    for ArGrid in ArGridList:
        ArGrid = np.asarray(ArGrid, dtype=float)
        locus3D = np.repeat(locus2D[:, :, np.newaxis], ArGrid.size, axis=2)
        for color in fitColors:
            locus3D[color] = locus3D[color] + ArGrid * (C[color[0]] - C[color[1]])
        locus3D["Ar"] = np.broadcast_to(ArGrid, locus3D.shape)
        locus3DList.append(locus3D)
    return locus3DList


def locusPlateau(locusData, xLabel="FeH", yLabel="Mr"):
    """Flag grid points that only repeat the next point along yLabel at the same [Fe/H].

    To fill a rectangular grid, isochrones that end before the edge of the tLoc range are padded with copies
    of their last point. Such copies are not separate models and must not add posterior weight.
    """
    nFeH = np.unique(locusData[xLabel]).size
    nMr = np.unique(locusData[yLabel]).size
    names = [n for n in locusData.colnames if n not in (xLabel, yLabel)]
    values = np.stack([np.asarray(locusData[n], dtype=float) for n in names], axis=-1).reshape(nFeH, nMr, -1)
    plateau = np.zeros((nFeH, nMr), dtype=bool)
    plateau[:, :-1] = np.all(values[:, :-1] == values[:, 1:], axis=-1)
    return plateau.ravel()


def extcoeff():
    """Extinction A_band / A_r (Berry et al. 2012 for ugriz, Cardelli et al. 1989 for y)."""
    return {"u": 1.810, "g": 1.400, "r": 1.000, "i": 0.759, "z": 0.561, "y": 0.484}


def getColorsFromMrFeHDSED(L, Lvalues, colors=("ug", "gr", "ri", "iz"), Mr_label="Mr"):
    """Colors of the locus point nearest to each star in (Mr or tLoc, [Fe/H]); used to simulate catalogs."""
    tree = KDTree(np.column_stack([L[Mr_label], L["FeH"]]))
    nearest = tree.query(np.column_stack([Lvalues[Mr_label], Lvalues["FeH"]]))[1]
    Lvalues[f"{Mr_label}Assigned"] = np.asarray(L[Mr_label])[nearest]
    for c in colors:
        Lvalues[c] = np.asarray(L[c])[nearest]
    return Lvalues


LSST_M5 = {
    "coadd": {"u": 25.73, "g": 26.86, "r": 26.88, "i": 26.34, "z": 25.63, "y": 24.87},
    "single": {"u": 23.50, "g": 24.44, "r": 23.98, "i": 23.41, "z": 22.77, "y": 22.01},
}
LSST_GAMMA = {"u": 0.038, "g": 0.039, "r": 0.039, "i": 0.039, "z": 0.039, "y": 0.039}


def getLSSTm5(data, depth="coadd", magVersion=False, suffix=""):
    """LSST photometric errors for the magnitudes in data (Ivezic et al. 2019, with a 0.005 mag floor).

    data holds the magnitudes under the band names ("u", ...) or, with magVersion, under "umag" + suffix etc.
    """
    m5 = LSST_M5["coadd" if depth == "coadd" else "single"]
    errors = {}
    for b in "ugrizy":
        x = 10 ** (0.4 * (data[b + "mag" + suffix if magVersion else b] - m5[b]))
        errors[b] = np.sqrt(0.005**2 + (0.04 - LSST_GAMMA[b]) * x + LSST_GAMMA[b] * x**2)
    return errors


def getLSSTm5err(mags, depth="coadd"):
    """Same as getLSSTm5, by interpolation in a 0.01 mag table."""
    magGrid = np.linspace(10, 30, 2001)
    errGrid = getLSSTm5({b: magGrid for b in "ugrizy"}, depth)
    return {b: np.interp(mags[b], magGrid, errGrid[b]) for b in "ugrizy"}


def splitMonotonicSegments(tLocVals, MrTrueVals, minSegmentLen=4):
    """Split Mr(tLoc) at one [Fe/H] into monotonic runs, as (start, end) index pairs ordered by tLoc.

    Runs shorter than minSegmentLen points (numerical noise) are merged into the preceding run.
    """
    signs = np.sign(np.diff(MrTrueVals))
    signs[signs == 0] = signs[signs != 0][0] if np.any(signs != 0) else 1
    breaks = [0, *(i for i in range(1, signs.size) if signs[i] != signs[i - 1]), len(tLocVals) - 1]
    merged = []
    for start, end in zip(breaks[:-1], breaks[1:], strict=True):
        if merged and end - start < minSegmentLen:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def buildSegmentData(globalParams, segmentLabelMap=None, turnoffTLoc=4.0):
    """Per [Fe/H] row: Mr range and interpolation table of each monotonic segment of Mr(tLoc) below the
    turn-off, and which TRILEGAL labels may be placed on it. Computed once and passed to assignTLocPartition.
    """
    segmentLabelMap = SEGMENT_LABEL_MAP if segmentLabelMap is None else segmentLabelMap
    tLoc1d = globalParams.Mr1d
    below = tLoc1d <= turnoffTLoc
    segmentData = {}
    for i, feh in enumerate(globalParams.FeH1d):
        MrRow = np.asarray(globalParams.MrTrueTable[i])[below]
        segments = splitMonotonicSegments(tLoc1d[below], MrRow)
        labels = segmentLabelMap.get(round(float(feh), 1), {})
        ranges = np.zeros((len(segments), 2))
        labelToSeg = np.zeros((10, len(segments)), dtype=bool)
        interpMr, interpTLoc = [], []
        for k, (s, e) in enumerate(segments):
            order = np.argsort(MrRow[s : e + 1])
            interpMr.append(MrRow[s : e + 1][order])
            interpTLoc.append(tLoc1d[below][s : e + 1][order])
            ranges[k] = interpMr[-1][0], interpMr[-1][-1]
            labelToSeg[list(labels.get(k, ())), k] = True
        segmentData[i] = {
            "ranges": ranges,
            "interpMr": interpMr,
            "interpTLoc": interpTLoc,
            "labelToSeg": labelToSeg,
        }
    return segmentData


def assignTLocPartition(
    df,
    segmentData,
    FeH1d,
    turnoffTLoc=4.0,
    starFeHCol="FeH",
    starMrCol="Mr",
    starLabelCol="label",
    newCol="tLoc",
):
    """tLoc for each star of a TRILEGAL catalog (partition), from its Mr, [Fe/H] and evolutionary label.

    Above the turn-off tLoc equals Mr. Below it, Mr(tLoc) is not monotonic; a star is placed on the segment
    whose Mr range contains its Mr, and if several do, on the one that matches its label. Stars that cannot be
    placed get NaN. The input is not modified.
    """
    starFeH = df[starFeHCol].to_numpy()
    starMr = df[starMrCol].to_numpy()
    starLabel = np.clip(df[starLabelCol].to_numpy().astype(int), 0, 9)
    tLoc = np.where(starMr > turnoffTLoc, starMr, np.nan)

    idx = np.clip(np.searchsorted(FeH1d, starFeH), 1, len(FeH1d) - 1)
    nearestFeH = np.where(np.abs(starFeH - FeH1d[idx - 1]) <= np.abs(starFeH - FeH1d[idx]), idx - 1, idx)
    below = starMr <= turnoffTLoc
    for i in np.unique(nearestFeH[below]):
        data = segmentData[i]
        stars = np.where(below & (nearestFeH == i))[0]
        if data["ranges"].size == 0:
            continue
        mr = starMr[stars]
        inRange = (mr[:, None] >= data["ranges"][None, :, 0]) & (mr[:, None] <= data["ranges"][None, :, 1])
        candidates = np.where(
            inRange.sum(axis=1)[:, None] > 1, inRange & data["labelToSeg"][starLabel[stars]], inRange
        )
        segment = np.where(candidates.any(axis=1), np.argmax(candidates, axis=1), -1)
        for k in np.unique(segment[segment >= 0]):
            sel = segment == k
            tLoc[stars[sel]] = np.interp(mr[sel], data["interpMr"][k], data["interpTLoc"][k])

    out = df.copy()
    out[newCol] = tLoc
    return out


def assignTLocFromLabel(
    trilegalCatalog,
    globalParams,
    segmentLabelMap=None,
    turnoffTLoc=4.0,
    starFeHCol="FeH",
    starMrCol="Mr",
    starLabelCol="label",
    newCol="tLoc",
):
    """assignTLocPartition for a whole catalog, building the segment data from globalParams.

    The tLoc column is added to trilegalCatalog, which is also returned.
    """
    segmentData = buildSegmentData(globalParams, segmentLabelMap, turnoffTLoc)
    withTLoc = assignTLocPartition(
        trilegalCatalog,
        segmentData,
        globalParams.FeH1d,
        turnoffTLoc,
        starFeHCol,
        starMrCol,
        starLabelCol,
        newCol,
    )
    trilegalCatalog[newCol] = withTLoc[newCol].to_numpy()
    return trilegalCatalog
