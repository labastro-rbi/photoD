"""TRILEGAL prior maps for a photoD run: p([Fe/H], tLoc | r) on a HEALPix grid, in one file.

  python scripts/make_priors.py --trilegal /path/to/TRILEGAL_cluster_v08 --out priors_dp2.npz \\
                                --footprint /path/to/rubin_dp2/object_collection/object_lc/skymap.6.fits

One map per r bin of getBayesConstants() and per HEALPix pixel of the given order, stored as an array with an
index from pixel to row, so that the run looks a sightline up rather than joining two catalogs. The model
stars of a pixel are read from a cone of the same area at its centre and placed on the tLoc axis of the locus
the fit will use, which is what makes the maps and the fit comparable.

Only pixels of the footprint are built. The footprint comes from the sky map of the object catalog, either a
sparse table of PIXEL and VALUE or a full HEALPix array; without one the whole sky is built. A pixel of the
footprint that ends up with no maps is a hole: the run keeps the stars of it, but with no estimate and bit 64
of fit_flags set. So the build reports every such pixel and fails unless --allow-missing says a partial file
is wanted.

The maps of a stellar locus leave out the model stars of TRILEGAL label 9, the white dwarfs and post-AGB
stars, while photod.priors.dumpPriorMaps_testing keeps every label. The difference is deliberate: a white
dwarf has no place on such a locus, so assigning it a tLoc puts it wherever the nearest segment of the locus
happens to be, which at faint r is enough of them to leave a ridge in the prior that no fitted star belongs
on. The in-package function is kept as it is because the published maps were made with it.

A locus with white dwarf tracks (photod.locus.WD_LOCUS_FILE, scripts/make_qso_locus.py) gives them a place of
their own. Its maps are built as populations: the stars, each white dwarf track and, on a locus with a QSO
track, the QSOs, each a shape of unit mass, with the share of every population per r bin written beside the
maps (popShare, popSegment), which the fit holds each population to. TRILEGAL's white dwarfs are all
hydrogen ones, and they are split between the tracks by the helium fraction of HELIUM_FRACTION. On a locus
whose white dwarf rows are a mass axis (scripts/make_wd_mass_locus.py), --wd-mass-grid also writes how the
prior on mass spreads each tLoc of a track over those rows (wdMassWeight), on the locus grid itself, since the
compact maps keep only every other [Fe/H] point. --qso-counts adds the QSOs: the share of the literature's QSO
counts among QSOs and model stars in each r bin, which takes the same share off the other populations.

The DP2 white dwarf and QSO maps were built with

  python scripts/make_priors.py --trilegal /path/to/TRILEGAL --footprint /path/to/dp2_inclusive/skymap.fits \\
      --order 5 --processes 8 --locus data/LSSTlocus_10Gyr_WDmass_QSO.txt \\
      --wd-mass-grid data/WD_massgrid_QSO.json --qso-counts --dust-map /path/to/sfddata --out priors_wd.npz

The population shares read every model star of a cone four times the radius of the maps, which in the plane of
the full TRILEGAL is about 47 GB per process.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import cdshealpix
import lsdb
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

import photod.locus as lt
from photod.priors import get2Dmap, getBayesConstants

LOCUS = lt.RUN_LOCUS_FILE  # the maps are indexed by its tLoc, so it must be the locus the run fits
# The fraction of white dwarfs with helium atmospheres, by Mr (bin edges, value per bin). TRILEGAL models only
# hydrogen ones, so its white dwarfs are the total and the split comes from outside: the 100 pc SDSS sample
# (Kilic et al. 2025, Table 4) for 5.5-22.5 kK and DESI DR1 (Kilic et al. 2026, Table 2) above, converted to
# Mr with the Montreal 0.6 Msun tables. The 9.5-10.5 bin, the DB gap, is raised from 0.044 to 0.07, where
# DESI and the Gaia XP classifications put it.
HELIUM_MR_EDGES = np.array([8.5, 9.5, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5])
HELIUM_FRACTION = np.array([0.137, 0.07, 0.119, 0.098, 0.146, 0.218, 0.264, 0.284, 0.298, 0.298])
COLORS = ("ug", "gr", "ri", "iz", "zy")
MODEL_COLUMNS = ["ra", "dec", "glon", "glat", "DM", "Av", "rmag", "FeH", "Mr", "label"]
MIN_STARS = 2000
MAX_REPORTED = 20  # pixels a report lists one by one before it only counts them
WORKER = {}


PRIOR_DECADES = 8.0  # how far below its own peak a compact map is kept

# QSO counts per 10,000 deg2 in bins of dereddened r (0.5 mag, centred on QSO_R) and redshift (1, centred on
# QSO_Z): Palanque-Delabrouille et al. 2016 (A&A 587, A41), Table 7, the PLE and the PLE+LEDE models, of which
# the prior takes the mean. Counts brighter or fainter than the table follow the log slope of its first or
# last magnitude in each redshift bin.
QSO_R = np.arange(15.75, 23.76, 0.5)
QSO_Z = np.arange(0.5, 5.6, 1.0)
QSO_PLE = np.array(
    [
        [54, 5, 14, 0, 0, 0],
        [127, 23, 41, 3, 0, 0],
        [323, 104, 108, 9, 0, 0],
        [860, 469, 290, 25, 1, 0],
        [2316, 2009, 793, 64, 2, 0],
        [5854, 7439, 2204, 167, 5, 0],
        [12619, 21145, 5994, 432, 12, 1],
        [22090, 44412, 14675, 1100, 31, 2],
        [32538, 72781, 29834, 2700, 80, 5],
        [43787, 102680, 50203, 6208, 201, 12],
        [57009, 136093, 74188, 12952, 502, 30],
        [73477, 176857, 101326, 24039, 1217, 75],
        [94414, 228620, 132051, 39635, 2811, 189],
        [121186, 295216, 167766, 58919, 6038, 470],
        [155438, 381240, 210616, 80786, 11825, 1144],
        [199196, 492527, 263236, 104700, 20944, 2660],
        [254945, 636610, 328743, 131042, 33694, 5769],
    ],
    dtype=float,
)
QSO_PLE_LEDE = np.array(
    [
        [74, 12, 2, 0, 0, 0],
        [173, 46, 8, 1, 0, 0],
        [425, 173, 29, 3, 0, 0],
        [1063, 645, 109, 11, 1, 0],
        [2637, 2318, 407, 37, 3, 1],
        [6171, 7534, 1476, 119, 10, 3],
        [12760, 20293, 4934, 380, 27, 7],
        [22256, 42853, 14012, 1146, 71, 15],
        [33079, 72019, 31432, 3068, 182, 33],
        [44377, 102993, 55189, 6846, 432, 69],
        [56787, 135480, 81153, 12487, 915, 133],
        [71385, 172381, 108661, 19287, 1689, 232],
        [89178, 216865, 139912, 26781, 2731, 367],
        [111154, 271898, 177682, 35182, 3976, 534],
        [138390, 340627, 224720, 45094, 5393, 732],
        [172133, 426753, 283998, 57240, 7015, 962],
        [213829, 534834, 359048, 72418, 8921, 1234],
    ],
    dtype=float,
)
# the continuum-slope scatter on the [Fe/H] rows of the QSO track (make_qso_locus.py), as a prior on them
QSO_FEH = (-1.0, 0.5)
# A_r per unit of the SFD E(B-V), which the QSO counts are dimmed by, since a QSO lies behind the whole
# column: run_dp2.AR_PER_EBV, the LSST r ratio 2.701 with the Schlafly & Finkbeiner 2011 0.86, which is also
# what the Ar_SFD column of the DP2 input is (median ratio 0.9999 to sfdExtinction, 1 % scatter from the
# pixel lookup)
QSO_AR_PER_EBV = 2.323
# The QSO share is a density against a density: QSOs per deg2 from the counts, model stars per deg2 of the
# cone. The model catalogue has holes, and a cone four times wider than a pixel often runs into one, so the
# stars are taken per deg2 of the part of the cone the catalogue covers, from its HATS point map coarsened to
# this order: 0.05 deg2 pixels, each with tens of model stars where it is covered even at the Galactic poles.
COVERAGE_ORDER = 8


def keptPoints(n):
    """Which of n grid points a compact map keeps: every other one, and the last whatever the parity.

    The last point cannot be dropped along with the rest. It is the end of the axis, and the fit fills
    anything outside a map with zero, so half a grid step of [Fe/H] would stop carrying a prior at all
    rather than be interpolated.
    """
    kept = np.arange(0, n, 2)
    return kept if kept[-1] == n - 1 else np.append(kept, n - 1)


def compact(cube):
    """The maps small enough to keep beside the code: every other grid point, and a byte of log each.

    They are smoothed densities on a grid finer than the smoothing, and the fit interpolates them onto the
    locus anyway, so half the points carry them. Storing the log relative to each map's own peak costs about
    0.008 dex where the prior has weight and takes the file from gigabytes to tens of megabytes.
    """
    cube = cube[:, :, keptPoints(cube.shape[2])][:, :, :, keptPoints(cube.shape[3])]
    flat = cube.reshape(cube.shape[0] * cube.shape[1], -1)
    top = flat.max(axis=1, keepdims=True)
    rel = np.log10(np.maximum(flat / np.where(top > 0, top, 1.0), 10**-PRIOR_DECADES))
    levels = float(np.iinfo(np.uint8).max)
    q = np.rint((rel + PRIOR_DECADES) / PRIOR_DECADES * levels).astype(np.uint8)
    return q.reshape(cube.shape), top.reshape(cube.shape[0], cube.shape[1]).astype(np.float32)


def writePriors(out, cube, rGrid, xGrid, yGrid, index, order, small, constants, model=None, extra=None):
    """The maps in one file, compact enough to keep beside the code when asked for.

    The constants the maps were built on go in beside them. The grid of getBayesConstants() says what the two
    axes of a map mean and the decades say how the bytes of a compact one decode, neither is recoverable from
    the arrays, and a reader that assumes its own copy of either mis-places every star of every sightline
    without anything looking wrong.
    """
    shared = dict(
        rmag=rGrid,
        index=index,
        order=order,
        decades=PRIOR_DECADES,
        constants=json.dumps(constants),
        # which locus the tLoc axis is and which model stars went in, so that a run can refuse other maps
        **({"model": json.dumps(model)} if model else {}),
        **(extra or {}),
    )
    if small:
        q, scale = compact(cube)
        np.savez_compressed(
            out,
            kde=q,
            kdeScale=scale,
            xGrid=xGrid[keptPoints(xGrid.size)],
            yGrid=yGrid[keptPoints(yGrid.size)],
            **shared,
        )
        written = q.shape
    else:
        np.savez_compressed(out, kde=cube, xGrid=xGrid, yGrid=yGrid, **shared)
        written = cube.shape
    size = Path(out).stat().st_size / 2**20
    print(f"{out}: {len(cube)} pixels of order {order}, maps {written}, {size:.1f} MB on disk")


def fromCatalog(url, out, small):
    """The same file, out of prior maps that already exist as a HATS catalog rather than from the model.

    The maps are the same either way; what changes is that a catalog of them has to be joined against the
    stars, and that join drops partitions wherever the maps are the coarser of the two trees.
    """
    from hats.io.paths import pixel_catalog_file

    catalog = lsdb.open_catalog(url)
    pixels = catalog.hc_structure.get_healpix_pixels()
    base = catalog.hc_structure.catalog_base_dir
    order = max(p.order for p in pixels)
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])

    cube, index = [], np.full(12 * 4**order, -1, dtype=np.int32)
    xGrid = yGrid = None
    for pixel in pixels:
        table = pd.read_parquet(str(pixel_catalog_file(base, pixel)))
        if not len(table):
            continue
        rmag = table["rmag"].to_numpy(dtype=float)
        x = np.frombuffer(table["xGrid"].iloc[0], dtype=np.float64)
        nX = np.unique(x).size
        x = x.reshape(-1, nX)
        y = np.frombuffer(table["yGrid"].iloc[0], dtype=np.float64).reshape(x.shape)
        maps = np.stack(
            [
                np.frombuffer(table["kde"].iloc[int(np.argmin(np.abs(rmag - r)))], dtype=np.float64).reshape(
                    x.shape
                )
                for r in rGrid
            ]
        ).astype(np.float32)
        xGrid, yGrid = x[0], y[:, 0]
        spread = 4 ** (order - pixel.order)
        index[pixel.pixel * spread : (pixel.pixel + 1) * spread] = len(cube)
        cube.append(maps)
    if not cube:
        raise SystemExit(f"no prior maps in {url}")
    writePriors(out, np.stack(cube), rGrid, xGrid, yGrid, index, order, small, bc)


def footprintPixels(path, order):
    """Pixels of the given order that the footprint covers, from the sky map of the object catalog."""
    if not path:
        return np.arange(12 * 4**order, dtype=np.int64)
    with fits.open(path) as hdus:
        columns = [c.name for c in hdus[1].columns]
        if "PIXEL" in columns:
            pixel = np.asarray(hdus[1].data.field("PIXEL")).ravel().astype(np.int64)
            value = np.asarray(hdus[1].data.field("VALUE")).ravel().astype(float)
            nside = int(hdus[1].header["NSIDE"])
        else:
            value = np.asarray(hdus[1].data.field(0)).ravel().astype(float)
            pixel = np.arange(value.size, dtype=np.int64)
            nside = int(round(np.sqrt(value.size / 12)))
    pixel = pixel[value > 0]
    mapOrder = int(round(np.log2(nside)))
    if mapOrder >= order:
        return np.unique(pixel >> (2 * (mapOrder - order)))
    spread = 4 ** (order - mapOrder)
    return np.unique(np.concatenate([pixel * spread + i for i in range(spread)]))


def readLocus(locusPath=LOCUS):
    """The locus the maps are for, with its segment column (all stars for a file that has none)."""
    return lt.LSSTsimsLocus(
        fixForStripe82=False, datafile=str(locusPath), colnames=["tLoc", "Mr", "FeH", *COLORS]
    )


def locusAxes(locusPath=LOCUS):
    """The [Fe/H] and tLoc axes of the locus the fit uses, the segments that put model stars on them, the Mr
    range of the model stars kept and the grid of the maps (getBayesConstants for the locus)."""
    locus = readLocus(locusPath)
    feH, tLoc = np.unique(locus["FeH"]), np.unique(locus["tLoc"])
    table = np.asarray(locus["Mr"]).reshape(feH.size, tLoc.size)
    params = SimpleNamespace(FeH1d=feH, Mr1d=tLoc, MrTrueTable=table, segment1d=lt.segmentAxis(locus))
    # the model stars kept are those within the Mr range of the stellar rows: the QSO rows have a placeholder
    stellar = ~np.isin(
        np.asarray(locus.columns.get("segment", np.zeros(len(locus)))), lt.EXTRAGALACTIC_SEGMENTS
    )
    mr = np.asarray(locus["Mr"])[stellar]
    bc = getBayesConstants(float(tLoc.min()))
    return feH, lt.buildSegmentData(params), float(np.min(mr)), float(np.max(mr)), bc


def whiteDwarfTracks(locusPath=LOCUS):
    """Mr and tLoc along each white dwarf track of the locus, by segment, in order of Mr.

    A white dwarf's colours do not depend on [Fe/H], so a track is the same at every [Fe/H] and is read off
    the first.
    """
    locus = readLocus(locusPath)
    feH, segment = np.asarray(locus["FeH"]), np.asarray(locus.columns.get("segment", np.zeros(len(locus))))
    tracks = {}
    for code in lt.WHITE_DWARF_SEGMENTS:
        rows = (segment == code) & (feH == feH.min())
        if rows.any():
            order = np.argsort(np.asarray(locus["Mr"])[rows])
            tracks[code] = (np.asarray(locus["Mr"])[rows][order], np.asarray(locus["tLoc"])[rows][order])
    return tracks


MASS_KEYS = {lt.SEGMENT_HYDROGEN_WD: "H", lt.SEGMENT_HELIUM_WD: "He"}  # wH, coverH, ... of WD_massgrid.json


def massWeights(grid, locusPath=LOCUS):
    """How the prior on mass spreads each tLoc of a white dwarf track over the [Fe/H] rows, which are masses.

    grid is WD_massgrid.json of make_wd_mass_locus.py. The track maps are flat in [Fe/H], and at each tLoc
    the rows share its density in proportion to the prior on mass per unit [Fe/H] (wH, wHe), among the rows
    whose model reaches the Mr of that tLoc (coverH, coverHe): a massive white dwarf brighter than its
    hottest model has none. The weights have a mean of one over the rows, so the density along tLoc is what
    the TRILEGAL model gives, and every other point of the locus has weight one. Returned on the locus grid,
    (nFeH, nTLoc): on the compact maps, which keep every other [Fe/H] point, a row would take half of the
    weight of each of its neighbours.
    """
    locus = readLocus(locusPath)
    feH, tLoc = np.unique(locus["FeH"]), np.unique(locus["tLoc"])
    if feH.size != len(grid["feH"]) or not np.allclose(feH, grid["feH"]):
        raise SystemExit(f"the [Fe/H] rows of {locusPath} are not the ones the mass grid is for")
    segment = lt.segmentAxis(locus)
    mr = np.asarray(locus["Mr"], dtype=float).reshape(feH.size, tLoc.size)[0]
    out = np.ones((feH.size, tLoc.size))
    for code, key in MASS_KEYS.items():
        on = segment == code
        if not on.any():
            continue
        cover = np.asarray(grid[f"cover{key}"], dtype=float)
        inside = (mr[None, on] >= cover[:, :1]) & (mr[None, on] <= cover[:, 1:])
        w = np.asarray(grid[f"w{key}"], dtype=float)[:, None] * inside
        total = w.sum(axis=0)
        if np.any(total <= 0):
            raise SystemExit(f"no mass of the grid has a model at Mr {mr[on][total <= 0].min():.2f} ({key})")
        out[:, on] = w / total * feH.size
    return out


def heliumFraction(mr):
    """The helium share of white dwarfs of absolute magnitude mr, from the table above."""
    bins = np.clip(np.searchsorted(HELIUM_MR_EDGES, mr, side="right") - 1, 0, HELIUM_FRACTION.size - 1)
    return HELIUM_FRACTION[bins]


def pixelMaps(
    pixel, order, catalog, radius, maxStars, feH, segments, mrMin, mrMax, bc, tracks=None, qso=None
):
    """The prior maps of one pixel: model stars of a cone at its centre, binned in r and smoothed.

    Returns the maps and their two axes, or None when the cone holds no usable model stars, which leaves the
    pixel out of the file and the stars of that sightline without an estimate. On a locus with white dwarf
    tracks (tracks, of whiteDwarfTracks) the share of each population per r bin comes with them
    (populationMaps), and with qso, the QSOs are a population as well.
    """
    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.array([pixel]), order)
    ra, dec = float(lon.deg[0]), float(lat.deg[0])
    # the model catalog has holes; widening the cone once reaches the nearest model stars, and the prior is
    # smooth enough at the latitudes where the holes are for that to be the right neighbour to borrow from
    for reach in (radius, 3 * radius):
        stars = lsdb.open_catalog(
            catalog,
            columns=MODEL_COLUMNS,
            search_filter=lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=reach * 3600),
        ).compute()
        if len(stars) >= MIN_STARS:
            break
    if len(stars) < 3:
        return None
    stars = pd.DataFrame(
        {c: stars[c].to_numpy(dtype=np.int32 if c == "label" else float) for c in MODEL_COLUMNS}
    )
    keep = stars["FeH"].between(feH[0], feH[-1]) & stars["Mr"].between(mrMin, mrMax)
    if not tracks:
        keep &= stars["label"] != lt.WHITE_DWARF_LABEL
    model = stars[keep].copy()
    del stars
    if len(model) < 3:
        return None
    dwarfs = model[model["label"] == lt.WHITE_DWARF_LABEL]
    # A model star whose Mr falls in more than one segment of its [Fe/H] track, and whose evolutionary label
    # does not say which, has no tLoc and cannot go on the map. That is a handful of stars per pixel; a large
    # share of them means the segment labels of locus.SEGMENT_LABEL_MAP no longer describe this locus, which
    # would take the giants out of the prior without anything failing.
    placed = lt.assignTLocPartition(model, segments, feH)
    lost = int(placed["tLoc"].isna().sum())
    if lost > 0.01 * len(placed):
        print(f"  pixel {pixel}: {lost} of {len(placed)} model stars have no tLoc on this locus")
    model = placed[placed["tLoc"].notna()]
    if len(model) < 3:
        return None

    metadata = np.array(
        [
            bc["FeHmin"],
            bc["FeHmax"],
            bc["FeHNpts"],
            bc["MrFaint"],
            bc["MrBright"],
            bc["MrNpts"],
            float(np.mean(model["glon"])),
            float(np.mean(model["glat"])),
            order,
            pixel,
        ]
    )
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    if not tracks:
        return binnedMaps(model, rGrid, bc, metadata, maxStars, pixel)
    # the white dwarfs go on the tracks by populationMaps, every one of them on both with the expected weights
    stars = model[model["label"] < lt.WHITE_DWARF_LABEL]
    pixelQso = None if qso is None else dict(gap=qso["gap"], ar=qso["ar"][int(pixel)])
    shares, wide = populationShares(
        catalog,
        ra,
        dec,
        4 * radius,
        rGrid,
        bc["rmagBinWidth"],
        feH,
        mrMin,
        mrMax,
        tracks,
        qsoAr=None if pixelQso is None else pixelQso["ar"],
        covered=None if qso is None else qso["covered"],
    )
    return populationMaps(stars, dwarfs, wide, shares, rGrid, bc, metadata, maxStars, pixel, tracks, pixelQso)


def populationSegments(tracks, qso=False):
    """The segments of the populations of a three-population build: the stars, then each track it has, and the
    QSOs last when they are built too."""
    codes = [lt.SEGMENT_STARS, *(code for code in lt.WHITE_DWARF_SEGMENTS if code in tracks)]
    return codes + [lt.SEGMENT_QSO] if qso else codes


def qsoDensity(r):
    """dN / dr / dz per deg2 at dereddened r, for each redshift bin of QSO_Z: the mean of the two models."""
    logT = np.log10(np.maximum(0.5 * (QSO_PLE + QSO_PLE_LEDE) / 1e4 / 0.5, 1e-6))
    faint = logT[-1] - logT[-3]  # per magnitude: the bins are half a magnitude apart
    bright = logT[2] - logT[0]
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty((r.size, QSO_Z.size))
    for j in range(QSO_Z.size):
        v = np.interp(r, QSO_R, logT[:, j])
        v = np.where(r > QSO_R[-1], logT[-1, j] + faint[j] * (r - QSO_R[-1]), v)
        v = np.where(r < QSO_R[0], logT[0, j] + bright[j] * (r - QSO_R[0]), v)
        out[:, j] = 10**v
    return out


QSO_ZFINE = np.linspace(0.0, 6.0, 601)


def qsoCounts(rLow, rHigh, ar):
    """QSOs per deg2 with observed r in [rLow, rHigh] behind an extinction ar and a redshift on the track, and
    the density of their redshifts on QSO_ZFINE. The redshift bins of the table are a unit wide, too coarse
    for a shape, so the counts per unit z are interpolated linearly between the bin centres."""
    r = np.linspace(rLow, rHigh, 1 + max(2, int(round(40 * (rHigh - rLow))))) - ar
    perZ = qsoDensity(r).mean(axis=0) * (rHigh - rLow)
    z = QSO_ZFINE
    dndz = np.where((z >= lt.QSO_REDSHIFT[0]) & (z <= lt.QSO_REDSHIFT[1]), np.interp(z, QSO_Z, perZ), 0.0)
    n = float(np.trapezoid(dndz, z))
    return n, dndz / n if n > 0 else dndz


def qsoGap(locusPath, compact):
    """The open stretch (low, high) of the map's tLoc axis that only the QSO points of the locus read.

    A locus point takes its prior from the two map points around it, so a point of another segment reads one
    map point inside the gap between the segments at most: the first above the last point below the track,
    and the last below the first point above it. Those two bound the stretch, on the grid the file stores.
    """
    locus = readLocus(locusPath)
    t = np.unique(np.asarray(locus["tLoc"], dtype=float))
    segment = lt.segmentAxis(locus)
    track = t[segment == lt.SEGMENT_QSO]
    others = t[segment != lt.SEGMENT_QSO]
    bc = getBayesConstants(t.min())
    grid = np.linspace(bc["MrFaint"], bc["MrBright"], bc["MrNpts"])
    grid = np.sort(grid[keptPoints(grid.size)] if compact else grid)
    eps = 1e-6
    low = grid[grid >= others[others < track.min()].max() - eps].min()
    high = grid[grid <= others[others > track.max()].min() + eps].max()
    if not (grid[grid <= track.min() + eps].max() > low and grid[grid >= track.max() - eps].min() < high):
        raise SystemExit(f"the QSO track of {locusPath} reads map points that its neighbours read too")
    return float(low), float(high)


def inGap(yGrid, gap, tolerance=1e-6):
    """The map points strictly inside the gap of qsoGap, its ends excluded however they were rounded."""
    return (yGrid > gap[0] + tolerance) & (yGrid < gap[1] - tolerance)


def qsoMaps(rGrid, halfWidth, ar, xGrid, yGrid, gap):
    """Maps of unit mass of the QSOs per r bin: their redshifts along the track, N(QSO_FEH) in [Fe/H].

    The redshift density is p(z | r) of qsoCounts carried to tLoc, held at the end values just past the
    ends of the track, on the map points of the gap (qsoGap).
    """
    dx, dy = xGrid[1] - xGrid[0], abs(yGrid[1] - yGrid[0])
    feH = np.exp(-0.5 * ((xGrid - QSO_FEH[0]) / QSO_FEH[1]) ** 2)
    feH /= feH.sum() * dx
    on = inGap(yGrid, gap)
    z = np.clip(lt.qsoRedshift(yGrid), *lt.QSO_REDSHIFT)
    dzdt = (lt.QSO_REDSHIFT[1] - lt.QSO_REDSHIFT[0]) / (lt.QSO_TLOC[1] - lt.QSO_TLOC[0])
    maps = np.zeros((rGrid.size, yGrid.size, xGrid.size))
    for i, r in enumerate(rGrid):
        _, pz = qsoCounts(r - halfWidth, r + halfWidth, ar)
        dens = np.where(on, np.interp(z, QSO_ZFINE, pz) * dzdt, 0.0)
        if dens.sum() > 0:
            maps[i] = (dens / (dens.sum() * dy))[:, None] * feH[None, :]
    return maps


def sfdExtinction(lon, lat, directory, arPerEbv=QSO_AR_PER_EBV):
    """A_r of the SFD map (Schlegel, Finkbeiner & Davis 1998) at Galactic lon, lat in degrees, nearest pixel.

    The map is two Lambert zenithal equal-area projections, one per Galactic hemisphere, as their headers
    give: x = sqrt(1 - n sin b) cos l * SCALE and y = -n sqrt(1 - n sin b) sin l * SCALE about the pole.
    """
    from astropy.io import fits

    lon, lat = np.radians(np.asarray(lon, dtype=float)), np.radians(np.asarray(lat, dtype=float))
    ebv = np.full(lon.shape, np.nan)
    for name, n in (("ngp", 1), ("sgp", -1)):
        sel = lat >= 0 if n == 1 else lat < 0
        with fits.open(Path(directory) / f"SFD_dust_4096_{name}.fits") as hdus:
            data, header = np.asarray(hdus[0].data), hdus[0].header
        rho = np.sqrt(1 - n * np.sin(lat[sel])) * header["LAM_SCAL"]
        x = np.rint(rho * np.cos(lon[sel]) + header["CRPIX1"] - 1).astype(int)
        y = np.rint(-n * rho * np.sin(lon[sel]) + header["CRPIX2"] - 1).astype(int)
        ebv[sel] = data[np.clip(y, 0, data.shape[0] - 1), np.clip(x, 0, data.shape[1] - 1)]
    return arPerEbv * ebv


def pixelExtinction(pixels, order, directory, arPerEbv=QSO_AR_PER_EBV, depth=4):
    """The median SFD A_r of each pixel, over the centres of its sub-pixels depth orders down."""
    from astropy.coordinates import SkyCoord

    spread = 4**depth
    children = (np.asarray(pixels, dtype=np.int64)[:, None] * spread + np.arange(spread)).ravel()
    lon, lat = cdshealpix.nested.healpix_to_lonlat(children, order + depth)
    galactic = SkyCoord(ra=lon, dec=lat, frame="icrs").galactic
    ar = sfdExtinction(galactic.l.deg, galactic.b.deg, directory, arPerEbv).reshape(-1, spread)
    return dict(zip((int(p) for p in pixels), np.median(ar, axis=1).tolist(), strict=True))


def catalogueCoverage(catalog, order=COVERAGE_ORDER):
    """Which nested HEALPix pixels of this order hold model stars, from the catalogue's HATS point map."""
    path = Path(catalog) / "point_map.fits"
    if not path.exists():
        raise SystemExit(f"{catalog} has no point_map.fits, which the QSO share takes the covered area from")
    with fits.open(path) as hdus:
        hdu = hdus[1]
        counts = np.asarray(hdu.data[hdu.columns.names[0]]).ravel()
        ordering, nside = str(hdu.header.get("ORDERING", "NESTED")).strip(), int(hdu.header["NSIDE"])
    mapOrder = int(round(np.log2(nside)))
    if ordering != "NESTED" or mapOrder < order or counts.size != 12 * 4**mapOrder:
        raise SystemExit(f"{path} is not a nested HEALPix map of order {order} or finer")
    return counts.reshape(-1, 4 ** (mapOrder - order)).sum(axis=1) > 0


def coneCoverage(covered, ra, dec, radius, order=COVERAGE_ORDER):
    """The fraction of a cone (degrees) that a coverage map of catalogueCoverage holds model stars in."""
    import astropy.units as u
    from astropy.coordinates import Latitude, Longitude

    pixels = cdshealpix.nested.cone_search(
        lon=Longitude(ra, unit="deg"),
        lat=Latitude(dec, unit="deg"),
        radius=radius * u.deg,
        depth=order,
        flat=True,
    )[0]
    return float(np.mean(covered[np.asarray(pixels, dtype=np.int64)]))


def trackWeights(mr, helium, code):
    """How much of each white dwarf goes on a track: 1 - f_He(Mr) on hydrogen and f_He(Mr) on helium, where
    the locus has a helium track (helium), and all of it on hydrogen where it has none.

    The expected fraction rather than a draw: a cone with a handful of white dwarfs in an r bin would
    otherwise give the helium track no prior at all, or a whole star's worth, by the luck of the draw.
    """
    fHe = heliumFraction(mr) if helium else np.zeros(np.shape(mr))
    return fHe if code == lt.SEGMENT_HELIUM_WD else 1.0 - fHe


def onTrack(mr, track):
    """Whether each Mr is within the stretch of a white dwarf track, (Mr, tLoc), where trackMaps places it."""
    mrTrack = np.asarray(track[0], dtype=float)
    return (np.asarray(mr, dtype=float) >= mrTrack.min()) & (np.asarray(mr, dtype=float) <= mrTrack.max())


def populationMaps(stars, dwarfs, wide, shares, rGrid, bc, metadata, maxStars, pixel, tracks, qso=None):
    """The maps of a locus with white dwarf tracks, and the share of each population in every r bin.

    Each population is a shape of unit mass, and the fit holds each to its share on its own stretch of the
    locus (run_dp2.renormalisedPopulations), so the stored maps are the sum of the shapes and say nothing
    by themselves about how the populations compare. The stars are the 2D map of binnedMaps. A white dwarf
    track is a 1D density in tLoc, flat in [Fe/H], since a white dwarf's colours do not depend on it. Each
    track has a map of its own: one map for both would take its bandwidth from the distance between the tracks
    (about 4 in tLoc), which smooths the luminosity function over two magnitudes and leaks a third of the mass
    into the gap and onto the other track. A cone with no model white dwarf takes the shape from the cone of
    the shares, four times wider. With QSOs (qso, the pixel's A_r and the gap of qsoGap) the other populations
    are set to zero on the map points that only the QSO track reads, where only their smoothing tails were,
    and the QSO maps are added there.
    """
    built = binnedMaps(stars, rGrid, bc, metadata, maxStars, pixel)
    if built is None:
        return None
    maps, xGrid, yGrid = built
    maps = maps.astype(np.float64)
    helium = lt.SEGMENT_HELIUM_WD in tracks
    for code in populationSegments(tracks)[1:]:
        shape = trackMaps(dwarfs, helium, code, tracks[code], rGrid, bc, xGrid, yGrid)
        if shape is None:
            shape = trackMaps(wide, helium, code, tracks[code], rGrid, bc, xGrid, yGrid)
        if shape is None:
            shape = trackMaps(None, helium, code, tracks[code], rGrid, bc, xGrid, yGrid)
        maps += shape
    if qso is not None:
        shape = qsoMaps(rGrid, bc["rmagBinWidth"], qso["ar"], xGrid, yGrid, qso["gap"])
        maps = withQsoMaps(maps, shape, yGrid, qso["gap"])
    return maps.astype(np.float32), xGrid, yGrid, shares


def withQsoMaps(maps, shape, yGrid, gap):
    """The maps of the other populations with the QSO shape in the gap, where only their tails were.

    The QSO shape is scaled down to the peak of the others wherever it is higher. Only a shape matters, since
    the fit holds each population to its share, and a compact file keeps each map to PRIOR_DECADES below its
    own peak: a concentrated QSO shape setting the peak would put the faintest points of the other
    populations under the floor, which makes them impossible rather than unlikely.
    """
    maps = np.array(maps, dtype=np.float64)
    maps[:, inGap(yGrid, gap), :] = 0.0
    others = maps.max(axis=(1, 2))
    top = shape.max(axis=(1, 2))
    scale = np.where(top > others, others / np.where(top > 0, top, 1.0), 1.0)
    return maps + shape * scale[:, None, None]


def trackMaps(dwarfs, helium, code, track, rGrid, bc, xGrid, yGrid, least=3, floor=0.02):
    """Maps of unit mass of one white dwarf track per r bin: a 1D kernel density in tLoc, flat in [Fe/H].

    Scott's rule on the track's own stars, n_eff^(-1/5) times their weighted spread, with floor (in tLoc) on
    both the spread and the bandwidth. An r bin with fewer than least white dwarfs borrows the nearest one
    that has them. With no white dwarfs at all (dwarfs None or empty) the track is flat along its length.
    """
    mrTrack, tTrack = track
    ascending = yGrid[0] < yGrid[-1]
    ya = yGrid if ascending else yGrid[::-1]
    dy = ya[1] - ya[0]
    dens = np.zeros((rGrid.size, ya.size))
    made = np.zeros(rGrid.size, dtype=bool)
    if dwarfs is not None and len(dwarfs):
        mr = dwarfs["Mr"].to_numpy(float)
        t = np.interp(mr, mrTrack, tTrack, left=np.nan, right=np.nan)
        w = trackWeights(mr, helium, code)
        ok = np.isfinite(t) & (w > 0)
        rmag, t, w = dwarfs["rmag"].to_numpy(float)[ok], t[ok], w[ok]
        for index, r in enumerate(rGrid):
            near = np.abs(rmag - r) <= bc["rmagBinWidth"]
            if np.count_nonzero(near) < least:
                continue
            y, wy = t[near], w[near]
            f = (y - ya[0]) / dy
            i = np.floor(f).astype(int)
            inside = (i >= 0) & (i < ya.size - 1)
            counts = np.zeros(ya.size)
            np.add.at(counts, i[inside], (wy * (1 - (f - np.floor(f))))[inside])
            np.add.at(counts, i[inside] + 1, (wy * (f - np.floor(f)))[inside])
            nEff = wy.sum() ** 2 / np.sum(wy**2)
            mean = np.sum(wy * y) / wy.sum()
            spread = np.sqrt(np.sum(wy * (y - mean) ** 2) / wy.sum())
            bandwidth = max(nEff ** (-1.0 / 5.0) * max(spread, floor), floor)
            # the helium track starts on the first point of the grid, and the smoothing has nothing beyond
            # it, so the density at the track's hot end falls to about half its value; the fit holds the
            # track to its share (run_dp2.renormalisedPopulations), not to that shape
            dens[index] = gaussian_filter1d(counts, bandwidth / dy, mode="constant") / (wy.sum() * dy)
            made[index] = True
    if not made.any():
        if dwarfs is not None:
            return None
        along = (ya >= tTrack.min()) & (ya <= tTrack.max())
        dens[:] = along / (along.sum() * dy)
    else:
        nearest = np.argsort(np.abs(np.arange(rGrid.size)[:, None] - np.where(made)[0][None, :]), axis=1)[
            :, 0
        ]
        dens = dens[np.where(made)[0][nearest]]
    if not ascending:
        dens = dens[:, ::-1]
    return np.repeat(dens[:, :, None] / (xGrid[-1] - xGrid[0]), xGrid.size, axis=2)


def populationShares(
    catalog,
    ra,
    dec,
    radius,
    rGrid,
    halfWidth,
    feH,
    mrMin,
    mrMax,
    tracks,
    enough=10,
    widest=2.0,
    qsoAr=None,
    covered=None,
):
    """The share of each population (populationSegments) among the model stars of each r bin, and the white
    dwarfs of the cone. The shares move slowly across the sky but steeply with r, so they are counted in a
    cone wider than the maps use (radius), and a bin short of white dwarfs takes them from more sky at the
    same r rather than from fainter r, where there are many more of them: its r window widens only where it
    holds fewer than enough white dwarfs, up to widest. The white dwarfs are split between the tracks by
    their expected helium fraction, as trackWeights, and only those within the Mr range of a track (onTrack)
    are counted at all, the same ones the track maps are made of.

    With qsoAr, the A_r of the pixel, the QSOs are a population as well, last: their share q is that of the
    literature counts in the same r window (qsoCounts) among those and the model stars per deg2 of the cone,
    and the shares of the stars and the white dwarfs are taken among the model stars times 1 - q. The area
    is that of the part of the cone the catalogue covers (covered, of catalogueCoverage; the whole cone when
    it is None), since a cone that runs into a hole of the catalogue holds fewer stars for it but no fewer
    QSOs, and taken over the whole cone that would raise q by the inverse of the fraction covered.
    """
    stars = lsdb.open_catalog(
        catalog,
        columns=["rmag", "label", "FeH", "Mr"],
        search_filter=lsdb.ConeSearch(ra=ra, dec=dec, radius_arcsec=radius * 3600),
    ).compute()
    keep = (stars["FeH"].between(feH[0], feH[-1]) & stars["Mr"].between(mrMin, mrMax)).to_numpy()
    rmag = stars["rmag"].to_numpy(float)[keep]
    mr = stars["Mr"].to_numpy(float)[keep]
    dwarf = stars["label"].to_numpy()[keep] >= lt.WHITE_DWARF_LABEL
    codes = populationSegments(tracks)
    helium = lt.SEGMENT_HELIUM_WD in tracks
    # A white dwarf counts where a track can hold it, as in trackMaps: TRILEGAL has them to Mr 16 and beyond,
    # the tracks stop at 14.5, and one past the end has no place on the locus. Counted here it would give the
    # tracks a share of stars their maps never hold, about a third more at r 23-24 than they do hold.
    weights = [trackWeights(mr, helium, code) * onTrack(mr, tracks[code]) for code in codes[1:]]
    held = ~dwarf | np.any([onTrack(mr, tracks[code]) for code in codes[1:]], axis=0)
    rmag, mr, dwarf, weights = rmag[held], mr[held], dwarf[held], [w[held] for w in weights]
    shares = np.zeros((rGrid.size, len(codes) + (qsoAr is not None)))
    area = 2 * np.pi * (1 - np.cos(np.radians(radius))) * (180 / np.pi) ** 2
    if qsoAr is not None and covered is not None:
        area *= max(coneCoverage(covered, ra, dec, radius), 1e-12)
    for i, r in enumerate(rGrid):
        w = halfWidth
        while True:
            near = np.abs(rmag - r) <= w
            if np.count_nonzero(near & dwarf) >= enough or w >= widest:
                break
            w += halfWidth
        n = max(np.count_nonzero(near), 1)
        shares[i, 0] = np.count_nonzero(near & ~dwarf) / n
        for k, weight in enumerate(weights, start=1):
            shares[i, k] = float(np.sum(weight[near & dwarf])) / n
        if qsoAr is not None:
            nQ = qsoCounts(r - w, r + w, qsoAr)[0]
            nStars = np.count_nonzero(near) / area
            q = nQ / (nQ + nStars) if nQ + nStars > 0 else 0.0
            shares[i, :-1] *= 1 - q
            shares[i, -1] = q
    wide = pd.DataFrame({"rmag": rmag[dwarf], "Mr": mr[dwarf]})
    return shares, wide


def binnedMaps(model, rGrid, bc, metadata, maxStars, pixel):
    """The maps of one population per r bin, an empty bin borrowing the nearest one."""
    maps, made, xGrid, yGrid = None, np.zeros(rGrid.size, dtype=bool), None, None
    for index, r in enumerate(rGrid):
        inBin = model[model["rmag"].between(r - bc["rmagBinWidth"], r + bc["rmagBinWidth"])]
        if len(inBin) < 3:
            continue
        if maxStars and len(inBin) > maxStars:
            inBin = inBin.sample(n=maxStars, random_state=int(pixel))
        xGrid, yGrid, kde = get2Dmap(inBin, ["FeH", "tLoc", "rmag"], metadata)
        nX = np.unique(xGrid).size
        if maps is None:
            maps = np.zeros((rGrid.size, kde.size // nX, nX), dtype=np.float32)
        maps[index] = kde.reshape(maps.shape[1], nX)
        made[index] = True
    if maps is None:
        return None
    # an r bin with no model stars borrows the nearest one that has them, so that every bin has a map
    nearest = np.argsort(np.abs(np.arange(rGrid.size)[:, None] - np.where(made)[0][None, :]), axis=1)[:, 0]
    maps = maps[np.where(made)[0][nearest]]
    return maps, xGrid[0], yGrid[:, 0] if yGrid.ndim > 1 else yGrid


def startWorker(order, catalog, radius, maxStars, locusPath=LOCUS, qso=None):
    """Read the locus once per process: it is the same for every pixel and costs more than a pixel does."""
    feH, segments, mrMin, mrMax, bc = locusAxes(locusPath)
    WORKER.update(
        qso=qso,
        tracks=whiteDwarfTracks(locusPath),
        order=order,
        catalog=catalog,
        radius=radius,
        maxStars=maxStars,
        axes=(feH, segments, mrMin, mrMax, bc),
    )


def buildPixel(pixel):
    """Pool entry point: one pixel with its maps, None if the catalog cannot serve it, or what it raised.

    A pixel that raises must not take a build of thousands of them down, and must not disappear either: the
    run fits no star of a sightline that has no maps, so the failure is handed back to be reported. It
    goes back as text because it has to survive being pickled out of the worker, which not every exception
    of a catalog reader does.
    """
    try:
        return int(pixel), pixelMaps(
            int(pixel),
            WORKER["order"],
            WORKER["catalog"],
            WORKER["radius"],
            WORKER["maxStars"],
            *WORKER["axes"],
            WORKER["tracks"],
            WORKER["qso"],
        )
    except Exception as error:
        return int(pixel), f"{type(error).__name__}: {error}"


def reportHoles(failed, empty, allowMissing):
    """List the pixels of the footprint that ended up with no maps, and stop unless a hole was intended.

    The run looks a star's sightline up in the index of the file, and the stars of a pixel that is not in it
    keep their rows with no estimate, so a hole here is stars without answers rather than a rougher prior. The
    file is written first either way: a build of thousands of pixels is hours of catalog reading and the
    partial file is worth keeping while the cause of the holes is found.
    """
    if failed:
        print(f"{len(failed)} pixels raised and have no maps:")
        for pixel, failure in failed[:MAX_REPORTED]:
            print(f"  {pixel}: {failure}")
        if len(failed) > MAX_REPORTED:
            print(f"  and {len(failed) - MAX_REPORTED} more")
    if empty:
        listed = ", ".join(str(pixel) for pixel in empty[:MAX_REPORTED])
        more = f" and {len(empty) - MAX_REPORTED} more" if len(empty) > MAX_REPORTED else ""
        print(f"{len(empty)} pixels had too few model stars, even in the widened cone: {listed}{more}")
    if (failed or empty) and not allowMissing:
        raise SystemExit(
            f"{len(failed) + len(empty)} pixels of the footprint have no maps in the file just written, and "
            "the run fits no star of a sightline that has none; find the cause or pass --allow-missing to "
            "take the file as it stands"
        )


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trilegal", default="", help="TRILEGAL model catalog (HATS) to build the maps from")
    ap.add_argument(
        "--from-catalog",
        default="",
        help="prior maps that already exist as a HATS catalog, to put in one file instead of building them",
    )
    ap.add_argument("--out", required=True, help="file to write the prior maps to (npz)")
    ap.add_argument("--footprint", default="", help="sky map of the object catalog (skymap.N.fits)")
    ap.add_argument("--order", type=int, default=5, help="HEALPix order of the maps (5 is 1.8 degrees)")
    ap.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="radius of the cone of model stars per pixel; a circle of the pixel's area by default",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="store the maps on every other grid point as a byte of log each, which is tens of megabytes "
        "instead of gigabytes and moves the median star by under two millimagnitudes",
    )
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument(
        "--max-stars",
        type=int,
        default=200000,
        help="most model stars used for one map; a smooth density needs no more and the plane holds millions",
    )
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=8,
        help="pixels a worker builds before it is replaced, which keeps the catalog reader from growing",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="write the file even though some pixels of the footprint have no maps, which is a build meant "
        "to be partial; without it such a build reports them and exits non-zero",
    )
    ap.add_argument(
        "--locus",
        default=str(LOCUS),
        help="the locus the maps are for, which must be the one the run fits (run_dp2.py --locus)",
    )
    ap.add_argument(
        "--pixels",
        default="",
        help="comma separated HEALPix pixel numbers at --order to build instead of the whole footprint, "
        "for a patch-sized build",
    )
    ap.add_argument(
        "--wd-mass-grid",
        default="",
        help="the mass grid of the locus (data/WD_massgrid_QSO.json of make_qso_locus.py), for a locus whose "
        "white dwarf rows are masses: the prior on mass then spreads each tLoc of a track over them",
    )
    ap.add_argument(
        "--qso-counts",
        action="store_true",
        help="for a locus with a QSO track: add the QSOs as a population, with the Palanque-Delabrouille et "
        "al. 2016 counts dimmed by the pixel's --dust-map A_r",
    )
    ap.add_argument("--dust-map", default="", help="directory of the SFD maps SFD_dust_4096_{ngp,sgp}.fits")
    args = ap.parse_args()

    if args.from_catalog:
        fromCatalog(args.from_catalog, args.out, args.compact)
        return
    if not args.trilegal:
        raise SystemExit("give either --trilegal to build the maps or --from-catalog to convert them")

    if args.pixels:
        pixels = np.asarray([int(p) for p in args.pixels.split(",") if p.strip()], dtype=np.int64)
    else:
        pixels = footprintPixels(args.footprint, args.order)
    area = 4 * np.pi * (180 / np.pi) ** 2 / (12 * 4**args.order)
    radius = args.radius or float(np.sqrt(area / np.pi))
    print(f"{pixels.size} pixels of order {args.order} ({area:.2f} deg2), cone radius {radius:.2f} deg")

    locus = readLocus(args.locus)
    tracks = whiteDwarfTracks(args.locus)
    massAxis = lt.hasWhiteDwarfMassAxis(locus)
    hasQso = lt.SEGMENT_QSO in lt.segmentAxis(locus)
    bc = getBayesConstants(float(np.min(locus["tLoc"])))
    model, extra, qso = None, {}, None
    if tracks:
        helium = "table" if lt.SEGMENT_HELIUM_WD in tracks else None
        model = dict(locus=Path(args.locus).name, labels="all", heliumFraction=helium, split="three")
        extra["popSegment"] = np.asarray(populationSegments(tracks), dtype=np.int32)
        model["populations"] = extra["popSegment"].tolist()
        model["whiteDwarfShape"] = "1D kernel density per track, expected helium fraction, flat in [Fe/H]"
        model["mixture"] = "maps are unit-mass shapes; popShare holds each population's share per r bin"
        model["whiteDwarfShare"] = "TRILEGAL label 9 within the Mr range of a track only"
    elif hasQso:
        raise SystemExit(f"{args.locus} has a QSO track without white dwarf tracks, which no build is for")
    if massAxis != bool(args.wd_mass_grid):
        raise SystemExit(
            f"the white dwarf rows of {args.locus} are masses: give their --wd-mass-grid"
            if massAxis
            else f"--wd-mass-grid needs a locus whose white dwarf rows are masses, not {args.locus}"
        )
    if args.wd_mass_grid:
        raw = Path(args.wd_mass_grid).read_bytes()
        grid = json.loads(raw)
        if grid["locus"] != Path(args.locus).name:
            raise SystemExit(f"{args.wd_mass_grid} is the mass grid of {grid['locus']}, not of {args.locus}")
        extra["wdMassWeight"] = massWeights(grid, args.locus).astype(np.float32)
        model["wdMass"] = dict(
            grid=Path(args.wd_mass_grid).name, masses=grid["masses"], md5=hashlib.md5(raw).hexdigest()
        )
        model["whiteDwarfShape"] += "; the prior on mass spreads each tLoc over the rows (wdMassWeight)"
    if hasQso != args.qso_counts:
        raise SystemExit(
            f"{args.locus} has a QSO track: give --qso-counts"
            if hasQso
            else f"--qso-counts needs a locus with a QSO track, not {args.locus}"
        )
    if args.qso_counts:
        if not args.dust_map:
            raise SystemExit("--qso-counts needs --dust-map")
        qso = dict(
            gap=qsoGap(args.locus, args.compact),
            ar=pixelExtinction(pixels, args.order, args.dust_map),
            covered=catalogueCoverage(args.trilegal),
        )
        extra["popSegment"] = np.append(extra["popSegment"], lt.SEGMENT_QSO).astype(np.int32)
        model["populations"] = extra["popSegment"].tolist()
        model["qso"] = dict(
            counts="Palanque-Delabrouille et al. 2016 Table 7, mean of PLE and PLE+LEDE, dimmed by the "
            "pixel A_r",
            redshift=list(lt.QSO_REDSHIFT),
            feH=list(QSO_FEH),
            dustMap=f"SFD median over the order-{args.order + 4} sub-pixels, A_r = {QSO_AR_PER_EBV} E(B-V)",
            dustMapFiles=str(Path(args.dust_map).resolve()),
            gap=list(qso["gap"]),
            starDensity=f"model stars per deg2 of the part of the share cone the catalogue covers (its "
            f"point map at order {COVERAGE_ORDER})",
        )
        values = np.asarray(list(qso["ar"].values()))
        print(
            f"QSO population: pixel A_r median {np.median(values):.3f}, max {values.max():.2f}; "
            f"gap {qso['gap']}"
        )
    if model:
        print(f"locus {model['locus']}, populations {model['populations']}, helium split {helium}")

    built, maps, axes, failed, empty, shareList = [], [], None, [], [], []
    with mp.Pool(
        args.processes,
        initializer=startWorker,
        initargs=(args.order, args.trilegal, radius, args.max_stars, args.locus, qso),
        maxtasksperchild=args.max_tasks,
    ) as pool:
        for done, (pixel, result) in enumerate(pool.imap_unordered(buildPixel, pixels, chunksize=1), start=1):
            if isinstance(result, str):
                failed.append((pixel, result))
            elif result is None:
                empty.append(pixel)
            else:
                cube, xGrid, yGrid = result[:3]
                built.append(pixel)
                maps.append(cube)
                shareList.append(result[3] if len(result) > 3 else None)
                if axes is None:
                    # the axes are the grid of getBayesConstants() and are the same for every pixel, so the
                    # first pixel to arrive sets them rather than whichever one happens to arrive last
                    axes = (xGrid, yGrid)
            if done % 100 == 0 or done == pixels.size:
                print(
                    f"  {done}/{pixels.size} pixels, {len(built)} built, {len(empty)} without model stars, "
                    f"{len(failed)} failed",
                    flush=True,
                )
    if not built:
        raise SystemExit("no prior maps were built: check the footprint and the model catalog")

    byPixel = np.argsort(built)
    cube = np.stack([maps[i] for i in byPixel]).astype(np.float32)
    index = np.full(12 * 4**args.order, -1, dtype=np.int32)
    index[np.asarray(built)[byPixel]] = np.arange(len(built), dtype=np.int32)
    if tracks:
        extra["popShare"] = np.stack([shareList[i] for i in byPixel]).astype(np.float64)
    if qso is not None:
        extra["qsoAr"] = np.asarray([qso["ar"][built[i]] for i in byPixel], dtype=np.float64)
        q = np.median(extra["popShare"][:, :, -1], axis=0)
        rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
        print(
            "median QSO share per r bin: "
            + " ".join(f"{r:.1f}:{v:.2e}" for r, v in zip(rGrid, q, strict=True))
        )
    writePriors(
        args.out,
        cube,
        np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"]),
        axes[0].astype(np.float64),
        axes[1].astype(np.float64),
        index,
        args.order,
        args.compact,
        bc,
        model,
        extra,
    )
    reportHoles(failed, empty, args.allow_missing)


if __name__ == "__main__":
    main()
