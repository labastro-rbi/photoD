"""TRILEGAL prior maps for a photoD run: p([Fe/H], tLoc | r) on a HEALPix grid, in one file.

  python scripts/make_priors.py --trilegal /path/to/TRILEGAL_cluster_v08 --out priors_dp2.npz \\
                                --footprint /path/to/rubin_dp2/object_collection/object_lc/skymap.6.fits

One map per r bin of getBayesConstants() and per HEALPix pixel of the given order, stored as an array with an
index from pixel to row, so that the run looks a sightline up rather than joining two catalogs. The model
stars of a pixel are read from a cone of the same area at its centre and placed on the tLoc axis of the locus
the fit will use, which is what makes the maps and the fit comparable.

Only pixels of the footprint are built. The footprint comes from the sky map of the object catalog, either a
sparse table of PIXEL and VALUE or a full HEALPix array; without one the whole sky is built.
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import cdshealpix
import lsdb
import numpy as np
import pandas as pd
from astropy.io import fits

import photod.locus as lt
from photod.priors import get2Dmap, getBayesConstants

LOCUS = Path(__file__).resolve().parents[1] / "data" / "LSSTlocus_10Gyr_DP2.txt"
COLORS = ("ug", "gr", "ri", "iz", "zy")
MODEL_COLUMNS = ["ra", "dec", "glon", "glat", "DM", "Av", "rmag", "FeH", "Mr", "label"]
MIN_STARS = 2000
WORKER = {}


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


def locusAxes():
    """The [Fe/H] and tLoc axes of the locus the fit uses, and the segments that put model stars on them."""
    locus = lt.LSSTsimsLocus(
        fixForStripe82=False, datafile=str(LOCUS), colnames=["tLoc", "Mr", "FeH", *COLORS]
    )
    feH, tLoc = np.unique(locus["FeH"]), np.unique(locus["tLoc"])
    table = np.asarray(locus["Mr"]).reshape(feH.size, tLoc.size)
    params = SimpleNamespace(FeH1d=feH, Mr1d=tLoc, MrTrueTable=table)
    return feH, lt.buildSegmentData(params), float(np.min(locus["Mr"])), float(np.max(locus["Mr"]))


def pixelMaps(pixel, order, catalog, radius, maxStars, feH, segments, mrMin, mrMax):
    """The prior maps of one pixel: model stars of a cone at its centre, binned in r and smoothed.

    Returns an empty list when the cone holds no usable model stars, leaving the pixel out of the catalog.
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
    model = stars[
        (stars["label"] != 9) & stars["FeH"].between(feH[0], feH[-1]) & stars["Mr"].between(mrMin, mrMax)
    ].copy()
    del stars
    if len(model) < 3:
        return None
    model = lt.assignTLocPartition(model, segments, feH)
    model = model[model["tLoc"].notna()]
    if len(model) < 3:
        return None

    bc = getBayesConstants()
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
    maps, made = None, np.zeros(rGrid.size, dtype=bool)
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
    order = np.argsort(np.abs(np.arange(rGrid.size)[:, None] - np.where(made)[0][None, :]), axis=1)[:, 0]
    maps = maps[np.where(made)[0][order]]
    return int(pixel), maps, xGrid[0], yGrid[:, 0] if yGrid.ndim > 1 else yGrid


def startWorker(order, catalog, radius, maxStars):
    """Read the locus once per process: it is the same for every pixel and costs more than a pixel does."""
    feH, segments, mrMin, mrMax = locusAxes()
    WORKER.update(
        order=order,
        catalog=catalog,
        radius=radius,
        maxStars=maxStars,
        axes=(feH, segments, mrMin, mrMax),
    )


def buildPixel(pixel):
    """Pool entry point: the maps of one pixel, or None if the model catalog cannot serve it."""
    try:
        return pixelMaps(
            int(pixel),
            WORKER["order"],
            WORKER["catalog"],
            WORKER["radius"],
            WORKER["maxStars"],
            *WORKER["axes"],
        )
    except Exception:  # a pixel the model catalog cannot serve must not take the whole build down
        return None


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trilegal", required=True, help="TRILEGAL model catalog (HATS)")
    ap.add_argument("--out", required=True, help="file to write the prior maps to (npz)")
    ap.add_argument("--footprint", default="", help="sky map of the object catalog (skymap.N.fits)")
    ap.add_argument("--order", type=int, default=5, help="HEALPix order of the maps (5 is 1.8 degrees)")
    ap.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="radius of the cone of model stars per pixel; a circle of the pixel's area by default",
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
    args = ap.parse_args()

    pixels = footprintPixels(args.footprint, args.order)
    area = 4 * np.pi * (180 / np.pi) ** 2 / (12 * 4**args.order)
    radius = args.radius or float(np.sqrt(area / np.pi))
    print(f"{pixels.size} pixels of order {args.order} ({area:.2f} deg2), cone radius {radius:.2f} deg")

    built, maps, axes = [], [], None
    with mp.Pool(
        args.processes,
        initializer=startWorker,
        initargs=(args.order, args.trilegal, radius, args.max_stars),
        maxtasksperchild=args.max_tasks,
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(buildPixel, pixels, chunksize=1), start=1):
            if result is not None:
                pixel, cube, xGrid, yGrid = result
                built.append(pixel)
                maps.append(cube)
                axes = (xGrid, yGrid)
            if done % 100 == 0 or done == pixels.size:
                print(f"  {done}/{pixels.size} pixels, {len(built)} built", flush=True)
    if not built:
        raise SystemExit("no prior maps were built: check the footprint and the model catalog")

    order = np.argsort(built)
    cube = np.stack([maps[i] for i in order]).astype(np.float32)
    index = np.full(12 * 4**args.order, -1, dtype=np.int32)
    index[np.asarray(built)[order]] = np.arange(len(built), dtype=np.int32)
    bc = getBayesConstants()
    np.savez(
        args.out,
        kde=cube,
        rmag=np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"]),
        xGrid=axes[0].astype(np.float64),
        yGrid=axes[1].astype(np.float64),
        index=index,
        order=args.order,
    )
    print(
        f"{args.out}: {len(built)} of {pixels.size} pixels, maps {cube.shape}, "
        f"{cube.nbytes / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
