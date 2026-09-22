"""TRILEGAL prior maps for a photoD run: p([Fe/H], tLoc | r) on a HEALPix grid, in one file.

  python scripts/make_priors.py --trilegal /path/to/TRILEGAL_cluster_v08 --out priors_dp2.npz \\
                                --footprint /path/to/rubin_dp2/object_collection/object_lc/skymap.6.fits

One map per r bin of getBayesConstants() and per HEALPix pixel of the given order, stored as an array with an
index from pixel to row, so that the run looks a sightline up rather than joining two catalogs. The model
stars of a pixel are read from a cone of the same area at its centre and placed on the tLoc axis of the locus
the fit will use, which is what makes the maps and the fit comparable.

Only pixels of the footprint are built. The footprint comes from the sky map of the object catalog, either a
sparse table of PIXEL and VALUE or a full HEALPix array; without one the whole sky is built. A pixel of the
footprint that ends up with no maps is a hole the run drops the stars of, so the build reports every one of
them and fails unless --allow-missing says a partial file is wanted.

The maps here leave out the model stars of TRILEGAL label 9, the white dwarfs and post-AGB stars, while
photod.priors.dumpPriorMaps_testing keeps every label. The difference is deliberate: a white dwarf has no
place on the locus the fit uses, so assigning it a tLoc puts it wherever the nearest segment of the locus
happens to be, which at faint r is enough of them to leave a ridge in the prior that no fitted star belongs
on. The in-package function is kept as it is because the published maps were made with it.
"""

import argparse
import json
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
MAX_REPORTED = 20  # pixels a report lists one by one before it only counts them
WORKER = {}


PRIOR_DECADES = 8.0  # how far below its own peak a compact map is kept


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


def writePriors(out, cube, rGrid, xGrid, yGrid, index, order, small):
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
        constants=json.dumps(getBayesConstants()),
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
    writePriors(out, np.stack(cube), rGrid, xGrid, yGrid, index, order, small)


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

    Returns the maps and their two axes, or None when the cone holds no usable model stars, which leaves the
    pixel out of the file and the stars of that sightline out of the run.
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
    nearest = np.argsort(np.abs(np.arange(rGrid.size)[:, None] - np.where(made)[0][None, :]), axis=1)[:, 0]
    maps = maps[np.where(made)[0][nearest]]
    return maps, xGrid[0], yGrid[:, 0] if yGrid.ndim > 1 else yGrid


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
    """Pool entry point: one pixel with its maps, None if the catalog cannot serve it, or what it raised.

    A pixel that raises must not take a build of thousands of them down, and must not disappear either: the
    run drops every star of a sightline that has no maps, so the failure is handed back to be reported. It
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
        )
    except Exception as error:
        return int(pixel), f"{type(error).__name__}: {error}"


def reportHoles(failed, empty, allowMissing):
    """List the pixels of the footprint that ended up with no maps, and stop unless a hole was intended.

    The run looks a star's sightline up in the index of the file and drops the stars of a pixel that is not
    in it without a word, so a hole here is stars missing from the results rather than a rougher prior. The
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
            "the run drops every star of a sightline that has none; find the cause or pass --allow-missing "
            "to take the file as it stands"
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
    args = ap.parse_args()

    if args.from_catalog:
        fromCatalog(args.from_catalog, args.out, args.compact)
        return
    if not args.trilegal:
        raise SystemExit("give either --trilegal to build the maps or --from-catalog to convert them")

    pixels = footprintPixels(args.footprint, args.order)
    area = 4 * np.pi * (180 / np.pi) ** 2 / (12 * 4**args.order)
    radius = args.radius or float(np.sqrt(area / np.pi))
    print(f"{pixels.size} pixels of order {args.order} ({area:.2f} deg2), cone radius {radius:.2f} deg")

    built, maps, axes, failed, empty = [], [], None, [], []
    with mp.Pool(
        args.processes,
        initializer=startWorker,
        initargs=(args.order, args.trilegal, radius, args.max_stars),
        maxtasksperchild=args.max_tasks,
    ) as pool:
        for done, (pixel, result) in enumerate(pool.imap_unordered(buildPixel, pixels, chunksize=1), start=1):
            if isinstance(result, str):
                failed.append((pixel, result))
            elif result is None:
                empty.append(pixel)
            else:
                cube, xGrid, yGrid = result
                built.append(pixel)
                maps.append(cube)
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
    bc = getBayesConstants()
    writePriors(
        args.out,
        cube,
        np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"]),
        axes[0].astype(np.float64),
        axes[1].astype(np.float64),
        index,
        args.order,
        args.compact,
    )
    reportHoles(failed, empty, args.allow_missing)


if __name__ == "__main__":
    main()
