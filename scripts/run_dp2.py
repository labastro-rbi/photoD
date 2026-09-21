"""Run photoD on Rubin DP2: point sources from the object catalog, the DP2 locus, TRILEGAL prior maps, a
pool of processes over the partition files, results written as a HATS catalog.

  python scripts/run_dp2.py --catalog /path/to/rubin_dp2/object_collection --priors /path/to/priors.npz \\
                            --out /path/to/results --name dp2_photod

Options: --cone RA DEC RADIUS_DEG to run a piece of sky, --workers for the processes (they share the GPUs
between them) and --batch-size for the JAX setup, --chunk for how many partitions a process handles before it
is replaced, --floor for the colour-error floor (0.03 mag), --no-dust-map to run the flat A_r prior,
--dust-curves to use a 3D dust map as the A_r prior (scripts/make_dust_curves.py), which matters at low
Galactic latitude, and --ar-max for the top of the A_r grid, which has to be above the extinction of the
field.

Input columns (DP2 object table): coord_ra, coord_dec, objectId, <band>_psfFlux and _psfFluxErr for ugrizy,
refExtendedness, ebv. Point sources are refExtendedness == 0 with r between 16.5 and 23.5 and S/N > 10 in r,
> 3 in g and i. A colour whose bands are not both at S/N > 3 is set to 0 with error 9.99 and carries no
weight. The dust-map A_r is 2.37 ebv (SFD with the Schlafly & Finkbeiner 2011 recalibration) and bounds the
A_r prior; where the 3D map has measured the column through the disc, the smaller of the two is the bound,
which matters towards the bulge, where the 2D map integrates to infinity and reaches tens of magnitudes.
"""

import argparse
import multiprocessing as mp
import os
import shutil
from pathlib import Path

# XLA's autotuner compiles and times dozens of variants of every kernel the first time it meets one, which
# costs minutes of CPU per worker with the GPU sitting idle and buys this fit nothing, since its cost is in
# one hand-written kernel rather than in library matrix multiplications. Both variables have to be set before
# JAX starts; the worker processes inherit them.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import lsdb  # noqa: E402
import nested_pandas as npd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from photod.bayes import getEstimatesMeta, makeBayesEstimates3d  # noqa: E402
from photod.locus import LSSTsimsLocus, get3DmodelList, make3DlocusList, subsampleLocusData  # noqa: E402
from photod.parameters import GlobalParams  # noqa: E402
from photod.priors import priorGridFromMaps  # noqa: E402

LOCUS = Path(__file__).resolve().parents[1] / "data" / "LSSTlocus_10Gyr_DP2.txt"
BANDS = "ugrizy"
COLORS = ("ug", "gr", "ri", "iz", "zy")
RAW_COLUMNS = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "ebv"] + [
    f"{b}_{c}" for b in BANDS for c in ("psfFlux", "psfFluxErr")
]
FIT_COLUMNS = ["objectId", "ra", "dec", "rmag"] + [c + s for c in COLORS for s in ("", "Err")]
INPUT_COLUMNS = RAW_COLUMNS  # kept for anything importing the old name
PRIOR_DECADES = 8.0  # how far below its own peak a compact prior map is kept
PRIORS = {}
PARAMS = {}
CURVES = {}
WORK = {}


def catalogColumns(url):
    """Every column the catalog holds, which is not the same as the ones it hands out by default."""
    base = lsdb.open_catalog(url).hc_structure.catalog_base_dir
    try:
        return list(pq.read_schema(f"{base}/dataset/_common_metadata").names)
    except Exception:
        return list(lsdb.open_catalog(url).columns)


def inputColumns(available, arColumn):
    """What to read from the catalog: one that already carries colours needs none of the fluxes."""
    if set(FIT_COLUMNS) <= set(available):
        return FIT_COLUMNS + ([arColumn] if arColumn in available else [])
    missing = [c for c in RAW_COLUMNS if c not in available]
    if missing:
        raise SystemExit(
            f"the catalog has neither the colours nor the fluxes to make them: missing {missing}"
        )
    return RAW_COLUMNS


def prepareStars(df, dustIndex=None, nside=0, arTotal=None, arColumn=""):
    """The stars of one partition with the columns the fit reads.

    A catalog prepared beforehand already carries the colours and their errors, and is passed through; one
    straight from the survey has its point sources selected and its colours built out of the PSF fluxes.
    """
    if set(FIT_COLUMNS) <= set(df.columns):
        out = pd.DataFrame({c: df[c].to_numpy() for c in FIT_COLUMNS})
        if arColumn and arColumn in df.columns:
            out["Ar"] = df[arColumn].to_numpy(dtype=float)
        elif "ebv" in df.columns:
            out["Ar"] = 2.37 * df["ebv"].to_numpy(dtype=float)
        else:
            raise SystemExit("the prepared catalog carries no extinction column: name it with --ar-column")
    else:
        out = starsFromFluxes(df)
    return withDust(out, dustIndex, nside, arTotal)


def starsFromFluxes(df):
    """Magnitudes, colours and errors of the point sources in one partition of the survey object table."""
    flux = {b: df[f"{b}_psfFlux"].to_numpy(dtype=float, na_value=np.nan) for b in BANDS}
    err = {b: df[f"{b}_psfFluxErr"].to_numpy(dtype=float, na_value=np.nan) for b in BANDS}
    snr = {b: flux[b] / err[b] for b in BANDS}
    mag = {b: -2.5 * np.log10(np.where(flux[b] > 0, flux[b], np.nan)) + 31.4 for b in BANDS}
    magErr = {b: 1.0857 / snr[b] for b in BANDS}
    keep = (
        (df["refExtendedness"].to_numpy(dtype=float, na_value=np.nan) == 0)
        & (mag["r"] > 16.5)
        & (mag["r"] < 23.5)
        & (snr["r"] > 10)
        & (snr["g"] > 3)
        & (snr["i"] > 3)
    )
    out = pd.DataFrame(
        {
            "objectId": df["objectId"].to_numpy()[keep],
            "ra": df["coord_ra"].to_numpy(dtype=float)[keep],
            "dec": df["coord_dec"].to_numpy(dtype=float)[keep],
            "rmag": mag["r"][keep],
            "Ar": 2.37 * df["ebv"].to_numpy(dtype=float, na_value=np.nan)[keep],
        }
    )
    for c in COLORS:
        a, b = c
        useful = ((snr[a] > 3) & (snr[b] > 3))[keep]
        out[c] = np.where(useful, (mag[a] - mag[b])[keep], 0.0)
        out[c + "Err"] = np.where(useful, np.hypot(magErr[a], magErr[b])[keep], 9.99)
    return out


def withDust(out, dustIndex, nside, arTotal):
    """The sightline each star sits on in the 3D dust map, and the bound its column puts on the extinction."""
    if dustIndex is None:
        out["dustIndex"] = np.zeros(len(out), dtype=np.int32)
    else:
        import cdshealpix
        from astropy.coordinates import Latitude, Longitude

        pixel = cdshealpix.nested.lonlat_to_healpix(
            Longitude(out.ra.to_numpy(), unit="deg"),
            Latitude(out.dec.to_numpy(), unit="deg"),
            int(np.log2(nside)),
        )
        row = dustIndex[np.asarray(pixel)]
        out["dustIndex"] = np.maximum(row, 0).astype(np.int32)
        if arTotal is not None:
            # The 2D map integrates the dust to infinity, which towards the bulge is tens of magnitudes and
            # says nothing about a star in front of it. Where a 3D map has measured the column out past the
            # far side of the disc, that measurement is the bound, and the larger 2D value is dropped.
            measured = np.where(row >= 0, arTotal[np.maximum(row, 0)], 0.0)
            out["Ar"] = np.where(measured > 0, np.minimum(out["Ar"].to_numpy(), measured), out["Ar"])
    return npd.NestedFrame(out)


def globalParameters(floor, useDustMap, curves=None, arMax=5.0):
    """The fit setup: the DP2 locus on the tLoc grid, the colour-error floor, the A_r prior.

    The A_r grid has to reach the extinction of the field: the standard "ArLarge" grid stops at 2.5 mag, and a
    star whose A_r is above the top of the grid has it pinned there, which throws its distance out with it.
    """
    locus = LSSTsimsLocus(fixForStripe82=False, datafile=str(LOCUS), colnames=["tLoc", "Mr", "FeH", *COLORS])
    locusData = subsampleLocusData(locus, kMr=1, kFeH=1, yLabel="tLoc")
    ArGridList, locus3DList = get3DmodelList(locusData, COLORS, yLabel="tLoc")
    ArGridList["ArLarge"] = np.arange(0, arMax + 1e-9, 0.02)
    locus3DList["ArLarge"] = make3DlocusList(locusData, COLORS, [ArGridList["ArLarge"]], yLabel="tLoc")[0]
    dust = {}
    if curves is not None:
        dust = dict(ArCurves=curves["shapes"], ArCurveMu=curves["mu"], ArCurveIndexColumn="dustIndex")
    return GlobalParams(
        COLORS,
        locusData,
        ArGridList,
        locus3DList,
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArMapColumn="Ar" if useDustMap else None,
        colorErrFloor=floor,
        **dust,
    )


def priorPixels(ra, dec, order):
    """The HEALPix pixel of the prior maps that each star falls in."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    return np.asarray(
        cdshealpix.nested.lonlat_to_healpix(
            Longitude(np.asarray(ra), unit="deg"), Latitude(np.asarray(dec), unit="deg"), order
        )
    )


def loadCurves(path):
    """The dust curves, read once per worker process."""
    if path not in CURVES:
        CURVES[path] = dict(np.load(path)) if path else {}
    return CURVES[path]


def mapFile(path):
    """Where the unpacked maps live: beside the file they came out of."""
    return Path(path).with_suffix(".kde.npy")


def priorMaps(data):
    """The maps as densities, undoing the quantisation a compact file stores them with.

    A compact file keeps the log of each map relative to its own peak, to a byte over eight decades, on every
    other point of the grid. The maps are smoothed densities, so that loses about 0.008 dex where the prior
    has any weight, against a chi2 that runs to hundreds.
    """
    kde = data["kde"]
    if kde.dtype != np.uint8:
        return kde
    levels = float(np.iinfo(np.uint8).max)
    out = 10 ** (kde.astype(np.float32) / levels * PRIOR_DECADES - PRIOR_DECADES)
    out *= data["kdeScale"][:, :, None, None]
    return np.where(kde == 0, 0.0, out).astype(np.float32)


def unpackPriors(path):
    """Write the maps out once as a plain array the workers can map.

    They are two gigabytes and every worker needs a different handful of sightlines out of them. Read as a
    file each worker holds its own copy, which at eight workers is seventeen gigabytes and, if the file is
    compressed, a minute of unpacking; mapped they share one copy and take about thirty megabytes each.
    """
    cache = mapFile(path)
    if cache.exists() and cache.stat().st_mtime >= Path(path).stat().st_mtime:
        return
    with np.load(path) as data:
        np.save(cache, priorMaps(data))


def loadPriors(path):
    """The prior maps of a run, mapped rather than read, once per worker process."""
    if path not in PRIORS:
        with np.load(path) as data:
            held = {name: data[name] for name in data.files if name not in ("kde", "kdeScale")}
            cache = mapFile(path)
            held["kde"] = np.load(cache, mmap_mode="r") if cache.exists() else priorMaps(data)
        PRIORS[path] = held
    return PRIORS[path]


def workerDevice(nDevices):
    """Which GPU this worker uses, by its place in the pool, so that the devices are shared evenly."""
    if nDevices < 2:
        return 0
    identity = getattr(mp.current_process(), "_identity", None)
    return ((identity[0] - 1) if identity else os.getpid()) % nDevices


def loadParams(setup):
    """The fit setup, built once per worker process rather than sent to it.

    The reddened locus of a wide A_r grid is a couple of gigabytes, and shipping that through the scheduler
    to every worker costs more memory in flight than building it where it is used.
    """
    if setup not in PARAMS:
        floor, useDustMap, curvePath, arMax = setup
        PARAMS[setup] = globalParameters(floor, useDustMap, np.load(curvePath) if curvePath else None, arMax)
    return PARAMS[setup]


def separation(ra, dec, ra0, dec0):
    """Angle in degrees between each position and one centre."""
    a, d, a0, d0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    cosine = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def fitAndWrite(source, pixel, base, priorPath, curvePath, setup, batchSize, cone, arColumn, columns):
    """Read one partition file, fit its stars and write them where the partition's HEALPix pixel belongs.

    The task carries a path rather than a piece of a catalog. Handing the workers pieces of a catalog sends
    each of them the structure of the whole survey along with it, which on DP2 is gigabytes per worker before
    a single star is read, and the results are far larger than the stars they came from, so collecting them
    for a separate writing step holds the survey in memory as well.
    """
    from hats.io.paths import pixel_catalog_file
    from hats.pixel_math import HealpixPixel

    frame = pq.read_table(source, columns=columns).to_pandas()
    curves = loadCurves(curvePath)
    stars = prepareStars(
        frame,
        dustIndex=curves.get("index"),
        nside=int(curves["nside"]) if curves else 0,
        arTotal=curves.get("total"),
        arColumn=arColumn,
    )
    del frame
    if cone is not None and len(stars):
        ra, dec, radius = cone
        stars = stars[separation(stars["ra"].to_numpy(), stars["dec"].to_numpy(), ra, dec) <= radius]
    estimates = fitPartition(stars, priorPath, setup, batchSize)
    if not len(estimates):
        return 0
    path = pixel_catalog_file(base, HealpixPixel(pixel.order, pixel.pixel))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates).to_parquet(path, index=False)
    return len(estimates)


def fitPartition(partition, priorPath, setup, batchSize):
    """One partition of stars, each group of them fitted with the prior maps of the sky pixel it lies in.

    Looking the sightline up beats joining against a catalog of maps: a join is between two pixel trees of
    different depth and quietly keeps only one of the star partitions that share a map.
    """
    empty = npd.NestedFrame(getEstimatesMeta(computeMrTrue=True).reset_index(drop=True))
    if not len(partition):
        return empty
    globalParams = loadParams(setup)
    priors = loadPriors(priorPath)
    index, order = priors["index"], int(priors["order"])
    row = index[priorPixels(partition["ra"].to_numpy(), partition["dec"].to_numpy(), order)]
    device = jax.devices()[workerDevice(len(jax.devices()))]
    pieces = []
    for value in np.unique(row[row >= 0]):
        stars = partition[row == value]
        grid = priorGridFromMaps(
            priors["kde"][value], priors["rmag"], priors["xGrid"], priors["yGrid"], globalParams
        )
        with jax.default_device(device):
            estimates, _ = makeBayesEstimates3d(
                stars, jax.numpy.array(list(grid.values())), globalParams, batchSize=batchSize
            )
        pieces.append(estimates)
    if not pieces:
        return empty
    return npd.NestedFrame(pd.concat(pieces, ignore_index=True))


def startWorker(base, priorPath, curvePath, setup, batchSize, cone, arColumn, columns):
    """What every partition of this run needs, held once per worker process."""
    WORK.update(
        base=base,
        priorPath=priorPath,
        curvePath=curvePath,
        setup=setup,
        batchSize=batchSize,
        cone=cone,
        arColumn=arColumn,
        columns=columns,
    )


def runPartition(source):
    """Pool entry point: one partition, or nothing if it holds no stars the fit can use."""
    pixel, path = source
    return fitAndWrite(
        path,
        pixel,
        WORK["base"],
        WORK["priorPath"],
        WORK["curvePath"],
        WORK["setup"],
        WORK["batchSize"],
        WORK["cone"],
        WORK["arColumn"],
        WORK["columns"],
    )


def partitionFiles(catalog):
    """Every partition the run covers, as its HEALPix pixel and the parquet file holding it."""
    from hats.io.paths import pixel_catalog_file

    root = catalog.hc_structure.catalog_base_dir
    return [(pixel, pixel_catalog_file(root, pixel)) for pixel in catalog.hc_structure.get_healpix_pixels()]


def writeCatalogMetadata(base, name, pixels, total):
    """The HATS metadata beside the parquet files, so that the result reads back as a catalog."""
    try:
        from hats.catalog import PartitionInfo, TableProperties
        from hats.io import write_parquet_metadata

        written = [p for p in pixels if (Path(str(pixelFile(base, p)))).exists()]
        PartitionInfo.from_healpix(written).write_to_file(catalog_path=base)
        TableProperties(
            catalog_name=name,
            catalog_type="object",
            total_rows=total,
            ra_column="ra",
            dec_column="dec",
        ).to_properties_file(base)
        write_parquet_metadata(base)
    except Exception as error:  # the parquet files are the result; the index can be rebuilt from them
        print(f"the parquet files are written but the catalog index is not: {type(error).__name__}: {error}")


def pixelFile(base, pixel):
    """Where one HEALPix pixel's parquet file goes."""
    from hats.io.paths import pixel_catalog_file
    from hats.pixel_math import HealpixPixel

    return pixel_catalog_file(base, HealpixPixel(pixel.order, pixel.pixel))


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="DP2 object collection (HATS)")
    ap.add_argument("--priors", required=True, help="TRILEGAL prior maps (scripts/make_priors.py)")
    ap.add_argument("--out", required=True, help="directory for the result catalog")
    ap.add_argument("--name", default="dp2_photod")
    ap.add_argument("--cone", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_DEG"))
    ap.add_argument("--floor", type=float, default=0.03, help="colour-error floor in magnitudes")
    ap.add_argument(
        "--ar-column",
        default="Ar",
        help="extinction column of a catalog prepared beforehand; from ebv when the fluxes are read instead",
    )
    ap.add_argument(
        "--ar-max",
        type=float,
        default=5.0,
        help="top of the A_r grid; keep it above 1.3 A_r(map) + 0.1 of the dustiest star in the field",
    )
    ap.add_argument("--no-dust-map", action="store_true", help="flat A_r prior instead of the dust-map bound")
    ap.add_argument(
        "--dust-curves",
        default="",
        help="npz from make_dust_curves.py: a 3D dust map as the A_r prior, worth having at |b| < 10",
    )
    ap.add_argument("--workers", type=int, default=1, help="processes, which share the GPUs between them")
    ap.add_argument(
        "--chunk",
        type=int,
        default=400,
        help="partitions a worker fits before it is replaced, which bounds the memory of a survey-wide run",
    )
    ap.add_argument("--overwrite", action="store_true", help="replace an existing result of the same name")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="stars per JAX call; a few hundred is the fastest, larger batches spill and slow down",
    )
    args = ap.parse_args()

    search = (
        lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=3600 * args.cone[2])
        if args.cone
        else None
    )
    columns = inputColumns(catalogColumns(args.catalog), args.ar_column)
    objects = lsdb.open_catalog(args.catalog, columns=columns, search_filter=search)
    sources = [(pixel, str(path)) for pixel, path in partitionFiles(objects)]
    print(f"reading {'prepared colours' if 'ug' in columns else 'PSF fluxes'} from {args.catalog}")
    cone = tuple(args.cone) if args.cone else None
    setup = (args.floor, not args.no_dust_map, args.dust_curves, args.ar_max)
    if args.dust_curves:
        curves = np.load(args.dust_curves)
        bounded = int((curves["total"] > 0).sum()) if "total" in curves.files else 0
        print(
            f"3D dust prior from {args.dust_curves}: {len(curves['shapes'])} sightlines, "
            f"{bounded} of them with a measured total column to bound the extinction"
        )

    unpackPriors(args.priors)
    base = Path(args.out) / args.name
    if base.exists() and args.overwrite:
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    print(f"{len(sources)} partitions to fit", flush=True)

    total, done = 0, 0
    with mp.Pool(
        args.workers,
        initializer=startWorker,
        initargs=(base, args.priors, args.dust_curves, setup, args.batch_size, cone, args.ar_column, columns),
        maxtasksperchild=args.chunk,
    ) as pool:
        for count in pool.imap_unordered(runPartition, sources, chunksize=1):
            total += count
            done += 1
            if done % 200 == 0 or done == len(sources):
                print(f"  {done}/{len(sources)} partitions, {total} stars", flush=True)

    writeCatalogMetadata(base, args.name, [p for p, _ in sources], total)
    print(f"written {base}: {total} stars")


if __name__ == "__main__":
    main()
