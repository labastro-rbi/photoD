"""Run photoD on Rubin DP2: point sources from the object catalog, the DP2 locus, TRILEGAL priors in HATS,
one lsdb merge_map over the sky, results written as a HATS catalog.

  python scripts/run_dp2.py --catalog /path/to/rubin_dp2/object_collection --priors /path/to/priors.npz \\
                            --out /path/to/results --name dp2_photod

Options: --cone RA DEC RADIUS_DEG to run a piece of sky, --workers and --batch-size for the dask/JAX setup,
--floor for the colour-error floor (0.03 mag), --no-dust-map to run the flat A_r prior, --dust-curves to use a
3D dust map as the A_r prior (scripts/make_dust_curves.py), which matters at low Galactic latitude, and
--ar-max for the top of the A_r grid, which has to be above the extinction of the field.

Input columns (DP2 object table): coord_ra, coord_dec, objectId, <band>_psfFlux and _psfFluxErr for ugrizy,
refExtendedness, ebv. Point sources are refExtendedness == 0 with r between 16.5 and 23.5 and S/N > 10 in r,
> 3 in g and i. A colour whose bands are not both at S/N > 3 is set to 0 with error 9.99 and carries no
weight. The dust-map A_r is 2.37 ebv (SFD with the Schlafly & Finkbeiner 2011 recalibration) and bounds the
A_r prior; where the 3D map has measured the column through the disc, the smaller of the two is the bound,
which matters towards the bulge, where the 2D map integrates to infinity and reaches tens of magnitudes.
"""

import argparse
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path

# XLA's autotuner compiles and times dozens of variants of every kernel the first time it meets one, which
# costs minutes of CPU per worker with the GPU sitting idle and buys this fit nothing, since its cost is in
# one hand-written kernel rather than in library matrix multiplications. Both variables have to be set before
# JAX starts; the dask workers inherit them.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Partitions of the object catalog are hundreds of megabytes each and are freed as soon as the stars have
# been taken out of them, but by default the allocator keeps the space rather than returning it, so a worker
# that has read a few hundred partitions looks far larger than the work it is holding.
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

import dask  # noqa: E402
import jax  # noqa: E402
import lsdb  # noqa: E402
import nested_pandas as npd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dask.distributed import Client, get_worker  # noqa: E402

from photod.bayes import getEstimatesMeta, makeBayesEstimates3d  # noqa: E402
from photod.locus import LSSTsimsLocus, get3DmodelList, make3DlocusList, subsampleLocusData  # noqa: E402
from photod.parameters import GlobalParams  # noqa: E402
from photod.priors import priorGridFromMaps  # noqa: E402

LOCUS = Path(__file__).resolve().parents[1] / "data" / "LSSTlocus_10Gyr_DP2.txt"
BANDS = "ugrizy"
COLORS = ("ug", "gr", "ri", "iz", "zy")
INPUT_COLUMNS = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "ebv"] + [
    f"{b}_{c}" for b in BANDS for c in ("psfFlux", "psfFluxErr")
]
PRIORS = {}
PARAMS = {}


def starsMeta():
    """Empty frame with the columns and types prepareStars returns (lsdb needs it to build the graph)."""
    cols = {"objectId": np.int64, "ra": float, "dec": float, "rmag": float, "Ar": float}
    for c in COLORS:
        cols[c], cols[c + "Err"] = float, float
    cols["dustIndex"] = np.int32
    return npd.NestedFrame({k: pd.Series([], dtype=v) for k, v in cols.items()})


def prepareStars(df, dustIndex=None, nside=0, arTotal=None):
    """Magnitudes, colours and errors of the point sources in one partition of the object table."""
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


def loadPriors(path):
    """The prior maps, read once per worker process and kept for the partitions that follow."""
    if path not in PRIORS:
        PRIORS[path] = dict(np.load(path))
    return PRIORS[path]


def workerDevice(nDevices):
    """Which GPU this worker uses: its name, which a replaced worker keeps, so the share stays even."""
    if nDevices < 2:
        return 0
    try:
        name = get_worker().name
        return int(name) % nDevices if str(name).lstrip("-").isdigit() else abs(hash(name)) % nDevices
    except Exception:
        return os.getpid() % nDevices


def loadParams(setup):
    """The fit setup, built once per worker process rather than sent to it.

    The reddened locus of a wide A_r grid is a couple of gigabytes, and shipping that through the scheduler
    to every worker costs more memory in flight than building it where it is used.
    """
    if setup not in PARAMS:
        floor, useDustMap, curvePath, arMax = setup
        PARAMS[setup] = globalParameters(floor, useDustMap, np.load(curvePath) if curvePath else None, arMax)
    return PARAMS[setup]


def fitAndWrite(partition, pixel, base, priorPath, setup, batchSize):
    """Fit one partition and write it where its HEALPix pixel belongs, returning only how many stars it held.

    The fit writes its own output rather than handing it back: results are far larger than the stars they
    came from, and a scheduler that collects them all before a separate writing step runs holds the survey
    in memory.
    """
    from hats.io.paths import pixel_catalog_file
    from hats.pixel_math import HealpixPixel

    estimates = fitPartition(partition, priorPath, setup, batchSize)
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
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--chunk",
        type=int,
        default=400,
        help="partitions a set of workers handles before it is replaced, which bounds the memory of a run",
    )
    ap.add_argument(
        "--worker-memory",
        default="16GB",
        help="memory a worker may hold before it is replaced, which is the machine's guarantee, not ours",
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
    objects = lsdb.open_catalog(args.catalog, columns=INPUT_COLUMNS, search_filter=search)
    curves = np.load(args.dust_curves) if args.dust_curves else None
    if curves is None:
        prepare = prepareStars
    else:
        total = curves["total"] if "total" in curves.files else None
        prepare = partial(prepareStars, dustIndex=curves["index"], nside=int(curves["nside"]), arTotal=total)
        bounded = 0 if total is None else int((total > 0).sum())
        print(
            f"3D dust prior from {args.dust_curves}: {len(curves['shapes'])} sightlines, "
            f"{bounded} of them with a measured total column to bound the extinction"
        )
    stars = objects.map_partitions(prepare, meta=starsMeta())
    setup = (args.floor, not args.no_dust_map, args.dust_curves, args.ar_max)

    # Reading a partition is an order of magnitude faster than fitting one, so a scheduler that is free to
    # run ahead reads the whole survey into memory while the GPUs work through the first few partitions.
    # Holding it to one unfinished read per worker keeps the memory flat.
    # The memory a worker cannot release is invisible to dask as anything it can spill, so its usual answer
    # to a large worker, pause it and write its data out, leaves the worker asleep holding memory it will
    # never give back and the run stops. Let a worker run until it is over its limit and replaced instead.
    dask.config.set(
        {
            "distributed.scheduler.worker-saturation": 1.0,
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
        }
    )
    # no dashboard: it profiles every worker continuously, and over a survey-sized graph those buffers grow
    # faster than the fit does
    base = Path(args.out) / args.name
    if base.exists() and args.overwrite:
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    pixels = stars.hc_structure.get_healpix_pixels()
    parts = stars.to_delayed()
    print(f"{len(parts)} partitions to fit", flush=True)

    total, done = 0, 0
    for start in range(0, len(parts), args.chunk):
        group = range(start, min(start + args.chunk, len(parts)))
        # a fresh set of workers for each chunk: reading a partition of the object catalog leaves memory
        # behind that no amount of releasing on our side recovers, so a worker that reads a few hundred of
        # them grows past any limit worth setting. Replacing them costs the seconds it takes to rebuild the
        # fit setup, and is what keeps a survey-wide run flat.
        with Client(
            n_workers=args.workers,
            threads_per_worker=1,
            memory_limit=args.worker_memory,
            dashboard_address=None,
            local_directory=tempfile.mkdtemp(prefix="dask-"),
        ):
            counts = dask.compute(
                *[
                    dask.delayed(fitAndWrite)(parts[i], pixels[i], base, args.priors, setup, args.batch_size)
                    for i in group
                ]
            )
        total += int(sum(counts))
        done += len(counts)
        print(f"  {done}/{len(parts)} partitions, {total} stars", flush=True)

    writeCatalogMetadata(base, args.name, pixels, total)
    print(f"written {base}: {total} stars")


if __name__ == "__main__":
    main()
