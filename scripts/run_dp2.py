"""Run photoD on Rubin DP2: point sources from the object catalog, the DP2 locus, TRILEGAL priors in HATS,
one lsdb merge_map over the sky, results written as a HATS catalog.

  python scripts/run_dp2.py --catalog /path/to/rubin_dp2/object_collection --priors /path/to/prior_maps \\
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
from functools import partial
from pathlib import Path

# XLA's autotuner compiles and times dozens of variants of every kernel the first time it meets one, which
# costs minutes of CPU per worker with the GPU sitting idle and buys this fit nothing, since its cost is in
# one hand-written kernel rather than in library matrix multiplications. Both variables have to be set before
# JAX starts; the dask workers inherit them.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import lsdb  # noqa: E402
import nested_pandas as npd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dask.distributed import Client, get_worker  # noqa: E402

from photod.bayes import getEstimatesMeta, makeBayesEstimates3d  # noqa: E402
from photod.locus import LSSTsimsLocus, get3DmodelList, make3DlocusList, subsampleLocusData  # noqa: E402
from photod.parameters import GlobalParams  # noqa: E402
from photod.priors import initializePriorGrid  # noqa: E402

LOCUS = Path(__file__).resolve().parents[1] / "data" / "LSSTlocus_10Gyr_DP2.txt"
BANDS = "ugrizy"
COLORS = ("ug", "gr", "ri", "iz", "zy")
INPUT_COLUMNS = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "ebv"] + [
    f"{b}_{c}" for b in BANDS for c in ("psfFlux", "psfFluxErr")
]


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


def fitPartition(
    partition, mapPartition, partitionPixel, mapPixel, globalParams, workerDevices, batchSize, **kwargs
):
    """merge_map worker: the prior maps of one sky pixel, then the fit of its stars on this worker."""
    priorGrid = initializePriorGrid(mapPartition, globalParams)
    device = jax.devices()[workerDevices[get_worker().id]]
    with jax.default_device(device):
        priorGrid = jax.numpy.array(list(priorGrid.values()))
        estimates, _ = makeBayesEstimates3d(partition, priorGrid, globalParams, batchSize=batchSize)
    return npd.NestedFrame(estimates)


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="DP2 object collection (HATS)")
    ap.add_argument(
        "--priors", required=True, help="TRILEGAL prior maps (HATS, one row per r bin and sky pixel)"
    )
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
    priors = lsdb.open_catalog(args.priors)
    params = globalParameters(args.floor, not args.no_dust_map, curves, args.ar_max)

    with Client(n_workers=args.workers) as client:
        workerIds = sorted(client.run(lambda dask_worker: dask_worker.id).values())
        nDevices = jax.device_count()
        devices = {wid: i % nDevices for i, wid in enumerate(workerIds)}
        result = stars.merge_map(
            priors,
            fitPartition,
            globalParams=client.scatter(params),
            workerDevices=devices,
            batchSize=args.batch_size,
            meta=getEstimatesMeta(computeMrTrue=True),
        )
        result.write_catalog(base_catalog_path=args.out, catalog_name=args.name, overwrite=True)
    print(f"written {Path(args.out) / args.name}")


if __name__ == "__main__":
    main()
