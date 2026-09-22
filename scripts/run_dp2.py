"""Run photoD on Rubin DP2: point sources from the object catalog, the DP2 locus, TRILEGAL prior maps, a
pool of processes over the partition files, results written as a HATS catalog.

  python scripts/run_dp2.py --catalog /path/to/rubin_dp2/object_collection --priors /path/to/priors.npz \\
                            --out /path/to/results --name dp2_photod

Options: --cone RA DEC RADIUS_DEG to run a piece of sky, --workers for the processes (they share the GPUs
between them), --batch-size and --batch-bytes for the JAX setup, the second of them a memory budget for one
batch of one worker, --chunk for how many partitions a process handles before it is replaced, --floor for the
colour-error floor (0.03 mag), --no-dust-map to run the flat A_r prior, which reads no extinction and no 3D
curves at all, --dust-curves to use a 3D dust map as the A_r prior (scripts/make_dust_curves.py), which
matters at low Galactic latitude, and --ar-max for the top of the A_r grid, which has to be above the
extinction of the field.

A partition whose answers are already written is left alone, so a run that stopped part way is finished by
repeating the command; --overwrite starts the result again from nothing. What the run was configured with is
recorded beside the answers, and a resume that asks for anything else is refused rather than allowed to mix
two fits inside one catalog. A partition that fails takes only itself down, and the run ends with a non-zero
status saying how much of the sky is missing.

Input columns (DP2 object table): coord_ra, coord_dec, objectId, <band>_psfFlux and _psfFluxErr for ugrizy,
refExtendedness, ebv. Point sources are refExtendedness == 0 with r between 16.5 and 23.5 and S/N > 10 in r,
> 3 in g and i. A colour whose bands are not both at S/N > 3 is set to 0 with error 9.99 and carries no
weight. The dust-map A_r is 2.37 ebv (SFD with the Schlafly & Finkbeiner 2011 recalibration) and bounds the
A_r prior; where the 3D map has measured the column through the disc, the smaller of the two is the bound,
which matters towards the bulge, where the 2D map integrates to infinity and reaches tens of magnitudes.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from photod.bayes import (  # noqa: E402
    FLAG_NO_PRIOR,
    getEstimatesMeta,
    makeBayesEstimates3d,
    unfittedEstimates,
)
from photod.locus import LSSTsimsLocus, get3DmodelList, make3DlocusList, subsampleLocusData  # noqa: E402
from photod.parameters import GlobalParams  # noqa: E402
from photod.priors import getBayesConstants, priorGridFromMaps  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
LOCUS = DATA / "LSSTlocus_10Gyr_DP2.txt"
PRIOR_FILE = DATA / "priors_dp2.npz"
DUST_FILE = DATA / "dust_dp2.npz"
BANDS = "ugrizy"
COLORS = ("ug", "gr", "ri", "iz", "zy")
RAW_COLUMNS = ["objectId", "coord_ra", "coord_dec", "refExtendedness", "ebv"] + [
    f"{b}_{c}" for b in BANDS for c in ("psfFlux", "psfFluxErr")
]
FIT_COLUMNS = ["objectId", "ra", "dec", "rmag"] + [c + s for c in COLORS for s in ("", "Err")]
INPUT_COLUMNS = RAW_COLUMNS  # kept for anything importing the old name
PRIOR_DECADES = 8.0  # how far below its own peak a compact prior map is kept, for a file that says nothing
FAILURES_REPORTED = 20  # failed partitions named one by one, after which only the count is kept
RUN_FILE = "photod_run.json"  # what the run was configured with, beside the answers it wrote
PARTIAL = ".partial"  # where a partition is written before it is renamed into the dataset
# ProcessPoolExecutor learned to replace a worker in Python 3.11, and the package supports 3.10 as well.
# Checked here rather than at the call so that both paths can be exercised without another interpreter.
WORKERS_REPLACED = sys.version_info >= (3, 11)
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
    """What to read from the catalog: one that already carries colours needs none of the fluxes.

    Everything the run needs of the catalog is settled here, in the parent process. A column that is only
    missed once the pool is running arrives as a worker that crashed, over and over, instead of a sentence.

    arColumn is None when the fit reads no extinction at all, which is what the flat A_r prior does, and a
    prepared catalog is then not asked for a column nothing will look at.
    """
    if set(FIT_COLUMNS) <= set(available):
        if arColumn is None:
            return list(FIT_COLUMNS)
        extinction = [c for c in (arColumn, "ebv") if c in available]
        if not extinction:
            raise SystemExit(
                f"the prepared catalog carries no extinction column: it has neither {arColumn} nor ebv, "
                "so name the one it does have with --ar-column"
            )
        return FIT_COLUMNS + extinction[:1]
    missing = [c for c in RAW_COLUMNS if c not in available]
    if missing:
        raise SystemExit(
            f"the catalog has neither the colours nor the fluxes to make them: missing {missing}"
        )
    return RAW_COLUMNS


def prepareStars(df, curves=None, arColumn=""):
    """The stars of one partition with the columns the fit reads.

    A catalog prepared beforehand already carries the colours and their errors, and is passed through; one
    straight from the survey has its point sources selected and its colours built out of the PSF fluxes.

    arColumn is None where the fit reads no extinction, as inputColumns describes, and no A_r is made.
    """
    if set(FIT_COLUMNS) <= set(df.columns):
        out = pd.DataFrame({c: df[c].to_numpy() for c in FIT_COLUMNS})
        if arColumn is not None:
            if arColumn in df.columns:
                out["Ar"] = df[arColumn].to_numpy(dtype=float)
            elif "ebv" in df.columns:
                out["Ar"] = 2.37 * df["ebv"].to_numpy(dtype=float)
            else:
                raise SystemExit(
                    "the prepared catalog carries no extinction column: name it with --ar-column"
                )
    else:
        out = starsFromFluxes(df)
    return withDust(out, curves)


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


def withDust(out, curves=None):
    """The sightline each star sits on in the 3D dust map, and the bound its column puts on the extinction."""
    if not curves:
        out["dustIndex"] = np.zeros(len(out), dtype=np.int32)
        return npd.NestedFrame(out)
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    pixel = cdshealpix.nested.lonlat_to_healpix(
        Longitude(out.ra.to_numpy(), unit="deg"),
        Latitude(out.dec.to_numpy(), unit="deg"),
        int(curves["order"]),
    )
    # The index of the file holds -1 where no 3D map has data, and readCurves keeps a flat sightline at the
    # end of the table for exactly those stars. Clipping the index to zero instead sends them to row 0,
    # which is a real line of sight somewhere else entirely, and they never see the flat prior they are due.
    row = np.asarray(curves["index"])[np.asarray(pixel)]
    index = np.where(row >= 0, row, len(curves["shapes"]) - 1).astype(np.int32)
    out["dustIndex"] = index
    total = curves.get("total")
    if total is not None:
        # The 2D map integrates the dust to infinity, which towards the bulge is tens of magnitudes and
        # says nothing about a star in front of it. Where a 3D map has measured the column out past the far
        # side of the disc, that measurement is the bound, and the larger 2D value is dropped. The flat
        # sightline carries a column of zero, so a star it covers keeps the bound it came with.
        measured = np.asarray(total)[index]
        out["Ar"] = np.where(measured > 0, np.minimum(out["Ar"].to_numpy(), measured), out["Ar"])
    return npd.NestedFrame(out)


def healpixOrder(nside):
    """The order of a HEALPix grid of the given nside, which the nested numbering needs a power of two.

    Rounding the logarithm of anything else quietly renumbers every pixel of the grid, so it is an error
    rather than something to make the best of.
    """
    order = int(round(np.log2(nside))) if nside > 0 else 0
    if nside <= 0 or 1 << order != int(nside):
        raise SystemExit(f"nside {nside} is not a power of two, so it is not a HEALPix grid")
    return order


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


def readCurves(path):
    """The 3D dust curves of a run, with a flat sightline added for the sky no 3D map covers.

    The index of the file holds -1 there, and those stars are meant to keep the flat A_r prior. One row of
    zeros at the end of the table and the -1 sent to it is what arranges that, with no special case anywhere
    else: the fit drops the Gaussian prior for a curve that ends at zero (photod.bayes.starPosterior tests
    the end of the curve), and a total column of zero leaves the extinction bounded by the 2D map alone.
    """
    if not path:
        return {}
    with np.load(path) as data:
        curves = {name: data[name] for name in data.files}
    shapes = np.asarray(curves["shapes"])
    curves["shapes"] = np.vstack([shapes, np.zeros((1, shapes.shape[1]), dtype=shapes.dtype)])
    if "total" in curves:
        total = np.asarray(curves["total"])
        curves["total"] = np.concatenate([total, np.zeros(1, dtype=total.dtype)])
    # the grid is checked here rather than where the pixels are looked up, which is inside a worker process
    curves["order"] = healpixOrder(int(curves["nside"]))
    return curves


def loadCurves(path):
    """The dust curves, read once per worker process."""
    if path not in CURVES:
        CURVES[path] = readCurves(path)
    return CURVES[path]


def mapFile(path):
    """Where the unpacked maps live: a scratch copy keyed to the file and the time it was written.

    Not beside the maps themselves, which ship with the repository and should not collect half a gigabyte of
    working file every time a run starts.
    """
    stamp = int(Path(path).stat().st_mtime)
    return Path(tempfile.gettempdir()) / f"photod-{Path(path).stem}-{stamp}.kde.npy"


def priorDecades(data):
    """How far below its own peak a compact prior map was kept, as the file carrying it says.

    The number is one end of the quantisation and scripts/make_priors.py holds the other, so a file is
    decoded with the value it was written with rather than with a copy of that value kept here, which would
    decode every shipped map wrong the day the other end moves. Files written before it was recorded all
    used eight decades.
    """
    return float(data["decades"]) if "decades" in data else PRIOR_DECADES


def priorConstants(data):
    """The grid constants a prior file records, as a dictionary; empty for a file that records none.

    scripts/make_priors.py writes getBayesConstants() beside the maps as JSON, which is one string and needs
    no pickle to read back. A table of name and value is read as well: what has to be agreed on is the file,
    not the shape one version of that script happened to store the numbers in.
    """
    if "constants" not in data:
        return {}
    stored = np.asarray(data["constants"])
    if stored.ndim == 2 and stored.shape[1] == 2:
        return {str(name): float(value) for name, value in stored}
    if stored.dtype.kind == "U" and stored.size == 1:
        return {str(name): float(value) for name, value in json.loads(str(stored.item())).items()}
    raise SystemExit(f"the prior file records its constants as {stored.dtype} of shape {stored.shape}")


def priorMaps(data):
    """The maps as densities, undoing the quantisation a compact file stores them with.

    A compact file keeps the log of each map relative to its own peak, to a byte over the decades the file
    records, on every other point of the grid. The maps are smoothed densities, so that loses about 0.008 dex
    where the prior has any weight, against a chi2 that runs to hundreds.
    """
    kde = data["kde"]
    if kde.dtype != np.uint8:
        return kde
    decades = priorDecades(data)
    levels = float(np.iinfo(np.uint8).max)
    out = 10 ** (kde.astype(np.float32) / levels * decades - decades)
    out *= data["kdeScale"][:, :, None, None]
    return np.where(kde == 0, 0.0, out).astype(np.float32)


def checkPriorFile(path):
    """Check a prior file against the code that is about to read it, before the pool starts.

    The maps are tabulated on the grid of photod.priors.getBayesConstants() and read back on it, so a file
    built with another grid comes out wrong everywhere rather than obviously; a file that records the grid it
    was built with says so here instead. An older file records nothing and is taken as it comes.
    """
    with np.load(path) as data:
        decades, constants = priorDecades(data), priorConstants(data)
    wanted = getBayesConstants()
    wrong = {name: value for name, value in constants.items() if float(wanted.get(name, value)) != value}
    if wrong:
        said = ", ".join(f"{name} = {value:g} rather than {wanted[name]:g}" for name, value in wrong.items())
        raise SystemExit(
            f"the maps in {path} were built with {said}: rebuild them, or fit with the code they came with"
        )
    grid = "built on the grid this code reads" if constants else "built on a grid the file does not record"
    print(f"prior maps from {path}: {grid}, {decades:g} decades of log in a compact file")


def unpackPriors(path):
    """Write the maps out once as a plain array the workers can map.

    They are two gigabytes and every worker needs a different handful of sightlines out of them. Read as a
    file each worker holds its own copy, which at eight workers is seventeen gigabytes and, if the file is
    compressed, a minute of unpacking; mapped they share one copy and take about thirty megabytes each.
    """
    cache = mapFile(path)
    if cache.exists():
        return
    with np.load(path) as data:
        maps = priorMaps(data)
    writeAtomically(cache, lambda name: np.save(name, maps))


def writeAtomically(path, write, scratch=None):
    """Put a file in place in one step: written under a temporary name, then renamed.

    A truncated file is worse than a missing one, because nothing downstream can tell: a resumed run counts
    whatever partition file it finds as done, and a half-written copy of the unpacked prior maps fails every
    worker of every run that follows. A rename inside one filesystem is atomic, so the file is either whole
    or not there, however the run is killed and however many runs are writing it at once.

    The temporary name goes in scratch, which has to be on the same filesystem for the rename to stay
    atomic, and by default beside the file itself. A partition of the result is written in a directory of
    its own instead: see partialDirectory.
    """
    path = Path(str(path))
    handle, temporary = tempfile.mkstemp(
        dir=Path(str(scratch)) if scratch else path.parent, prefix=f"{path.stem}-", suffix=path.suffix
    )
    os.close(handle)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def partialDirectory(base):
    """Where a partition is written before it is renamed into place: beside the dataset, not inside it.

    hats.io.write_parquet_metadata reads every parquet file under <base>/dataset, so a temporary file left
    there by a run the kernel killed breaks the metadata step of that run and of every run that follows, and
    nothing in the catalog says which file to remove. One directory up is the same filesystem, so the rename
    is still atomic, and the scan never sees the file at all.
    """
    directory = Path(str(base)) / PARTIAL
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sweepOrphans(base):
    """Drop what a run of an earlier version left half written inside the dataset, and say so.

    Those runs wrote a partition under its temporary name beside the partition itself, so a kill between the
    write and the rename left a Npix=<pixel>-<suffix>.parquet that the metadata step of every later run then
    failed on, with nothing in the catalog to say which file to remove. It is reported and removed rather
    than only reported, because the catalog is broken for as long as it is there and no version of this
    script writes such a name any more, so nothing can be in the middle of writing one.

    What this version leaves behind instead is a file in PARTIAL, which nothing reads and which the writer
    itself removes unless the kernel takes the process without warning.
    """
    orphans = sorted(Path(str(base)).glob("dataset/Norder=*/Dir=*/Npix=*-*.parquet"))
    for path in orphans:
        print(f"removing {path}, which a killed run of an earlier version left half written")
        path.unlink(missing_ok=True)
    return len(orphans)


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
        # the same curves the sightlines were looked up in, flat row and all, so that the index of a star
        # whose sky no 3D map covers points at that row here as well
        PARAMS[setup] = globalParameters(floor, useDustMap, loadCurves(curvePath) or None, arMax)
    return PARAMS[setup]


def separation(ra, dec, ra0, dec0):
    """Angle in degrees between each position and one centre."""
    a, d, a0, d0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    cosine = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def fitAndWrite(
    source, pixel, base, priorPath, curvePath, setup, batchSize, batchBytes, cone, arColumn, columns
):
    """Read one partition file, fit its stars and write them where the partition's HEALPix pixel belongs.

    The task carries a path rather than a piece of a catalog. Handing the workers pieces of a catalog sends
    each of them the structure of the whole survey along with it, which on DP2 is gigabytes per worker before
    a single star is read, and the results are far larger than the stars they came from, so collecting them
    for a separate writing step holds the survey in memory as well.
    """
    frame = pq.read_table(source, columns=columns).to_pandas()
    stars = prepareStars(frame, loadCurves(curvePath), arColumn)
    del frame
    if cone is not None and len(stars):
        ra, dec, radius = cone
        stars = stars[separation(stars["ra"].to_numpy(), stars["dec"].to_numpy(), ra, dec) <= radius]
    estimates = fitPartition(stars, priorPath, setup, batchSize, batchBytes)
    if not len(estimates):
        return 0
    writePartition(base, pixel, estimates)
    return len(estimates)


def writePartition(base, pixel, estimates):
    """One partition of answers in place, under the name its HEALPix pixel gives it.

    The whole of what a partition looks like on disk is here, so that what a run writes and what a test
    reads back are the same thing rather than two spellings of it.
    """
    path = Path(str(pixelFile(base, pixel)))
    path.parent.mkdir(parents=True, exist_ok=True)
    answers = withSpatialIndex(pd.DataFrame(estimates))
    writeAtomically(path, lambda name: answers.to_parquet(name, index=False), scratch=partialDirectory(base))
    return path


def withSpatialIndex(estimates):
    """The answers as a HATS partition: a column of the order 29 HEALPix cell of each row, in its order.

    That index is what makes the result a catalog rather than a heap of parquet files. A reader uses it to
    find the rows of a partition that lie in a region without reading the positions, and expects the rows in
    its order.

    It is a plain column and not the pandas index of the frame, which is how lsdb writes a catalog of its
    own. A file whose parquet metadata names an index column hands the reader a frame with that column
    already taken out of it, and lsdb then asks a partition for the index column it means to read the rows
    by and is told it is not there: open_catalog(columns=...) and cone_search both fail on such a file,
    while a plain read of the whole catalog works, so the fault only shows up in the readers that matter.
    """
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN, compute_spatial_index

    index = compute_spatial_index(estimates["ra"].to_numpy(), estimates["dec"].to_numpy())
    out = estimates.assign(**{SPATIAL_INDEX_COLUMN: index}).sort_values(SPATIAL_INDEX_COLUMN)
    return out.reset_index(drop=True)


def fitPartition(partition, priorPath, setup, batchSize, batchBytes=None):
    """One partition of stars, each group of them fitted with the prior maps of the sky pixel it lies in.

    Looking the sightline up beats joining against a catalog of maps: a join is between two pixel trees of
    different depth and quietly keeps only one of the star partitions that share a map.

    A star whose sky pixel has no map at all keeps its row, empty and flagged. The maps are built for a
    footprint, and a footprint never lines up exactly with the survey that produced the stars: the DP2 maps
    that ship here are missing three of the pixels their own dust file covers. Dropping those stars leaves a
    catalog that holds fewer stars than the selection it came from, and such a catalog cannot count stars,
    which is most of what these catalogs are for. The rows come back in no particular order.

    batchBytes bounds the memory of one batch, and so of one worker at a time: a run of several workers asks
    for that much again for each of them.
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
    if (row < 0).any():
        pieces.append(unfittedEstimates(partition[row < 0], globalParams, FLAG_NO_PRIOR))
    for value in np.unique(row[row >= 0]):
        stars = partition[row == value]
        grid = priorGridFromMaps(
            priors["kde"][value], priors["rmag"], priors["xGrid"], priors["yGrid"], globalParams
        )
        with jax.default_device(device):
            estimates, _ = makeBayesEstimates3d(
                stars,
                jax.numpy.array(list(grid.values())),
                globalParams,
                batchSize=batchSize,
                batchBytes=batchBytes,
            )
        pieces.append(estimates)
    if not pieces:
        return empty
    return npd.NestedFrame(pd.concat(pieces, ignore_index=True))


def startWorker(base, priorPath, curvePath, setup, batchSize, batchBytes, cone, arColumn, columns):
    """What every partition of this run needs, held once per worker process."""
    WORK.update(
        base=base,
        priorPath=priorPath,
        curvePath=curvePath,
        setup=setup,
        batchSize=batchSize,
        batchBytes=batchBytes,
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
        WORK["batchBytes"],
        WORK["cone"],
        WORK["arColumn"],
        WORK["columns"],
    )


def partitionFiles(catalog):
    """Every partition the run covers, as its HEALPix pixel and the parquet file holding it."""
    from hats.io.paths import pixel_catalog_file

    root = catalog.hc_structure.catalog_base_dir
    return [(pixel, pixel_catalog_file(root, pixel)) for pixel in catalog.hc_structure.get_healpix_pixels()]


def writtenRows(path):
    """Stars already written for a partition, or -1 when there is no answer there worth keeping.

    A file that will not read is not counted as done: the answers are written atomically, so a half-written
    one came from something else, and fitting that partition again costs far less than a hole in the catalog.
    """
    path = Path(str(path))
    if not path.exists():
        return -1
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:  # not a parquet file this run can count on, so it is refitted
        return -1


def partitionsToFit(base, sources):
    """The partitions still to fit, and the stars the ones already written hold.

    A partition whose answers are there is left alone, so that a survey run killed in the middle is finished
    by repeating the command rather than started again, and the stars of those partitions are counted for the
    catalog metadata. A partition that held no stars the fit could use has nothing to find and is read again,
    which costs one parquet file.
    """
    todo, rows = [], 0
    for pixel, path in sources:
        written = writtenRows(pixelFile(base, pixel))
        if written < 0:
            todo.append((pixel, path))
        else:
            rows += written
    return todo, rows


def runConfiguration(args):
    """What this run asks of the fit, as the settings that decide the answers and not one more.

    A partition is resumed on the strength of the file being there, which says nothing about what produced
    it, so the settings are recorded beside the answers and a resume that asks for others is refused. The
    cone belongs here: a narrower one leaves the partitions outside it untouched but is also a fine filter
    inside the ones it keeps, so resuming with it would leave whole partitions of stars that are outside it.
    How the work is divided up is not here, neither the workers nor the chunk nor either batch setting,
    because none of it changes an answer and a run must be free to finish on a smaller machine than it
    started on.
    """
    return {
        # a directory completed by the shell carries a trailing slash and the same command typed again may
        # not, which is the same catalog and must not read as another fit
        "catalog": str(args.catalog).rstrip("/"),
        "priors": str(args.priors),
        "floor": float(args.floor),
        "arMax": float(args.ar_max),
        # what the fit actually reads, so that --no-dust-map with curves named and --no-dust-map without
        # them, which give the same answers to the last bit, resume one another
        "arColumn": args.ar_column if not args.no_dust_map else None,
        "dustMap": not args.no_dust_map,
        "dustCurves": str(args.dust_curves) if not args.no_dust_map else "",
        "cone": [float(x) for x in args.cone] if args.cone else None,
    }


def recordConfiguration(base, configuration):
    """Refuse a resume that asks for another fit than the answers already there were made with.

    An answer already written is kept whatever this run was asked for, so two settings inside one catalog
    are not something a reader can see, let alone undo. A result written before this was recorded says
    nothing about itself and is taken as it comes, which is the one case where there is nothing to compare.

    What is protected is the answers, so a directory that holds none takes the settings it is given. A run
    that stopped before its first partition, on a prior file it could not read or a catalog column it could
    not find, recorded itself all the same, and the corrected command would otherwise be refused by the
    empty directory the first one left.
    """
    path = Path(str(base)) / RUN_FILE
    answers = any(Path(str(base)).glob("dataset/Norder=*/Dir=*/Npix=*.parquet"))
    if path.exists() and answers:
        found = json.loads(path.read_text())
        differs = {
            name: (found.get(name), value)
            for name, value in configuration.items()
            if found.get(name) != value
        }
        if differs:
            said = ", ".join(f"{name} {was!r} rather than {now!r}" for name, (was, now) in differs.items())
            raise SystemExit(
                f"{base} was fitted with {said}: a resume keeps every partition already written, so this "
                "would leave two fits inside one catalog. Fit it again with --overwrite, or write this one "
                "under another --name"
            )
    elif answers and not path.exists():
        print(f"{base} holds answers but no record of what made them, so they are taken as this run's")
    path.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n")


def poolOptions(chunk):
    """How the pool bounds the memory of a survey-wide run, as far as this interpreter allows.

    A worker is replaced every chunk partitions, which the pool only does with processes it starts itself
    rather than forks. ProcessPoolExecutor learned to do it in Python 3.11 and this package supports 3.10,
    where the argument is not merely ignored but a TypeError as the pool is built.
    """
    options = {"mp_context": mp.get_context("spawn")}
    if chunk <= 0:
        return options
    if not WORKERS_REPLACED:
        print(
            "this Python cannot replace a worker of the pool (3.11 and up can), so the workers live for the "
            "whole run and their memory grows with it: split a survey-wide run by --cone if it runs out",
            flush=True,
        )
        return options
    return options | {"max_tasks_per_child": chunk}


def fitPartitions(sources, workers, chunk, initargs):
    """Every partition of the run, fitted in a pool of processes, a failure reported rather than fatal.

    A partition can fail on its own account: a parquet file that will not read, or a worker the kernel kills
    for the memory it asked for. A survey run is hours long and the rest of the sky is unaffected by any of
    that, so a failure costs its own partition, is counted, and leaves the exit status of the run saying that
    some of the sky is missing. The pool comes from concurrent.futures because a worker that dies there
    breaks the pool and raises, where multiprocessing.Pool waits for an answer that can no longer come.
    """
    total, failed = 0, 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=startWorker,
        initargs=initargs,
        **poolOptions(chunk),
    ) as pool:
        futures = {pool.submit(runPartition, source): source for source in sources}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                total += future.result()
            except Exception as error:  # the partition, or the pool the worker of it took with it
                failed += 1
                if failed <= FAILURES_REPORTED:
                    print(f"  {futures[future][1]}: {type(error).__name__}: {error}", flush=True)
            if done % 200 == 0 or done == len(sources):
                print(f"  {done}/{len(sources)} partitions, {total} stars, {failed} failed", flush=True)
    return total, failed


def writeCatalogMetadata(base, name, pixels, total):
    """The HATS metadata beside the parquet files, so that the result reads back as a catalog.

    True when it is written. The parquet files are the result and the index can be rebuilt from them, so a
    failure here is not worth throwing the run away over, but it leaves something that does not read back as
    a catalog and the caller says so in the exit status.
    """
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
        return True
    except Exception as error:
        print(f"the parquet files are written but the catalog index is not: {type(error).__name__}: {error}")
        return False


def pixelFile(base, pixel):
    """Where one HEALPix pixel's parquet file goes."""
    from hats.io.paths import pixel_catalog_file
    from hats.pixel_math import HealpixPixel

    return pixel_catalog_file(base, HealpixPixel(pixel.order, pixel.pixel))


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True, help="DP2 object collection (HATS)")
    ap.add_argument(
        "--priors",
        default=str(PRIOR_FILE),
        help="TRILEGAL prior maps; the ones the DP2 footprint was fitted with come with the repository",
    )
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
        default=8.0,
        help="top of the A_r grid; keep it above 1.3 A_r(map) + 0.1 of the dustiest star in the field",
    )
    ap.add_argument(
        "--no-dust-map",
        action="store_true",
        help="flat A_r prior instead of the dust-map bound, which reads neither the extinction column nor "
        "any 3D curve: the Gaussian prior of a 3D map is its curve scaled by A_r(map), so without that map "
        "the curves weigh nothing, and --dust-curves is ignored",
    )
    ap.add_argument(
        "--dust-curves",
        default=str(DUST_FILE),
        help="3D dust map as the A_r prior, which is what matters at low Galactic latitude; "
        'the curves for the DP2 footprint come with the repository, and "" turns it off',
    )
    ap.add_argument("--workers", type=int, default=1, help="processes, which share the GPUs between them")
    ap.add_argument(
        "--chunk",
        type=int,
        default=400,
        help="partitions a worker fits before it is replaced, which bounds the memory of a survey-wide run",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="fit a result of the same name again from nothing, instead of keeping the partitions it holds",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="stars per JAX call; a few hundred is the fastest, larger batches spill and slow down",
    )
    ap.add_argument(
        "--batch-bytes",
        type=int,
        default=2 << 30,
        help="memory one batch of the fit may take, per worker: a pool of several asks for that much each",
    )
    args = ap.parse_args()

    # what the fit reads, which is not quite what was asked for: the flat A_r prior reads no extinction and
    # no 3D curves, so the run is configured, resumed and reported as the fit it actually is
    configuration = runConfiguration(args)
    arColumn, curvePath = configuration["arColumn"], configuration["dustCurves"]
    search = (
        lsdb.ConeSearch(ra=args.cone[0], dec=args.cone[1], radius_arcsec=3600 * args.cone[2])
        if args.cone
        else None
    )
    columns = inputColumns(catalogColumns(args.catalog), arColumn)
    objects = lsdb.open_catalog(args.catalog, columns=columns, search_filter=search)
    sources = [(pixel, str(path)) for pixel, path in partitionFiles(objects)]
    print(f"reading {'prepared colours' if 'ug' in columns else 'PSF fluxes'} from {args.catalog}")
    cone = tuple(args.cone) if args.cone else None
    setup = (args.floor, not args.no_dust_map, curvePath, args.ar_max)
    if curvePath:
        curves = readCurves(curvePath)
        measured = curves.get("total")
        bounded = int((measured > 0).sum()) if measured is not None else 0
        print(
            f"3D dust prior from {curvePath}: {len(curves['shapes']) - 1} sightlines, "
            f"{bounded} of them with a measured total column to bound the extinction; "
            "a star the maps do not reach keeps the flat A_r prior"
        )
    elif args.no_dust_map:
        print(
            "flat A_r prior over the whole grid: no extinction column is read and no 3D curve is opened"
            + (f", so {args.dust_curves} is left alone" if args.dust_curves else "")
        )

    base = Path(args.out) / args.name
    if base.exists() and args.overwrite:
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    recordConfiguration(base, configuration)
    sweepOrphans(base)
    checkPriorFile(args.priors)
    unpackPriors(args.priors)

    todo, total = partitionsToFit(base, sources)
    kept = len(sources) - len(todo)
    print(
        f"{len(todo)} of {len(sources)} partitions to fit"
        + (f", {kept} already written with {total} stars; --overwrite to fit them again" if kept else ""),
        flush=True,
    )

    initargs = (
        base,
        args.priors,
        curvePath,
        setup,
        args.batch_size,
        args.batch_bytes,
        cone,
        arColumn,
        columns,
    )
    fitted, failed = fitPartitions(todo, args.workers, args.chunk, initargs)
    total += fitted
    indexed = writeCatalogMetadata(base, args.name, [p for p, _ in sources], total)
    print(f"written {base}: {total} stars")
    if failed:
        raise SystemExit(f"{failed} of {len(todo)} partitions failed; run the same command again for them")
    if not indexed:
        raise SystemExit(f"{base} does not read back as a catalog until its parquet metadata is written")


if __name__ == "__main__":
    main()
