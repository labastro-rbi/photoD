"""The run script: the parts of it that decide which stars are fitted, with what, and what is written.

scripts/ is not a package, so the script is imported by its path. Nothing here fits a star or starts a pool
of processes: a worker of this run imports JAX and lsdb and builds the reddened locus, which is minutes, so
the pool is watched through a stand-in and the fit itself is covered by the tests of photod.bayes.
"""

import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from photod.bayes import FLAG_NO_PRIOR, FLAG_POOR_FIT, getEstimatesMeta, unfittedEstimates
from photod.priors import getBayesConstants

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_dp2.py"
DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.fixture(scope="module")
def run():
    """The run script as a module, imported once for the whole file."""
    spec = importlib.util.spec_from_file_location("run_dp2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_dp2"] = module
    spec.loader.exec_module(module)
    return module


def pixelCentre(pixel, order):
    """Where the centre of one nested HEALPix pixel is, so that a star can be put on a chosen sightline."""
    import cdshealpix

    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.array([pixel]), order)
    return float(lon.deg[0]), float(lat.deg[0])


def dustFile(path, nside=8, covered=(0, 1)):
    """A dust curve file of two sightlines, covering the given pixels and nothing else."""
    mu = np.arange(4.0, 16.01, 3.0)
    shapes = np.stack([np.linspace(0, 1, mu.size), np.where(mu < 10, 0.0, 1.0)]).astype(np.float32)
    index = np.full(12 * nside**2, -1, dtype=np.int32)
    index[list(covered)] = np.arange(len(covered), dtype=np.int32)
    np.savez(
        path,
        shapes=shapes,
        total=np.array([0.5, 0.0], dtype=np.float32),
        mu=mu,
        index=index,
        nside=nside,
    )
    return path


def asTable(constants):
    """The grid constants as a table of name and value, the other shape a prior file may record them in."""
    return np.array([[name, str(value)] for name, value in constants.items()])


def estimatesFrame(ra, dec):
    """Answers with the columns and the types the fit writes, one row per position."""
    meta = getEstimatesMeta(computeMrTrue=True)
    frame = pd.DataFrame({name: np.zeros(len(ra), dtype=dtype) for name, dtype in meta.dtypes.items()})
    frame["objectId"] = np.arange(len(ra), dtype=np.int64)
    frame["ra"], frame["dec"] = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    return frame


def test_a_sightline_no_3d_map_covers_keeps_the_flat_prior(run, tmp_path):
    """The -1 of the index has to reach a flat curve, not row 0, which is a real sightline somewhere else.

    A curve of zeros is what turns the Gaussian A_r prior off (photod.bayes.starPosterior tests the end of
    the curve, and tests/photod/test_dust_prior.py checks that such a curve gives the flat-prior answer), and
    a total column of zero is what leaves the extinction bounded by the 2D map alone.
    """
    curves = run.readCurves(str(dustFile(tmp_path / "curves.npz")))
    ra, dec = zip(*[pixelCentre(pixel, 3) for pixel in (0, 1, 700)], strict=True)
    stars = run.withDust(pd.DataFrame({"ra": ra, "dec": dec, "Ar": [3.0, 3.0, 3.0]}), curves)

    index = stars["dustIndex"].to_numpy()
    assert index[2] == len(curves["shapes"]) - 1, "the uncovered star was sent to a sightline of the file"
    assert np.all(curves["shapes"][index[2]] == 0), "the uncovered star got a curve with dust on it"
    assert curves["total"][index[2]] == 0, "the uncovered star got a measured column to be bounded by"
    assert stars["Ar"].to_numpy()[2] == 3.0, "the uncovered star had its extinction bounded"
    assert list(index[:2]) == [0, 1], "a covered star lost its own sightline"
    assert stars["Ar"].to_numpy()[0] == 0.5, "a measured column no longer bounds the extinction"


def test_the_dust_curves_that_ship_leave_their_uncovered_sky_flat(run):
    """The same on the real file, which covers 22400 of the 196608 pixels of its grid."""
    curves = run.readCurves(str(DATA / "dust_dp2.npz"))
    index = np.asarray(curves["index"])
    ra, dec = zip(
        *[
            pixelCentre(pixel, curves["order"])
            for pixel in (int(np.argmax(index >= 0)), int(np.argmin(index)))
        ],
        strict=True,
    )
    stars = run.withDust(pd.DataFrame({"ra": ra, "dec": dec, "Ar": [9.0, 9.0]}), curves)

    covered, uncovered = stars["dustIndex"].to_numpy()
    assert curves["shapes"][covered][-1] == 1, "a covered sightline no longer carries all of its dust"
    assert np.all(curves["shapes"][uncovered] == 0), "an uncovered sightline came with dust on it"
    assert stars["Ar"].to_numpy()[1] == 9.0, "an uncovered star had its extinction bounded"


def test_the_fit_reads_the_curves_the_sightlines_were_looked_up_in(run, tmp_path, monkeypatch):
    """The flat row is only the flat row if the table the fit reads is the one the index was resolved in."""
    monkeypatch.setattr(run, "PARAMS", {})
    monkeypatch.setattr(run, "CURVES", {})
    monkeypatch.setattr(run, "globalParameters", lambda floor, useDustMap, curves=None, arMax=5.0: curves)
    path = str(dustFile(tmp_path / "curves.npz"))

    curves = run.loadParams((0.03, True, path, 8.0))
    assert curves is run.loadCurves(path), "the fit was given another table than the index was resolved in"
    assert np.all(curves["shapes"][-1] == 0), "the table the fit reads has no flat sightline at its end"
    assert run.loadParams((0.03, True, "", 8.0)) is None, "a run with no dust curves was given some"


def test_an_nside_that_is_not_a_power_of_two_is_an_error(run, tmp_path):
    """int(log2(nside)) on anything else renumbers every pixel of the grid, quietly and completely."""
    assert [run.healpixOrder(n) for n in (1, 8, 128, 4096)] == [0, 3, 7, 12]
    for nside in (0, -4, 100, 129):
        with pytest.raises(SystemExit):
            run.healpixOrder(nside)
    with pytest.raises(SystemExit):
        run.readCurves(str(dustFile(tmp_path / "odd.npz", nside=12)))


def test_a_star_with_no_prior_map_is_kept_and_flagged(run, monkeypatch):
    """The maps cover the footprint they were built for, which is not the one the stars came from.

    The pixels come from the index of the maps that ship here, which is 49 kB of that 140 MB file and the
    only part of it this reads. When this was written the file was missing three of the order 5 pixels its
    own dust file covers, so the sky it does not reach is not a hypothetical.
    """
    globalParams = SimpleNamespace(computeMrTrue=True)
    with np.load(DATA / "priors_dp2.npz") as data:
        index, order = np.asarray(data["index"]), int(data["order"])
    covered = np.where(index >= 0)[0][:2]
    missing = np.where(index < 0)[0][:2]
    maps = np.zeros((int(index.max()) + 1, 4, 4, 4))
    monkeypatch.setattr(run, "loadParams", lambda setup: globalParams)
    monkeypatch.setattr(
        run,
        "loadPriors",
        lambda path: {"index": index, "order": order, "kde": maps}
        | dict.fromkeys(("rmag", "xGrid", "yGrid"), np.zeros(4)),
    )
    monkeypatch.setattr(run, "priorGridFromMaps", lambda *args: {0: np.zeros(4)})
    # a real fit is minutes of locus; what matters here is which stars reach it and what becomes of the rest
    monkeypatch.setattr(
        run,
        "makeBayesEstimates3d",
        lambda stars, *args, **kwargs: (unfittedEstimates(stars, globalParams, 0), None),
    )

    pixels = [covered[0], covered[1], covered[1], missing[0], missing[1]]
    ra, dec = zip(*[pixelCentre(pixel, order) for pixel in pixels], strict=True)
    stars = pd.DataFrame(
        {"objectId": np.arange(5), "ra": ra, "dec": dec, "rmag": np.full(5, 20.0), "Ar": np.zeros(5)}
    )
    estimates = run.fitPartition(stars, "priors.npz", (0.03, True, "", 8.0), 100)

    assert len(estimates) == len(stars), "a star was dropped for the sky it sits in"
    assert sorted(estimates["objectId"]) == sorted(stars["objectId"])
    flags = dict(zip(estimates["objectId"], estimates["flags"], strict=True))
    assert all(flags[star] == FLAG_NO_PRIOR | FLAG_POOR_FIT for star in (3, 4)), "the rows do not say why"
    assert not any(flags[star] & FLAG_NO_PRIOR for star in (0, 1, 2)), "a fitted star was marked unfittable"


def test_an_answer_is_either_whole_or_not_there(run, tmp_path):
    """A truncated file is worse than a missing one, because a later run counts it as done."""
    path = tmp_path / "answers.parquet"
    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (0, 1, 2)], strict=True))
    run.writeAtomically(path, lambda name: run.withSpatialIndex(frame).to_parquet(name, index=True))
    assert run.writtenRows(path) == 3

    def killed(name):
        Path(name).write_bytes(b"PAR1 and then the run was killed")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run.writeAtomically(tmp_path / "half.parquet", killed)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["answers.parquet"], "a temporary file was left"


def test_a_partition_that_is_already_written_is_not_fitted_again(run, tmp_path):
    """Resuming a survey run that was killed in the middle, rather than starting it again."""
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "catalog"
    sources = [(HealpixPixel(5, pixel), f"partition-{pixel}.parquet") for pixel in (0, 1, 2)]
    todo, rows = run.partitionsToFit(base, sources)
    assert todo == sources and rows == 0, "nothing is written yet, so everything is to be fitted"

    path = Path(str(run.pixelFile(base, sources[1][0])))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (7, 8)], strict=True))
    run.writeAtomically(path, lambda name: run.withSpatialIndex(frame).to_parquet(name, index=True))
    todo, rows = run.partitionsToFit(base, sources)
    assert todo == [sources[0], sources[2]], "a partition that is written was fitted again"
    assert rows == 2, "the stars already written are not counted towards the catalog"

    path.write_bytes(b"PAR1 and then the run was killed")
    todo, rows = run.partitionsToFit(base, sources)
    assert todo == sources and rows == 0, "a file that will not read was counted as an answer"


def test_a_failed_partition_costs_its_own_partition(run, monkeypatch, capsys):
    """A survey run is hours long: a partition that fails is reported and counted, and the rest go on."""
    started = []

    class LocalPool:
        """The pool, running its tasks here, so that the loop over them can be watched."""

        def __init__(self, **kwargs):
            started.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *problem):
            return False

        def submit(self, task, *args):
            future = Future()
            try:
                future.set_result(task(*args))
            except Exception as error:
                future.set_exception(error)
            return future

    def fitOne(source):
        if source[1] == "bad.parquet":
            raise MemoryError("the worker asked for more than the machine has")
        return 3

    monkeypatch.setattr(run, "ProcessPoolExecutor", LocalPool)
    monkeypatch.setattr(run, "runPartition", fitOne)
    sources = [(0, "good.parquet"), (1, "bad.parquet"), (2, "also-good.parquet")]
    total, failed = run.fitPartitions(sources, workers=2, chunk=400, initargs=("catalog",))

    assert (total, failed) == (6, 1), "a failure took the rest of the run with it, or was not counted"
    printed = capsys.readouterr().out
    assert "bad.parquet" in printed and "MemoryError" in printed, "the failure does not say what or where"
    assert started[0]["max_tasks_per_child"] == 400, "the workers are no longer replaced"
    assert started[0]["initializer"] is run.startWorker and started[0]["max_workers"] == 2


def test_the_answers_read_back_as_a_catalog(run, tmp_path):
    """Written without the spatial index the result is a heap of parquet files rather than a HATS catalog."""
    import lsdb
    import pyarrow.parquet as pq
    from hats.pixel_math import HealpixPixel
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0), HealpixPixel(3, 5)]
    for pixel in pixels:
        path = Path(str(run.pixelFile(base, pixel)))
        path.parent.mkdir(parents=True, exist_ok=True)
        inside = [pixelCentre(pixel.pixel * 4 + corner, 4) for corner in range(4)]
        frame = run.withSpatialIndex(estimatesFrame(*zip(*inside, strict=True)))
        assert frame.index.name == SPATIAL_INDEX_COLUMN
        assert np.all(np.diff(frame.index.to_numpy()) >= 0), "the rows are not in the order of the index"
        run.writeAtomically(path, lambda name, frame=frame: frame.to_parquet(name, index=True))
        assert SPATIAL_INDEX_COLUMN in pq.read_schema(path).names, "the file has no spatial index column"
    run.writeCatalogMetadata(base, "photod", pixels, 8)

    catalog = lsdb.open_catalog(base)
    answers = catalog.compute()
    assert len(answers) == 8
    assert answers.index.name == SPATIAL_INDEX_COLUMN, "the catalog came back without its index"
    assert list(answers.columns) == list(getEstimatesMeta(computeMrTrue=True).columns)


def test_the_maps_are_decoded_the_way_the_file_that_holds_them_says(run):
    """The quantisation is shared with scripts/make_priors.py, so the file has to carry its own end of it."""
    kde = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
    scale = np.ones((1, 1), dtype=np.float32)
    eight = run.priorMaps({"kde": kde, "kdeScale": scale})
    four = run.priorMaps({"kde": kde, "kdeScale": scale, "decades": 4.0})

    assert eight[0, 0, 0, 2] == pytest.approx(1.0) and four[0, 0, 0, 2] == pytest.approx(1.0)
    assert eight[0, 0, 0, 0] == 0.0, "an empty cell of the map came back with weight in it"
    # half a byte below the peak is four decades down in one file and eight in the other, a factor of a
    # hundred, and reading a map with the wrong one of them tilts every prior it holds
    assert eight[0, 0, 0, 1] == pytest.approx(10 ** (128 / 255 * 8 - 8), rel=1e-5)
    assert four[0, 0, 0, 1] == pytest.approx(10 ** (128 / 255 * 4 - 4), rel=1e-5)
    assert run.priorDecades({"kde": kde}) == run.PRIOR_DECADES


def test_prior_maps_built_on_another_grid_are_refused(run, tmp_path):
    """The maps are read back on the grid they were tabulated on; another grid is wrong everywhere.

    scripts/make_priors.py records the grid as JSON, and a table of name and value is read as well, so that
    the two scripts have to agree on the file rather than on the shape of one entry of it.
    """
    constants = getBayesConstants()
    maps = np.zeros((1, 1, 2, 2), dtype=np.uint8)
    for shape, recorded in (("json", json.dumps(constants)), ("table", asTable(constants))):
        np.savez(tmp_path / f"{shape}.npz", kde=maps, decades=6.0, constants=recorded)
        with np.load(tmp_path / f"{shape}.npz") as data:
            assert run.priorConstants(data) == constants, f"the {shape} the grid is recorded in was not read"
        run.checkPriorFile(tmp_path / f"{shape}.npz")

    np.savez(tmp_path / "wrong.npz", kde=maps, constants=asTable(constants | {"FeHNpts": 48}))
    with pytest.raises(SystemExit, match="FeHNpts"):
        run.checkPriorFile(tmp_path / "wrong.npz")
    np.savez(tmp_path / "odd.npz", kde=maps, constants=np.zeros(3))
    with pytest.raises(SystemExit, match="constants"):
        run.checkPriorFile(tmp_path / "odd.npz")


def test_the_prior_maps_that_ship_are_taken_as_they_come(run):
    """They record neither the grid nor the decades, and reading them must not depend on either.

    Only the metadata of the file is touched here: npz members are read when they are asked for, and the
    maps themselves are 140 MB.
    """
    run.checkPriorFile(DATA / "priors_dp2.npz")


def test_a_prepared_catalog_with_no_extinction_column_is_refused_before_the_pool(run):
    """A column that is only missed inside a worker arrives as a crashed pool instead of a sentence."""
    prepared = run.FIT_COLUMNS + ["Ar_SFD"]
    assert run.inputColumns(prepared, "Ar_SFD") == run.FIT_COLUMNS + ["Ar_SFD"]
    assert run.inputColumns(run.FIT_COLUMNS + ["ebv"], "Ar") == run.FIT_COLUMNS + ["ebv"]
    with pytest.raises(SystemExit, match="extinction"):
        run.inputColumns(prepared, "Ar")
    with pytest.raises(SystemExit, match="missing"):
        run.inputColumns(["objectId", "coord_ra"], "Ar")
