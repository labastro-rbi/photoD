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
# the command line of a run, as far as what it asks of the fit is concerned
SETTINGS = dict(
    catalog="/data/dp2",
    priors="/data/priors.npz",
    floor=0.03,
    ar_column="Ar",
    ar_max=8.0,
    no_dust_map=False,
    dust_curves="/data/dust.npz",
    cone=None,
)


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
    globalParams = SimpleNamespace(computeMrTrue=True, fitColors=("ug", "gr", "ri", "iz", "zy"))
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
        | {color + "Err": np.full(5, 0.02) for color in globalParams.fitColors}
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
    run.writeAtomically(path, lambda name: run.withSpatialIndex(frame).to_parquet(name, index=False))
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

    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (7, 8)], strict=True))
    path = run.writePartition(base, sources[1][0], frame)
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
        inside = [pixelCentre(pixel.pixel * 4 + corner, 4) for corner in range(4)]
        estimates = estimatesFrame(*zip(*inside, strict=True))
        index = run.withSpatialIndex(estimates)[SPATIAL_INDEX_COLUMN].to_numpy()
        assert np.all(np.diff(index) >= 0), "the rows are not in the order of the index"
        path = run.writePartition(base, pixel, estimates)
        assert SPATIAL_INDEX_COLUMN in pq.read_schema(path).names, "the file has no spatial index column"
    assert run.writeCatalogMetadata(base, "photod", pixels, 8) is True

    catalog = lsdb.open_catalog(base)
    answers = catalog.compute()
    assert len(answers) == 8
    assert answers.index.name == SPATIAL_INDEX_COLUMN, "the catalog came back without its index"
    assert list(answers.columns) == list(getEstimatesMeta(computeMrTrue=True).columns)


def test_a_shard_leaves_the_catalog_metadata_to_the_last_one_to_finish(run, tmp_path):
    """Shards split the partitions; the catalog they write together must read back as the whole survey."""
    import lsdb
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0), HealpixPixel(3, 5), HealpixPixel(3, 9), HealpixPixel(3, 40)]
    everything = [(pixel, f"partition-{pixel.pixel}.parquet") for pixel in pixels]
    shards = [(0, 2), (1, 2)]
    for shard in shards:
        run.startShard(base, shard)

    def fit(shard):
        for pixel, _ in everything[shard[0] :: shard[1]]:
            inside = [pixelCentre(pixel.pixel * 4 + corner, 4) for corner in range(4)]
            run.writePartition(base, pixel, estimatesFrame(*zip(*inside, strict=True)))

    fit(shards[0])
    assert run.finishShard(base, "photod", shards[0], everything) is None, "wrote the metadata before shard 1"
    assert not (base / "partition_info.csv").exists()
    fit(shards[1])
    assert run.finishShard(base, "photod", shards[1], everything) is True
    catalog = lsdb.open_catalog(base)
    assert len(catalog.get_healpix_pixels()) == 4, "the partition list is not the whole survey"
    assert catalog.hc_structure.catalog_info.total_rows == 16, "the row count is not the whole survey"
    assert len(catalog.compute()) == 16
    # a shard finishing again after the metadata is written does not write it a second time
    assert run.finishShard(base, "photod", shards[0], everything) is None


def test_a_resumed_shard_writes_the_metadata_again(run, tmp_path):
    """Partitions a resume adds are not in the metadata written before, so the claim is cleared on start."""
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0), HealpixPixel(3, 5)]
    everything = [(pixel, f"partition-{pixel.pixel}.parquet") for pixel in pixels]
    for shard in [(0, 2), (1, 2)]:
        run.startShard(base, shard)
    inside = [pixelCentre(pixels[0].pixel * 4 + corner, 4) for corner in range(4)]
    run.writePartition(base, pixels[0], estimatesFrame(*zip(*inside, strict=True)))
    assert run.finishShard(base, "photod", (0, 2), everything) is None
    assert run.finishShard(base, "photod", (1, 2), everything) is True   # shard 1's partition failed
    run.startShard(base, (1, 2))                                            # its resume
    inside = [pixelCentre(pixels[1].pixel * 4 + corner, 4) for corner in range(4)]
    run.writePartition(base, pixels[1], estimatesFrame(*zip(*inside, strict=True)))
    assert run.finishShard(base, "photod", (1, 2), everything) is True
    assert (base / "partition_info.csv").read_text().count("\n") == 3, "the resumed partition is not listed"


@pytest.mark.parametrize("text", ["4/4", "-1/4", "1", "a/b"])
def test_a_shard_that_is_not_one_of_n_is_refused(run, text):
    with pytest.raises(SystemExit):
        run.parseShard(text)


def test_a_shard_is_read_as_a_pair(run):
    assert run.parseShard("") is None
    assert run.parseShard("2/4") == (2, 4)


def test_a_reader_can_ask_a_written_catalog_for_columns_and_a_cone(run, tmp_path):
    """The two readers that matter, on a catalog written the way a run writes one.

    The order 29 cell of each row has to be a plain column of the partition file and not the pandas index of
    the frame it was written from. A file whose parquet metadata names an index column hands back a frame
    with that column already taken out of it, and lsdb, which reads the rows by that column, is told it is
    not there: only the plain read of the whole catalog survives, which is the one nobody uses on a survey.
    """
    import lsdb
    import pyarrow.parquet as pq
    from hats.io.validation import is_valid_catalog
    from hats.pixel_math import HealpixPixel
    from hats.pixel_math.spatial_index import SPATIAL_INDEX_COLUMN

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0), HealpixPixel(3, 5)]
    centre = None
    for pixel in pixels:
        inside = [pixelCentre(pixel.pixel * 4 + corner, 4) for corner in range(4)]
        centre = centre or inside[0]
        path = run.writePartition(base, pixel, estimatesFrame(*zip(*inside, strict=True)))
    assert run.writeCatalogMetadata(base, "photod", pixels, 8) is True

    picked = lsdb.open_catalog(base, columns=["ra", "dec", "chi2min"]).compute()
    assert list(picked.columns) == ["ra", "dec", "chi2min"] and len(picked) == 8
    assert picked.index.name == SPATIAL_INDEX_COLUMN, "the rows came back without their spatial index"
    cone = lsdb.open_catalog(base).cone_search(ra=centre[0], dec=centre[1], radius_arcsec=600).compute()
    assert len(cone) == 1, "a cone over one corner of one partition did not come back with that one row"
    recorded = pq.read_schema(path).pandas_metadata
    assert not (recorded or {}).get("index_columns"), "the file still records a pandas index column"
    assert is_valid_catalog(base, strict=True), "the result no longer validates as a HATS catalog"


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


def test_the_pool_is_kept_fed_rather_than_handed_the_whole_survey(run, monkeypatch):
    """A pool that replaces its workers starts the replacement only when something is submitted to it.

    Handed every partition at once it goes quiet for good the moment the last of its first workers reaches
    its chunk: the queue is full, every worker has retired, and nothing will submit the task that would
    start another. A survey run of 8840 partitions stopped dead at 2400 of them, which is 6 workers times a
    chunk of 400, and sat there until it was killed. The stand-in below retires a worker and starts its
    replacement in the same places the real pool does, so a loop that stops submitting stops the run.
    """
    CHUNK, WORKERS = 3, 2

    class RetiringPool:
        """The pool as far as this matters: a worker lives for CHUNK tasks, a replacement starts on submit."""

        def __init__(self, max_workers, max_tasks_per_child=None, **kwargs):
            self.workers, self.chunk = max_workers, max_tasks_per_child or len(sources)
            self.live, self.queued = [], []
            self.outstanding, self.peak, self.ran = 0, 0, 0

        def __enter__(self):
            return self

        def __exit__(self, *problem):
            return False

        def submit(self, task, *args):
            future = Future()
            self.queued.append((future, task, args))
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
            if len(self.live) < self.workers:  # the replacement the real pool starts here and nowhere else
                self.live.append(self.chunk)
            return future

        def work(self):
            finished = set()
            while self.queued and any(left > 0 for left in self.live):
                worker = next(i for i, left in enumerate(self.live) if left > 0)
                future, task, args = self.queued.pop(0)
                self.live[worker] -= 1
                self.ran += 1
                future.set_result(task(*args))
                finished.add(future)
                self.outstanding -= 1
            self.live = [left for left in self.live if left > 0]
            return finished

    pools = []

    def build(**kwargs):
        pools.append(RetiringPool(**kwargs))
        return pools[-1]

    def waitFor(futures, return_when=None, timeout=None):
        finished = pools[-1].work()
        assert finished, "the run waited on a pool whose workers have all retired, with nothing submitted"
        return finished, {future for future in futures if future not in finished}

    monkeypatch.setattr(run, "ProcessPoolExecutor", build)
    monkeypatch.setattr(run, "wait", waitFor)
    monkeypatch.setattr(run, "runPartition", lambda source: 1)
    sources = [(i, f"{i}.parquet") for i in range(20)]
    total, failed = run.fitPartitions(sources, workers=WORKERS, chunk=CHUNK, initargs=())

    assert (total, failed) == (20, 0), "the run did not fit every partition"
    assert pools[-1].ran == 20 and not pools[-1].queued
    assert pools[-1].peak <= run.WINDOW * WORKERS, f"{pools[-1].peak} partitions were in the pool at once"
    assert pools[-1].peak > WORKERS, "a worker had nothing waiting for it while the run had work left"


def test_a_prepared_catalog_with_no_extinction_column_is_refused_before_the_pool(run):
    """A column that is only missed inside a worker arrives as a crashed pool instead of a sentence."""
    prepared = run.FIT_COLUMNS + ["Ar_SFD"]
    assert run.inputColumns(prepared, "Ar_SFD") == run.FIT_COLUMNS + ["Ar_SFD"]
    assert run.inputColumns(run.FIT_COLUMNS + ["ebv"], "Ar") == run.FIT_COLUMNS + ["ebv"]
    with pytest.raises(SystemExit, match="extinction"):
        run.inputColumns(prepared, "Ar")
    with pytest.raises(SystemExit, match="missing"):
        run.inputColumns(["objectId", "coord_ra"], "Ar")


def test_a_catalog_is_asked_for_extinction_only_where_the_fit_reads_it(run):
    """The flat A_r prior reads no extinction, so a prepared catalog carrying none is not its problem.

    arColumn is None for that mode, all the way from the command line to the frame the fit is handed, so
    nothing asks the catalog for a column and nothing makes an A_r that would never be looked at.
    """
    prepared = run.FIT_COLUMNS + ["something_else"]
    assert run.inputColumns(prepared, None) == run.FIT_COLUMNS, "a column nothing reads was asked for"
    assert run.inputColumns(run.FIT_COLUMNS, None) == run.FIT_COLUMNS

    frame = pd.DataFrame({c: np.zeros(2) for c in run.FIT_COLUMNS})
    stars = run.prepareStars(frame, {}, None)
    assert "Ar" not in stars.columns, "an extinction was made for a fit that does not read one"
    assert list(stars["dustIndex"]) == [0, 0], "a run with no curves still looked a sightline up"
    with pytest.raises(SystemExit, match="extinction"):
        run.prepareStars(frame, {}, "Ar")


def test_a_resume_that_asks_for_another_fit_is_refused(run, tmp_path):
    """A partition already written is kept whatever this run asked for, so the two have to agree.

    Nothing in a catalog says which settings each of its partitions was fitted with, so a resume under other
    settings cannot be seen afterwards, let alone undone. The cone is part of that: it keeps whole
    partitions outside itself, and inside the ones it keeps it is a filter on single stars, so resuming a
    wider run with a narrower cone leaves partitions full of stars the narrower run would never have kept.
    """
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    # what the refusal protects is the answers already written, so the catalog holds one
    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (7, 8)], strict=True))
    run.writePartition(base, HealpixPixel(5, 1), frame)
    settings = SETTINGS
    wanted = run.runConfiguration(SimpleNamespace(**settings))
    run.recordConfiguration(base, wanted)
    assert json.loads((base / run.RUN_FILE).read_text()) == wanted, "the run recorded something else"
    run.recordConfiguration(base, wanted)  # the same command again, which is how a run is resumed

    for name, value in (
        ("ar_max", 4.0),
        ("floor", 0.05),
        ("no_dust_map", True),
        ("dust_curves", "/data/other.npz"),
        ("cone", [30.0, 15.0, 1.0]),
        ("catalog", "/data/dp3"),
    ):
        other = run.runConfiguration(SimpleNamespace(**(settings | {name: value})))
        with pytest.raises(SystemExit, match="--overwrite"):
            run.recordConfiguration(base, other)

    # how the work is divided up does not change an answer, so a run must be free to finish elsewhere
    divided = settings | {"workers": 8, "chunk": 10, "batch_size": 4000, "batch_bytes": 1 << 30}
    run.recordConfiguration(base, run.runConfiguration(SimpleNamespace(**divided)))
    # and the same catalog named with the trailing slash the shell completes it with is the same catalog
    completed = run.runConfiguration(SimpleNamespace(**(settings | {"catalog": "/data/dp2/"})))
    assert completed == wanted, "a path the shell completed read as another catalog"
    run.recordConfiguration(base, completed)
    # and the flat A_r prior reads no curves, so naming some of them is the same run either way
    flat = run.runConfiguration(SimpleNamespace(**(settings | {"no_dust_map": True})))
    bare = run.runConfiguration(SimpleNamespace(**(settings | {"no_dust_map": True, "dust_curves": ""})))
    assert flat == bare and flat["dustCurves"] == "" and flat["arColumn"] is None


def test_a_directory_with_no_answers_in_it_takes_the_settings_it_is_given(run, tmp_path, capsys):
    """What the refusal protects is the answers, and a run can exit before it has written any.

    A prior file it cannot read or a column the catalog has not got ends a run after it has recorded itself,
    and the corrected command would then be refused by the empty directory the first one left.
    """
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    base.mkdir()
    first = run.runConfiguration(SimpleNamespace(**(SETTINGS | {"priors": "/data/mistyped.npz"})))
    run.recordConfiguration(base, first)
    wanted = run.runConfiguration(SimpleNamespace(**SETTINGS))
    run.recordConfiguration(base, wanted)
    assert json.loads((base / run.RUN_FILE).read_text()) == wanted, "the corrected command was not recorded"
    assert capsys.readouterr().out == "", "an empty directory was reported as holding answers"

    # and once there are answers the settings they were made with are what a resume has to agree with
    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (7, 8)], strict=True))
    run.writePartition(base, HealpixPixel(5, 1), frame)
    with pytest.raises(SystemExit, match="--overwrite"):
        run.recordConfiguration(base, first)


def test_a_result_with_no_record_of_its_settings_is_taken_as_it_comes(run, tmp_path, capsys):
    """The one case with nothing to compare against: answers written before this was recorded."""
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    frame = estimatesFrame(*zip(*[pixelCentre(p, 5) for p in (7, 8)], strict=True))
    run.writePartition(base, HealpixPixel(5, 1), frame)
    wanted = {"floor": 0.03}
    run.recordConfiguration(base, wanted)

    assert "no record" in capsys.readouterr().out, "a resume of an older result said nothing about it"
    assert json.loads((base / run.RUN_FILE).read_text()) == wanted


def test_a_partition_is_written_where_the_dataset_scan_cannot_see_it(run, tmp_path, monkeypatch):
    """A temporary file inside the dataset breaks the metadata step of this run and of every later one.

    hats.io.write_parquet_metadata reads every parquet file under the dataset directory, so one truncated
    file there is enough, and nothing in the catalog says which file it is. Only a kill between the write
    and the rename can leave one, so what is checked here is the name the writer writes under: it has to be
    outside the dataset, and one directory up is the same filesystem, so the rename stays atomic.
    """
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0), HealpixPixel(3, 5), HealpixPixel(3, 40)]
    for pixel in pixels[:2]:
        inside = [pixelCentre(pixel.pixel * 4 + corner, 4) for corner in range(4)]
        run.writePartition(base, pixel, estimatesFrame(*zip(*inside, strict=True)))

    seen = []
    writeAtomically = run.writeAtomically

    def watched(path, write, scratch=None):
        def peek(name):
            seen.append(Path(name))
            return write(name)

        return writeAtomically(path, peek, scratch=scratch)

    monkeypatch.setattr(run, "writeAtomically", watched)
    inside = [pixelCentre(pixels[2].pixel * 4 + corner, 4) for corner in range(4)]
    run.writePartition(base, pixels[2], estimatesFrame(*zip(*inside, strict=True)))
    assert seen[0].parent == base / run.PARTIAL, "the temporary file was written inside the catalog dataset"

    # the file put back where the write had it, as a kernel kill between the write and the rename leaves it
    seen[0].write_bytes(b"PAR1 and then the run was killed")
    assert run.writeCatalogMetadata(base, "photod", pixels, 12) is True, "a leftover broke the metadata step"
    assert run.sweepOrphans(base) == 0, "there was nothing in the dataset to sweep"


def test_an_orphan_of_an_earlier_version_is_swept_and_the_failure_is_visible(run, tmp_path, capsys):
    """Earlier runs wrote the temporary file beside the partition, where it breaks every run that follows."""
    from hats.pixel_math import HealpixPixel

    base = tmp_path / "photod"
    pixels = [HealpixPixel(3, 0)]
    inside = [pixelCentre(corner, 4) for corner in range(4)]
    path = run.writePartition(base, pixels[0], estimatesFrame(*zip(*inside, strict=True)))
    orphan = path.with_name("Npix=40-ab12cd34.parquet")
    orphan.write_bytes(b"PAR1 and then the run was killed")

    assert run.writeCatalogMetadata(base, "photod", pixels, 4) is False, "the orphan did not break anything"
    assert "the catalog index is not" in capsys.readouterr().out, "the failure was not reported"
    assert run.sweepOrphans(base) == 1 and not orphan.exists(), "the orphan was not swept"
    assert run.writeCatalogMetadata(base, "photod", pixels, 4) is True, "sweeping did not repair the catalog"
    assert path.exists(), "sweeping took a real partition with it"


def test_the_pool_replaces_its_workers_only_where_the_interpreter_can(run, monkeypatch):
    """max_tasks_per_child arrived in Python 3.11 and this package supports 3.10, where it is a TypeError."""
    import multiprocessing

    monkeypatch.setattr(run, "WORKERS_REPLACED", True)
    options = run.poolOptions(400)
    assert options["max_tasks_per_child"] == 400, "the workers are no longer replaced"
    assert options["mp_context"].get_start_method() == "spawn"
    assert "max_tasks_per_child" not in run.poolOptions(0), "a chunk of zero asked for replacement anyway"

    monkeypatch.setattr(run, "WORKERS_REPLACED", False)
    older = run.poolOptions(400)
    assert "max_tasks_per_child" not in older, "an older interpreter was handed an argument it has not got"
    assert isinstance(older["mp_context"], type(multiprocessing.get_context("spawn")))
    # the pool has to be buildable with exactly what poolOptions returns, on either interpreter
    run.ProcessPoolExecutor(max_workers=1, **older).shutdown()


def test_the_pool_says_when_it_cannot_replace_its_workers(run, monkeypatch, capsys):
    """A survey-wide run on 3.10 grows without bound, so it is worth one line rather than a surprise."""
    monkeypatch.setattr(run, "WORKERS_REPLACED", False)
    run.poolOptions(400)
    printed = capsys.readouterr().out
    assert "replace a worker" in printed and "memory" in printed, "nothing said the memory would grow"
    monkeypatch.setattr(run, "WORKERS_REPLACED", True)
    run.poolOptions(400)
    assert capsys.readouterr().out == "", "an interpreter that can replace a worker was warned anyway"


def test_the_batch_memory_budget_reaches_the_fit(run, monkeypatch):
    """--batch-bytes bounds one batch of one worker, and has to arrive at the call that makes the batch."""
    globalParams = SimpleNamespace(computeMrTrue=True, fitColors=("ug", "gr", "ri", "iz", "zy"))
    with np.load(DATA / "priors_dp2.npz") as data:
        index, order = np.asarray(data["index"]), int(data["order"])
    covered = int(np.where(index >= 0)[0][0])
    maps = np.zeros((int(index.max()) + 1, 4, 4, 4))
    asked = {}
    monkeypatch.setattr(run, "loadParams", lambda setup: globalParams)
    monkeypatch.setattr(
        run,
        "loadPriors",
        lambda path: {"index": index, "order": order, "kde": maps}
        | dict.fromkeys(("rmag", "xGrid", "yGrid"), np.zeros(4)),
    )
    monkeypatch.setattr(run, "priorGridFromMaps", lambda *args: {0: np.zeros(4)})

    def watched(stars, *args, **kwargs):
        asked.update(kwargs)
        return unfittedEstimates(stars, globalParams, 0), None

    monkeypatch.setattr(run, "makeBayesEstimates3d", watched)
    ra, dec = pixelCentre(covered, order)
    stars = pd.DataFrame(
        {"objectId": [0], "ra": [ra], "dec": [dec], "rmag": [20.0], "Ar": [0.0]}
        | {color + "Err": [0.02] for color in globalParams.fitColors}
    )
    run.fitPartition(stars, "priors.npz", (0.03, True, "", 8.0), 100, 1 << 30)

    assert asked == {"batchSize": 100, "batchBytes": 1 << 30}, "the batch budget did not reach the fit"
