"""The distance-aware A_r band on the white dwarf and QSO rows, the class chi2min and the goodness-of-fit
flags of photod.quality, and the scripts that build their inputs."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from test_dust_prior import bruteQuantiles, float64
from test_wd_classes import flatPrior
from wd_loci import COLORS, NAMES, script, singleMassLoci

import photod.locus as lt
from photod.bayes import (
    FLAG_POOR_FIT_P,
    FLAG_TOO_FEW_COLOURS,
    arBandEdges,
    makeBayesEstimates3d,
)
from photod.parameters import BAND_SEGMENTS, GlobalParams
from photod.quality import (
    LATITUDE_EDGES,
    QC_CLASSES,
    colourPattern,
    latitudeClass,
    patternColours,
    readQcTable,
)

MU = np.round(np.arange(2.0, 20.0001, 0.1), 6)


@pytest.fixture(scope="module")
def wdLocus(tmp_path_factory):
    """The stars with white dwarf tracks of one mass (and the tracks alone beside them)."""
    return singleMassLoci(tmp_path_factory.mktemp("bandloci"))[0]


def expoShape(absb=30.0, h=125.0):
    """One sightline of an exponential dust layer."""
    d = 10 ** (MU / 5 + 1)
    return (1 - np.exp(-d * np.sin(np.radians(absb)) / h)).astype(np.float32)


def bandParams(path, mode="band", shape=None, kFeH=10, **kwargs):
    """GlobalParams of a locus with the dust map, one dust curve and the curve mode (None: no curves)."""
    locus = lt.LSSTsimsLocus(fixForStripe82=False, datafile=path, colnames=NAMES)
    locusData = lt.subsampleLocusData(locus, kMr=1, kFeH=kFeH, yLabel="tLoc")
    curves = {}
    if mode is not None:
        shape = expoShape() if shape is None else shape
        curves = dict(ArCurves=shape[None, :], ArCurveMu=MU, ArCurveIndexColumn="dustIndex", ArCurveMode=mode)
    return GlobalParams(
        COLORS,
        locusData,
        lt.getArGridList(),
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArGridRange="Medium",
        ArMapColumn="Ar",
        colorErrFloor=0.03,
        **curves,
        **kwargs,
    )


def starsOn(gp, rows, ar, rmag, arMap=0.5, err=0.02, seed=3):
    """Stars with the colours of the given locus points reddened by ar, at r = rmag, with a little noise."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    stars = pd.DataFrame(
        {
            "objectId": np.arange(n),
            "ra": np.full(n, 150.0),
            "dec": np.full(n, 2.0),
            "rmag": np.asarray(rmag, dtype=float) * np.ones(n),
            "Ar": np.full(n, arMap),
            "dustIndex": np.zeros(n, dtype=np.int32),
        }
    )
    colours = gp.locusColors2d[rows] + np.asarray(ar, dtype=float).reshape(-1, 1) * gp.reddVector
    for k, c in enumerate(COLORS):
        stars[c] = colours[:, k] + rng.normal(0, err, n)
        stars[c + "Err"] = err
    return stars


def solveEdge(edge, MrTrue, rmag):
    """A = edge(r - MrTrue - A) by bisection, edge given on MU and held flat past its ends."""
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - np.interp(rmag - MrTrue - mid, MU, edge) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def brutePosterior(star, gp, shape):
    """The posterior of one star on the full grid, written out directly, and the chi2 of every cell."""
    Ar1d = np.asarray(gp.Ar1d, dtype=float)
    dA = Ar1d[1] - Ar1d[0]
    obs = np.array([star[c] for c in COLORS])
    err2 = np.array([star[c + "Err"] for c in COLORS]) ** 2 + gp.colorErrFloor**2
    model = gp.locusColors2d[:, None, :] + Ar1d[None, :, None] * gp.reddVector[None, None, :]
    chi2 = np.sum((obs - model) ** 2 / err2, axis=-1)
    arMax = gp.ArPriorScale * star["Ar"] + gp.ArPriorOffset
    allowed = np.broadcast_to(Ar1d <= arMax + 1e-12, chi2.shape).copy()
    column = shape.astype(float) * star["Ar"]
    lowEdge, highEdge = (
        column / gp.ArPriorScale - gp.ArPriorOffset,
        gp.ArPriorScale * column + gp.ArPriorOffset,
    )
    for i in np.where(gp.bandFlat)[0] if gp.bandFlat is not None else []:
        lo = max(solveEdge(lowEdge, gp.MrTrueFlat[i], star["rmag"]), 0.0)
        hi = solveEdge(highEdge, gp.MrTrueFlat[i], star["rmag"])
        allowed[i] &= (Ar1d >= lo - 1e-9 * dA) & (Ar1d <= hi + 1e-9 * dA)
    width = allowed.sum(axis=1) * dA
    cells = (gp.priorWeights * gp.locusValid)[:, None]
    post = np.exp(-0.5 * (chi2 - chi2.min())) * allowed * cells / width[:, None]
    post = post.reshape(gp.FeH1d.size, gp.Mr1d.size, Ar1d.size)
    chi2 = np.where(allowed & (cells > 0), chi2, np.inf).reshape(post.shape)
    return post, chi2


def test_the_band_follows_the_dust_in_front_of_each_point_and_is_normalised_over_it(wdLocus):
    """The fit agrees with the posterior written out directly: the band on the white dwarf rows, the box on
    the rest, each normalised over its own cells."""
    gp = bandParams(wdLocus)
    track = np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    dwarfs = np.where((gp.segmentFlat == 0) & gp.locusValid & (gp.MrTrueFlat > 4) & (gp.MrTrueFlat < 7))[0]
    rows = np.array([track[40], track[200], dwarfs[3], dwarfs[-3]])
    stars = starsOn(gp, rows, ar=[0.05, 0.2, 0.3, 0.1], rmag=[19.0, 21.0, 20.0, 18.0])
    with float64():
        estimates, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=4)
    for row, (_, star) in enumerate(stars.iterrows()):
        post, chi2 = brutePosterior(star, gp, expoShape())
        masks = [np.broadcast_to(m, (gp.FeH1d.size, gp.Mr1d.size)) for m in gp.classAxisMasks.values()]
        masses = [post[m].sum() for m in masks]
        assert_allclose(estimates["p_WD"][row], masses[1] / sum(masses), rtol=1e-6, atol=1e-9)
        for k, name in enumerate(QC_CLASSES):
            want = chi2[masks[k]].min() if masks[k].any() else np.inf
            got = estimates[f"chi2min{name}"][row]
            assert (np.isnan(got) and not np.isfinite(want)) or np.isclose(got, want, rtol=1e-6), name
        keep = masks[int(np.argmax(masses[:2]))]
        expected = bruteQuantiles(post * keep[:, :, None], gp)
        for name, values in expected.items():
            for q, value in zip(("lo", "median", "hi"), values, strict=True):
                assert_allclose(
                    estimates[f"{name}_quantile_{q}"][row],
                    value,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"{row} {name} {q}",
                )


def test_the_band_keeps_a_nearby_white_dwarf_in_front_of_the_dust(wdLocus):
    """A white dwarf 50 pc away gets no more A_r than the dust in front of it; the box allows the column."""
    only = wdLocus.parent / "LSSTlocus_WD_only.txt"
    gp = bandParams(only)
    box = bandParams(only, mode=None)
    point = int(
        np.argmin(np.abs(np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD, gp.MrTrueFlat, 99) - 13.0))
    )
    rmag, arMap = 13.0 + 3.5, 0.6
    stars = starsOn(gp, [point], ar=[0.4], rmag=[rmag], arMap=arMap, err=0.01)
    a, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=1)
    b, _ = makeBayesEstimates3d(stars, flatPrior(box), box, batchSize=1)
    _, top = arBandEdges(
        gp.MrTrueFlat[[point]], rmag, expoShape(), MU, arMap, 1.3, 0.1, np.array([True]), np.array([False])
    )
    assert float(top[0]) < 0.3
    assert a["Ar_quantile_hi"][0] <= float(top[0]) + gp.dAr
    assert abs(b["Ar_quantile_median"][0] - 0.4) < 0.05


def test_a_locus_without_band_rows_is_fitted_as_with_the_box():
    """On the stars alone every row keeps the box, and normalising it by its width changes nothing."""
    stars = lt.DEFAULT_LOCUS_FILE.parent / "LSSTlocus_10Gyr_DP2.txt"
    gp = bandParams(stars, kFeH=8)
    box = bandParams(stars, mode=None, kFeH=8)
    assert not gp.bandFlat.any()
    dwarfs = np.where(gp.locusValid & (gp.MrTrueFlat > 3) & (gp.MrTrueFlat < 9))[0]
    stars = starsOn(gp, dwarfs[::400][:6], ar=0.2, rmag=19.0)
    a, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=6)
    b, _ = makeBayesEstimates3d(stars, flatPrior(box), box, batchSize=6)
    for column in a.columns:
        if column.endswith(("_lo", "_median", "_hi", "_std")) or column.startswith(("p_", "chi2min")):
            assert_allclose(a[column], b[column], rtol=1e-5, atol=1e-5, err_msg=column)


def test_a_star_the_dust_map_has_no_value_for_is_fitted_as_with_the_box(qsoLoci):
    """Without a map value there is no column to bound a band by: the white dwarf and QSO points keep the box
    as the stars do, so the star comes out as it does without the curves."""
    gp = bandParams(qsoLoci[1])
    box = bandParams(qsoLoci[1], mode=None)
    qso = np.where((gp.segmentFlat == lt.SEGMENT_QSO) & gp.locusValid)[0]
    track = np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    dwarfs = np.where((gp.segmentFlat == 0) & gp.locusValid & (gp.MrTrueFlat > 4) & (gp.MrTrueFlat < 7))[0]
    rows = np.array([qso[40], qso[100], track[200], dwarfs[3]])
    stars = starsOn(gp, rows, ar=[0.3, 0.6, 0.2, 0.3], rmag=20.0, arMap=np.nan)
    a, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=4)
    b, _ = makeBayesEstimates3d(stars, flatPrior(box), box, batchSize=4)
    assert a["p_QSO"].iloc[1] > 0.5
    for column in a.columns:
        if column.endswith(("_lo", "_median", "_hi", "_std")) or column.startswith(("p_", "chi2min")):
            assert_allclose(a[column], b[column], rtol=1e-5, atol=1e-5, err_msg=column)
    assert np.array_equal(a["fit_flags"], b["fit_flags"])


def test_the_band_edges():
    """Each edge solves A = e(r - Mr - A); a QSO takes the whole column; other rows and bare sky keep the
    box."""
    shape = expoShape(10.0)
    arMap, r, scale, offset = 0.8, 20.0, 1.3, 0.1
    MrTrue = np.array([10.0, 12.0, 14.0, 5.0, -30.0])
    band = np.array([True, True, True, False, True])
    extragalactic = np.array([False, False, False, False, True])
    with float64():
        lo, hi = (
            np.asarray(x)
            for x in arBandEdges(MrTrue, r, shape, MU, arMap, scale, offset, band, extragalactic)
        )
    column = shape.astype(float) * arMap
    for i in range(3):
        assert_allclose(hi[i], solveEdge(scale * column + offset, MrTrue[i], r), atol=1e-6)
        assert_allclose(lo[i], max(solveEdge(column / scale - offset, MrTrue[i], r), 0.0), atol=1e-6)
    assert hi[0] > hi[1] > hi[2]  # the brighter, the farther, the more dust in front
    assert (lo[3], hi[3]) == (0.0, np.inf)
    assert_allclose((lo[4], hi[4]), (arMap / scale - offset, scale * arMap + offset))
    with float64():
        lo, hi = arBandEdges(MrTrue, r, np.zeros_like(shape), MU, arMap, scale, offset, band, extragalactic)
    assert np.all(np.asarray(lo)[:4] == 0) and np.all(np.isinf(np.asarray(hi)[:4]))


def test_a_band_or_qc_table_of_the_wrong_kind_is_refused(wdLocus, tmp_path):
    """A band with no curves, a curve mode that does not exist, and a table naming other colours."""
    run = script("run_dp2")
    with pytest.raises(SystemExit, match="needs the curves"):
        run.globalParameters(0.03, True, None, 2.0, wdLocus, curveMode="band")
    with pytest.raises(ValueError, match="needs dust curves"):
        bandParams(wdLocus, mode=None, ArCurveMode="band")
    with pytest.raises(ValueError, match="ArCurveMode"):
        bandParams(wdLocus, mode="cap")
    path = tmp_path / "qc.csv"
    table = pd.DataFrame(
        {
            "class": ["WD"],
            "latitude": [0],
            "pattern": [7],
            "colours": ["ug,gr,iz"],
            "nColors": [3],
            "chi2": [9.0],
        }
        | {"level": ["cell"], "n": [10], "nEff": [10.0]}
    )
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="another set"):
        readQcTable(path, COLORS)
    table.assign(colours="ug,gr,ri").to_csv(path, index=False)
    thresholds = readQcTable(path, COLORS)
    assert thresholds.shape == (3, len(LATITUDE_EDGES) + 1, 32)
    assert (
        thresholds[QC_CLASSES.index("WD"), 0, 7] == 9.0 and np.isnan(thresholds).sum() == thresholds.size - 1
    )


def test_the_quality_flags_test_the_claimed_class_and_count_the_colours(wdLocus):
    """FLAG_POOR_FIT_P is set from the claimed class's chi2min against its own threshold; too few colours is
    flagged whatever the table says."""
    plain = bandParams(wdLocus, mode=None)
    nLat, nPattern = len(LATITUDE_EDGES) + 1, 1 << len(COLORS)
    thresholds = np.full((len(QC_CLASSES), nLat, nPattern), np.inf)
    thresholds[QC_CLASSES.index("WD")] = -1.0  # every white dwarf call fails, no star does
    gp = bandParams(wdLocus, mode=None, qcThresholds=thresholds)
    track = np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    dwarfs = np.where((gp.segmentFlat == 0) & gp.locusValid & (gp.MrTrueFlat > 5) & (gp.MrTrueFlat < 7))[0]
    stars = starsOn(gp, np.concatenate([track[[50, 150]], dwarfs[[10]], track[[100]]]), ar=0.0, rmag=19.0)
    for c in ("ug", "iz", "zy"):
        stars.loc[3, c + "Err"] = 9.99
    a, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=4)
    b, _ = makeBayesEstimates3d(stars, flatPrior(plain), plain, batchSize=4)
    calledWD = a["p_WD"].to_numpy() > 0.5
    assert calledWD[:2].all() and not calledWD[2]
    assert np.array_equal((a["fit_flags"] & FLAG_POOR_FIT_P) > 0, calledWD)
    assert not (b["fit_flags"] & FLAG_POOR_FIT_P).any()
    assert np.array_equal(a["fit_flags"] & ~FLAG_POOR_FIT_P, b["fit_flags"])
    assert a["nColors"].tolist() == [5, 5, 5, 2]
    assert ((a["fit_flags"] & FLAG_TOO_FEW_COLOURS) > 0).tolist() == [False, False, False, True]
    assert np.isnan(a["chi2minQSO"]).all() and (a["chi2minWD"] >= a["chi2min"] - 1e-4).all()


def test_the_pattern_and_latitude_of_a_star():
    """The measured colours as a bitmask, and the |b| class from the position."""
    errors = np.array([[0.1, 0.1, np.inf, 0.1, np.inf], [np.inf] * 5, [0.1] * 5])
    assert colourPattern(errors).tolist() == [1 + 2 + 8, 0, 31]
    assert patternColours(11, COLORS) == "ug,gr,iz"
    from astropy.coordinates import SkyCoord

    b = np.array([5.0, -25.0, 60.0, 39.9])
    c = SkyCoord(l=np.full(4, 100.0), b=b, unit="deg", frame="galactic").icrs
    assert latitudeClass(c.ra.deg, c.dec.deg).tolist() == [0, 1, 2, 1]


def test_the_band_and_the_qc_table_are_recorded_with_the_run(tmp_path):
    """Each is left out of the record when unused; a table goes with its md5."""
    run = script("run_dp2")
    settings = dict(
        catalog="/data/dp2",
        priors="/data/priors.npz",
        floor=0.03,
        ar_column="Ar",
        ar_scale=1.0,
        ar_max=8.0,
        no_dust_map=False,
        dust_curves="/data/expo.npz",
        cone=None,
        no_input_columns=False,
    )
    before = run.runConfiguration(SimpleNamespace(**settings))
    assert not {"dustCurveMode", "qcTable"} & before.keys()
    qc = tmp_path / "qc.csv"
    qc.write_text("class,latitude,pattern,colours,nColors,chi2,level,n,nEff\n")
    asked = settings | {"dust_curve_mode": "band", "qc_table": str(qc)}
    now = run.runConfiguration(SimpleNamespace(**asked))
    assert now["dustCurveMode"] == "band" and now["dustCurveRows"] == list(BAND_SEGMENTS)
    assert now["qcTable"] == str(qc) and now["qcTableMd5"] == run.fileMd5(qc)
    json.dumps(now)
    flat = run.runConfiguration(SimpleNamespace(**(asked | {"no_dust_map": True})))
    assert "dustCurveMode" not in flat


def test_the_exponential_dust_curves():
    """1 - exp(-d sin|b| / h) on each |b| row, and every pixel sent to the row of its own |b|."""
    expo = script("make_expo_dust_curves")
    shapes, mu, index, rows = expo.curves(scaleHeight=125.0, nside=4, muMin=2.0, muMax=20.0)
    d = 10 ** (mu / 5 + 1)
    k = int(round(30 / 0.1))
    assert_allclose(shapes[k], 1 - np.exp(-d * 0.5 / 125.0), rtol=1e-6)
    assert (np.diff(shapes, axis=1) >= 0).all() and shapes[0, 0] > 0
    import astropy.units as u
    import cdshealpix
    from astropy.coordinates import SkyCoord

    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.arange(12 * 16), 2)
    absb = np.abs(SkyCoord(ra=lon.to(u.deg), dec=lat.to(u.deg)).galactic.b.deg)
    assert np.all(np.abs(rows[index] - absb) <= 0.05 + 1e-9)


def test_the_prior_cube_of_a_star_is_the_band(wdLocus):
    """The cubes returned for inspection hold the same band: no prior outside a point's allowed cells."""
    only = wdLocus.parent / "LSSTlocus_WD_only.txt"
    gp = bandParams(only)
    point = int(np.argmin(np.abs(gp.MrTrueFlat - 13.0)))
    stars = starsOn(gp, [point], ar=[0.0], rmag=[16.5], arMap=0.6)
    _, results = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=1, returnPosteriors=True)
    prior = np.asarray(results.priorCube)[0].reshape(-1, gp.Ar1d.size)
    lo, hi = (
        np.asarray(x)
        for x in arBandEdges(
            gp.MrTrueFlat, 16.5, expoShape(), MU, 0.6, 1.3, 0.1, gp.bandFlat, gp.extragalacticFlat
        )
    )
    outside = (gp.Ar1d[None, :] > hi[:, None] + 1e-6) | (gp.Ar1d[None, :] < lo[:, None] - 1e-6)
    assert prior[outside].max() == 0 and prior[~outside].max() > 0
    assert np.isfinite(np.asarray(results.likeCube)).all()
