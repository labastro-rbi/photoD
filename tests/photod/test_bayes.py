import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose
from scipy.stats import gaussian_kde

import photod.locus as lt
from photod.bayes import _starBatch, makeBayesEstimates3d
from photod.parameters import GlobalParams
from photod.priors import get2Dmap, getBayesConstants, getPriorMapIndex, initializePriorGrid
from photod.stats import multimodality

try:
    from jax import enable_x64

    def float64():
        """Context in which JAX computes in float64."""
        return enable_x64(True)

except ImportError:
    from jax.experimental import enable_x64 as float64

COLORS = ("ug", "gr", "ri")


def make_locus(tLoc=False):
    """Small smooth locus on a regular grid; with tLoc=True the grid column is tLoc and Mr differs below 4."""
    FeH1d = np.arange(-2.0, 0.01, 0.5)
    y1d = np.arange(1.0, 10.01, 0.25)
    FeH, y = np.meshgrid(FeH1d, y1d, indexing="ij")
    table = Table()
    if tLoc:
        table["tLoc"] = y.ravel()
        table["Mr"] = np.where(y > 4, y, 4 - 2.5 * (4 - y) + 0.2 * FeH).ravel()
    else:
        table["Mr"] = y.ravel()
    table["FeH"] = FeH.ravel()
    table["ug"] = (0.8 + 0.15 * y + 0.2 * FeH + 0.02 * y**2).ravel()
    table["gr"] = (0.2 + 0.08 * y + 0.05 * FeH).ravel()
    table["ri"] = (0.05 + 0.06 * y + 0.01 * FeH * y).ravel()
    return table


def make_params(locus, tLoc=False, ArGridRange="Small", ArMapColumn="ArMap"):
    """GlobalParams for a test locus."""
    yLabel = "tLoc" if tLoc else "Mr"
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS, xLabel="FeH", yLabel=yLabel)
    return GlobalParams(
        COLORS,
        locus,
        ArGridList,
        locus3DList,
        xLabel="FeH",
        yLabel=yLabel,
        MrColumn=yLabel,
        ArGridRange=ArGridRange,
        computeMrTrue=tLoc,
        ArMapColumn=ArMapColumn,
    )


def make_stars(locus, n=12, seed=3):
    """Stars with locus colors, reddening and noise."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(locus), n)
    Ar = rng.uniform(0.02, 0.2, n)
    C = lt.extcoeff()
    stars = pd.DataFrame(
        {
            "objectId": np.arange(n),
            "ra": rng.uniform(0, 1, n),
            "dec": rng.uniform(0, 1, n),
            "rmag": rng.uniform(15, 26, n),
            "ArMap": Ar,
        }
    )
    for c in COLORS:
        err = rng.uniform(0.02, 0.1, n)
        stars[c] = np.asarray(locus[c])[rows] + Ar * (C[c[0]] - C[c[1]]) + rng.normal(0, err)
        stars[c + "Err"] = err
    return stars


def multimodality_by_hand(x, p, nLevels=300):
    """The second largest excess over any cut of the density, the runs of bins above each cut split out."""
    edges = np.concatenate([[1.5 * x[0] - 0.5 * x[1]], 0.5 * (x[1:] + x[:-1]), [1.5 * x[-1] - 0.5 * x[-2]]])
    width = np.diff(edges)
    density = p / p.sum() / width
    best = 0.0
    for level in np.linspace(0, density.max(), nLevels)[1:-1]:
        above = density >= level
        runs = np.split(np.arange(density.size), np.flatnonzero(np.diff(above)) + 1)
        excess = sorted(np.sum((density[r] - level) * width[r]) for r in runs if above[r[0]])
        best = max([best, *excess[-2:-1]])
    return best


def brute_force(star, params, priorGrid):
    """Posterior statistics from the full (FeH, Mr, Ar) cube, written out directly."""
    nFeH, nMr, A = params.FeH1d.size, params.Mr1d.size, params.Ar1d
    model = np.stack([params.locus3DList[f"Ar{params.ArGridRange}"][c] for c in COLORS], axis=-1)
    obs, err = star[list(COLORS)].to_numpy(float), star[[c + "Err" for c in COLORS]].to_numpy(float)
    chi2 = np.sum(((obs - model) / err) ** 2, axis=-1)
    arMax = np.inf if params.ArMapColumn is None else 1.3 * star["ArMap"] + 0.1
    allowed = (arMax >= A) | (A.size == 1)
    prior = priorGrid[getPriorMapIndex(star["rmag"])].reshape(nFeH, nMr)
    post = prior[:, :, None] * np.exp(-0.5 * (chi2 - chi2.min())) * allowed

    def quantiles(x, p):
        cdf = (np.cumsum(p) - 0.5 * p) / p.sum()
        return np.interp([0.16, 0.5, 0.84], cdf, x)

    def histogram(values, weights):
        grid, index = np.unique(np.round(values, 3), return_inverse=True)
        return grid, np.bincount(index.ravel(), weights.ravel(), minlength=grid.size)

    def entropy(p, dx):
        """Entropy of a sampled density, in bits, so with the width of its own bin."""
        p = p[p > 0]
        return -np.sum(p * np.log2(p)) * dx

    pnorm = lambda p, dx: p / p.sum() / dx  # noqa: E731
    MrTrue = np.broadcast_to(params.Mr1d, (nFeH, nMr))
    if params.computeMrTrue:
        MrTrue = np.where(params.Mr1d > 4, params.Mr1d, params.MrTrueTable)
    margs = {
        "tLoc": (params.Mr1d, pnorm(post.sum(axis=(0, 2)), params.dMr)),
        "FeH": (params.FeH1d, pnorm(post.sum(axis=(1, 2)), params.dFeH)),
        "Ar": (A, pnorm(post.sum(axis=(0, 1)), params.dAr)),
        "Qr": histogram(MrTrue[:, :, None] + A, post),
    }
    if params.computeMrTrue:
        margs["Mr"] = histogram(MrTrue, post.sum(axis=2))
    out = {"chi2min": chi2.min()}
    for name, (x, p) in margs.items():
        out.update(
            zip([f"{name}_quantile_{q}" for q in ("lo", "median", "hi")], quantiles(x, p), strict=True)
        )
    out["MrdS"] = entropy(margs["tLoc"][1], params.dMr) - entropy(
        pnorm(prior.sum(axis=0), params.dMr), params.dMr
    )
    out["FeHdS"] = entropy(margs["FeH"][1], params.dFeH) - entropy(
        pnorm(prior.sum(axis=1), params.dFeH), params.dFeH
    )
    out["ArdS"] = entropy(margs["Ar"][1], params.dAr) - entropy(pnorm(allowed * 1.0, params.dAr), params.dAr)
    out["multimodality"] = multimodality_by_hand(*margs["Qr"])
    return out


@pytest.mark.parametrize("tLoc", [False, True])
@pytest.mark.parametrize("ArGridRange,ArMapColumn", [("Small", "ArMap"), ("Small", None), ("Fixed", None)])
def test_estimates_match_brute_force(tLoc, ArGridRange, ArMapColumn):
    """The fast computation gives the statistics of the full posterior cube.

    Read from the results rather than from the frame, which carries the float32 of the output schema.
    """
    locus = make_locus(tLoc)
    params = make_params(locus, tLoc, ArGridRange, ArMapColumn)
    stars = make_stars(locus)
    priorGrid = np.random.default_rng(0).uniform(0.1, 1.0, (getBayesConstants()["rmagNsteps"], len(locus)))
    with float64():
        _, results = makeBayesEstimates3d(stars, priorGrid, params, batchSize=5)
    for i, star in stars.iterrows():
        for name, value in brute_force(star, params, priorGrid).items():
            computed = results.chi2min if name == "chi2min" else results.statistics[name]
            assert_allclose(computed[i], value, rtol=1e-9, atol=1e-9, err_msg=name)


def test_posteriors_match_estimates():
    """returnPosteriors gives the same statistics, and cubes whose marginals are normalized."""
    locus = make_locus(True)
    params = make_params(locus, True)
    stars = make_stars(locus, n=3)
    priorGrid = np.ones((getBayesConstants()["rmagNsteps"], len(locus)))
    estimates, _ = makeBayesEstimates3d(stars, priorGrid, params)
    withCubes, results = makeBayesEstimates3d(stars, priorGrid, params, returnPosteriors=True)
    assert_allclose(
        withCubes.drop(columns="chi2min"), estimates.drop(columns="chi2min"), rtol=1e-5, atol=1e-5
    )
    assert results.postCube.shape == (3, params.FeH1d.size, params.Mr1d.size, params.Ar1d.size)
    for k in range(3):
        assert_allclose(results.margpostMr[k].sum(axis=1) * params.dMr, 1, rtol=1e-5)


def test_empty_partition():
    """An empty partition gives an empty frame with the output columns."""
    locus = make_locus()
    estimates, _ = makeBayesEstimates3d(
        make_stars(locus).iloc[:0], np.ones((27, len(locus))), make_params(locus)
    )
    assert len(estimates) == 0 and "tLoc_quantile_median" in estimates.columns


def test_small_partitions():
    """Partitions smaller than batchSize give the same estimates and do not compile one batch size each."""
    locus = make_locus()
    params = make_params(locus, ArMapColumn=None)
    stars = make_stars(locus, n=20)
    priorGrid = np.ones((getBayesConstants()["rmagNsteps"], len(locus)))
    with float64():
        whole, _ = makeBayesEstimates3d(stars, priorGrid, params, batchSize=64)
        parts = [
            makeBayesEstimates3d(stars.iloc[a:b], priorGrid, params, batchSize=64)[0]
            for a, b in ((0, 5), (5, 11), (11, 18), (18, 20))
        ]
        assert_allclose(pd.concat(parts).to_numpy(float), whole.to_numpy(float), rtol=1e-9, atol=1e-9)
        compiled = _starBatch._cache_size()
        for n in (6, 7, 8):
            makeBayesEstimates3d(stars.iloc[:n], priorGrid, params, batchSize=64)
        assert _starBatch._cache_size() == compiled


def test_a_tLoc_grid_needs_the_true_absolute_magnitude():
    """Otherwise Qr and the distance modulus would be built from tLoc and look like any other answer."""
    locus = make_locus(True)
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS, xLabel="FeH", yLabel="tLoc")
    with pytest.raises(ValueError, match="computeMrTrue"):
        GlobalParams(COLORS, locus, ArGridList, locus3DList, xLabel="FeH", yLabel="tLoc", MrColumn="tLoc")


def test_the_dust_prior_needs_a_width_where_there_is_no_dust():
    """The map puts no extinction in front of the nearest stars, so a zero floor is a prior of zero width."""
    locus = make_locus(True)
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS, xLabel="FeH", yLabel="tLoc")
    with pytest.raises(ValueError, match="ArCurveFloor"):
        GlobalParams(
            COLORS,
            locus,
            ArGridList,
            locus3DList,
            xLabel="FeH",
            yLabel="tLoc",
            MrColumn="tLoc",
            computeMrTrue=True,
            ArCurves=np.ones((1, 5), dtype=np.float32),
            ArCurveMu=np.linspace(4, 16, 5),
            ArCurveIndexColumn="dustIndex",
            ArCurveFloor=0.0,
        )


def test_the_locus_rows_have_to_be_the_grid_they_claim():
    """The table is reshaped as it stands, so another row order would build a mirrored model in silence."""
    locus = make_locus()
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS)
    shuffled = locus[::-1]
    with pytest.raises(ValueError, match="ascending"):
        GlobalParams(COLORS, shuffled, ArGridList, locus3DList)
    with pytest.raises(ValueError, match="rectangular"):
        GlobalParams(COLORS, locus[:-1], ArGridList, locus3DList)


def test_subsampling_keeps_the_end_of_each_axis():
    """The reddest dwarfs and the metal-rich end are the two places a model must not quietly stop."""
    locus = make_locus(True)
    for kMr, kFeH in ((3, 2), (7, 3), (1, 1)):
        sub = lt.subsampleLocusData(locus, kMr, kFeH, yLabel="tLoc")
        for label, k in (("tLoc", kMr), ("FeH", kFeH)):
            values, whole = np.unique(np.asarray(sub[label])), np.unique(np.asarray(locus[label]))
            assert values[-1] == whole[-1], f"{label} stops at {values[-1]} of {whole[-1]} for k={k}"
            assert values[0] == whole[0]
        assert len(sub) == np.unique(np.asarray(sub["FeH"])).size * np.unique(np.asarray(sub["tLoc"])).size


def test_reddening_follows_color_names():
    """Reddening is added to each fitted color by name, whatever the column order of the locus."""
    locus = make_locus(True)
    locus = locus[["FeH", "ri", "tLoc", "gr", "Mr", "ug"]]
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS, xLabel="FeH", yLabel="tLoc")
    C = lt.extcoeff()
    model, Ar = locus3DList["ArSmall"], ArGridList["ArSmall"]
    for c in COLORS:
        assert_allclose(model[c] - model[c][:, :, :1], np.broadcast_to(Ar * (C[c[0]] - C[c[1]]), model.shape))
    for c in ("FeH", "tLoc", "Mr"):
        assert_allclose(model[c], np.broadcast_to(model[c][:, :, :1], model.shape))


def test_locus_plateau():
    """Points that only repeat the next point along the grid are flagged, the last of them is kept."""
    locus = make_locus(True)
    first = locus["tLoc"] <= 1.5
    for name in ("Mr", *COLORS):
        column = np.asarray(locus[name]).reshape(5, -1)
        column[:, :3] = column[:, 2:3]
        locus[name] = column.ravel()
    plateau = lt.locusPlateau(locus, "FeH", "tLoc")
    assert np.array_equal(plateau, np.asarray(first & (locus["tLoc"] < 1.5)))
    params = make_params(locus, True)
    assert np.array_equal(params.locusValid, ~plateau)


def test_stripe82_fix_is_off_by_default(locus_file_path):
    """The Stripe 82 color corrections are only applied on request."""
    datafile = locus_file_path
    raw = Table.read(datafile, format="ascii", names=["Mr", "FeH", "ug", "gr", "ri", "iz", "zy"])
    locus = lt.LSSTsimsLocus(datafile=datafile)
    assert_allclose(locus["ug"], raw["ug"])
    assert not np.allclose(lt.LSSTsimsLocus(fixForStripe82=True, datafile=datafile)["ug"], raw["ug"])


def test_prior_grid_interpolation():
    """Maps are matched to r bins by magnitude and interpolated bilinearly onto the locus."""
    locus = make_locus()
    params = make_params(locus)
    xgrid, ygrid = np.linspace(-2.5, 1.0, 36), np.linspace(17, -2, 96)
    X, Y = np.meshgrid(xgrid, ygrid)
    rows = [
        {"rmag": r, "kde": (r + X + 2 * Y).tobytes(), "xGrid": X.tobytes(), "yGrid": Y.tobytes()}
        for r in (14.0, 20.0, 27.0)
    ]
    priorGrid = initializePriorGrid(pd.DataFrame(rows), params)
    FeH, Mr = np.asarray(locus["FeH"]), np.asarray(locus["Mr"])
    assert_allclose(priorGrid[0], 14 + FeH + 2 * Mr)
    assert_allclose(priorGrid[12], 20 + FeH + 2 * Mr)
    assert_allclose(priorGrid[26], 27 + FeH + 2 * Mr)


def test_binned_kde_matches_gaussian_kde():
    """The binned KDE of the prior maps agrees with scipy's gaussian_kde."""
    rng = np.random.default_rng(1)
    sample = pd.DataFrame({"FeH": rng.normal(-1.0, 0.4, 100000), "Mr": rng.normal(6.0, 2.0, 100000)})
    bc = getBayesConstants()
    metadata = [bc["FeHmin"], bc["FeHmax"], bc["FeHNpts"], bc["MrFaint"], bc["MrBright"], bc["MrNpts"]]
    X, Y, Z = get2Dmap(sample, ["FeH", "Mr"], metadata)
    reference = gaussian_kde(np.vstack([sample["FeH"], sample["Mr"]]))(np.vstack([X.ravel(), Y.ravel()]))
    assert np.max(np.abs(Z - reference)) < 0.01 * reference.max()


def bumps(x, centres, widths, weights):
    """A posterior made of Gaussian peaks."""
    peaks = zip(centres, widths, weights, strict=True)
    return sum(w * np.exp(-0.5 * ((x - c) / s) ** 2) / s for c, s, w in peaks)


def multimodality_of(x, p):
    """The multimodality of one posterior."""
    return float(multimodality(x, np.asarray(p)[None, :])[0])


def test_a_single_peak_has_no_multimodality():
    """However skewed, long-tailed or flat-topped, a posterior with one peak has nothing in a second one."""
    x = np.arange(0.0, 20.0, 0.01)
    skewed = np.where(x < 8, np.exp(-0.5 * ((x - 8) / 0.3) ** 2), np.exp(-(x - 8) / 3.0))
    for p in (bumps(x, [8], [0.4], [1]), skewed, ((x > 5) & (x < 9)) * 1.0):
        assert multimodality_of(x, p) == 0.0


def test_the_multimodality_of_two_peaks_approaches_the_weight_of_the_smaller():
    """It grows as two peaks move apart and stops a little short of the weight in the smaller one.

    Two widths apart the smaller peak is only a shoulder on the larger, the posterior has one peak, and the
    multimodality is zero.
    """
    x = np.arange(0.0, 20.0, 0.01)
    gaps = (0.6, 1.0, 2.0, 6.0)
    values = [multimodality_of(x, bumps(x, [6, 6 + gap], [0.3, 0.3], [0.8, 0.2])) for gap in gaps]
    assert values[0] == 0 and 0 < values[1] < values[2] < values[3]
    assert 0.18 < values[-1] < 0.2


def test_the_second_peak_need_not_be_next_to_the_first():
    """Two equal peaks with a small one between them: what lies in a second peak is the other big one."""
    x = np.arange(0.0, 20.0, 0.01)
    assert 0.4 < multimodality_of(x, bumps(x, [4, 6, 8], [0.1, 0.1, 0.1], [0.45, 0.1, 0.45])) < 0.45


def test_the_multimodality_of_the_distance_modulus_is_that_of_qr():
    """DM = r - Qr turns the axis round, and neither that nor the scale of the weights changes it."""
    x = np.arange(-3.0, 24.0, 0.01)
    p = bumps(x, [4.0, 6.5, 9.0], [0.4, 0.2, 0.6], [0.5, 0.2, 0.3])
    assert_allclose(multimodality_of(22.7 - x[::-1], 5.0 * p[::-1]), multimodality_of(x, p), rtol=1e-5)


def test_the_multimodality_is_the_count_by_hand():
    """Over every cut, on evenly and unevenly spaced values, for posteriors with one to three peaks and noise.

    The posteriors of one grid go in as a batch, with an empty one among them.
    """
    rng = np.random.default_rng(5)
    with float64():
        for x in (np.arange(0.0, 20.0, 0.01), np.sort(rng.uniform(0.0, 20.0, 1500))):
            batch = []
            for n in (1, 2, 3, 2, 3, 2, 3, 3):
                p = bumps(x, rng.uniform(3, 17, n), rng.uniform(0.1, 1.0, n), rng.uniform(0.1, 1.0, n))
                batch.append(p * rng.uniform(0.9, 1.1, x.size))
            values = np.asarray(multimodality(x, np.array(batch + [np.zeros_like(x)])))
            assert_allclose(values[:-1], [multimodality_by_hand(x, p) for p in batch], rtol=1e-9, atol=1e-12)
            assert np.isnan(values[-1])


def test_calls_of_the_tLoc_parametrization():
    """Calls written for the tLoc parametrization still work: DSED is accepted, tLoc is added in place."""
    locus = make_locus(True)
    ArGridList, locus3DList = lt.get3DmodelList(locus, COLORS, DSED=True, xLabel="FeH", yLabel="tLoc")
    params = GlobalParams(COLORS, locus, ArGridList, locus3DList, "FeH", "tLoc", "tLoc", "Small", True)
    stars = pd.DataFrame({"FeH": [-1.0, -0.5, 0.0], "Mr": [6.0, 3.0, 1.5], "label": [1, 2, 3]})
    expected = lt.assignTLocPartition(stars, lt.buildSegmentData(params), params.FeH1d)["tLoc"]
    returned = lt.assignTLocFromLabel(stars, params, None, 4.0, "FeH", "Mr", "label", "tLoc")
    assert returned is stars
    assert_allclose(stars["tLoc"], expected)
