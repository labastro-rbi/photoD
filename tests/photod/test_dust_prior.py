import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import photod.locus as lt
from photod.bayes import makeBayesEstimates3d
from photod.parameters import GlobalParams
from photod.priors import getBayesConstants

try:
    from jax import enable_x64

    def float64():
        """Context in which JAX computes in float64."""
        return enable_x64(True)

except ImportError:
    from jax.experimental import enable_x64 as float64

COLORS = ("ug", "gr", "ri")
MU = np.arange(4.0, 16.01, 0.25)


def setup(**kwargs):
    """A small tLoc locus, a flat prior grid and a handful of stars, with the given A_r prior settings."""
    locus = lt.LSSTsimsLocus(
        fixForStripe82=False,
        datafile=lt.DEFAULT_LOCUS_FILE.parent / "LSSTlocus_10Gyr_DP2.txt",
        colnames=["tLoc", "Mr", "FeH", "ug", "gr", "ri", "iz", "zy"],
    )
    locusData = lt.subsampleLocusData(locus, 20, 3, yLabel="tLoc")
    ArGridList, locus3DList = lt.get3DmodelList(locusData, COLORS, yLabel="tLoc")
    params = GlobalParams(
        COLORS,
        locusData,
        ArGridList,
        locus3DList,
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArMapColumn="Ar",
        **kwargs,
    )
    nFeH, nMr = params.FeH1d.size, params.Mr1d.size
    priorGrid = np.ones((getBayesConstants()["rmagNsteps"], nFeH * nMr))
    rng = np.random.default_rng(4)
    n = 12
    catalog = pd.DataFrame(
        {
            "objectId": np.arange(n),
            "ra": np.zeros(n),
            "dec": np.zeros(n),
            "rmag": rng.uniform(17.0, 22.0, n),
            "Ar": np.full(n, 2.0),
            "dustIndex": np.zeros(n, dtype=np.int32),
            "ug": rng.uniform(1.0, 2.5, n),
            "gr": rng.uniform(0.6, 1.6, n),
            "ri": rng.uniform(0.3, 0.9, n),
            "ugErr": np.full(n, 0.05),
            "grErr": np.full(n, 0.02),
            "riErr": np.full(n, 0.02),
        }
    )
    return catalog, priorGrid, params


def brutePosterior(star, params, shape, arMapColumn="Ar"):
    """The posterior of one star on the full (locus, A_r) grid, written out directly.

    The prior from the 3D map is the density N(A_r; A*_i, sigma_i) of the extinction the map puts at the
    distance locus point i implies, so it carries its own normalisation, which varies with A*_i.
    """
    colors2d = np.asarray(params.locusColors2d, dtype=float)
    redd = np.asarray(params.reddVector, dtype=float)
    Ar1d = np.asarray(params.Ar1d, dtype=float)
    MrTrue = np.asarray(params.MrTrueFlat, dtype=float)
    curve = shape.astype(float) * star[arMapColumn]

    obs = np.array([star[c] for c in COLORS])
    err = np.array([star[c + "Err"] for c in COLORS])
    model = colors2d[:, None, :] + Ar1d[None, :, None] * redd[None, None, :]
    chi2 = np.sum(((obs - model) / err) ** 2, axis=-1)

    Astar = np.interp(MrTrue, (star["rmag"] - np.asarray(MU, dtype=float) - curve)[::-1], curve[::-1])
    w = 1.0 / ((params.ArCurveFrac * Astar) ** 2 + params.ArCurveFloor**2)
    logPrior = -0.5 * w[:, None] * (Ar1d[None, :] - Astar[:, None]) ** 2 + 0.5 * np.log(w)[:, None]
    allowed = Ar1d <= params.ArPriorScale * star[arMapColumn] + params.ArPriorOffset
    post = np.exp(logPrior - 0.5 * chi2 + 0.5 * chi2.min()) * allowed
    return post.reshape(params.FeH1d.size, params.Mr1d.size, Ar1d.size)


def bruteQuantiles(post, params):
    """The 14th, 50th and 86th percentiles of A_r, [Fe/H] and the true Mr of a posterior cube."""

    def quantiles(values, weights):
        order = np.argsort(values)
        values, weights = np.asarray(values)[order], np.asarray(weights)[order]
        cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
        return np.interp([0.14, 0.5, 0.86], cdf, values)

    MrTrue = np.zeros(np.asarray(params.MrTrueGrid).size)
    np.add.at(MrTrue, np.asarray(params.MrTrueIndices), post.sum(axis=2))
    return {
        "Ar": quantiles(params.Ar1d, post.sum(axis=(0, 1))),
        "FeH": quantiles(params.FeH1d, post.sum(axis=(1, 2))),
        "Mr_true": quantiles(params.MrTrueGrid, MrTrue),
    }


def test_the_prior_is_the_density_the_map_implies():
    """The fit must agree with the posterior written out directly, normalisation of the dust prior included.

    The width of that prior follows A_r itself, so leaving its normalisation out would weigh the locus points
    the map reddens most too heavily, and those are the distant, luminous solutions.
    """
    shape = np.clip((MU - 8.0) / 5.0, 0.0, 1.0).astype(np.float32)
    _, priorGrid, params = setup(
        ArCurves=shape[None, :], ArCurveMu=MU, ArCurveIndexColumn="dustIndex", ArCurveFrac=0.15
    )
    priorGrid = np.ones_like(priorGrid)
    rng = np.random.default_rng(13)
    index = int(np.argmin(np.abs(np.asarray(params.MrTrueFlat) - 5.0)))
    catalog = pd.DataFrame([modelStar(params, index, mu, 1.0, rng) for mu in (9.0, 11.0, 13.0)])
    with float64():
        estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=3)
    for row, (_, star) in enumerate(catalog.iterrows()):
        expected = bruteQuantiles(brutePosterior(star, params, shape), params)
        for name, values in expected.items():
            for q, value in zip(("lo", "median", "hi"), values, strict=True):
                assert_allclose(
                    estimates[f"{name}_quantile_{q}"].to_numpy()[row],
                    value,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"star {row}, {name} {q}",
                )


def test_a_flat_shape_reproduces_the_flat_prior():
    """A curve of zeros means no 3D map for that star, and the fit must be the one without curves at all."""
    catalog, priorGrid, flat = setup()
    reference, _ = makeBayesEstimates3d(catalog, priorGrid, flat, batchSize=12)
    _, _, withCurves = setup(
        ArCurves=np.zeros((1, MU.size), dtype=np.float32), ArCurveMu=MU, ArCurveIndexColumn="dustIndex"
    )
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, withCurves, batchSize=12)
    for column in reference.columns:
        if column.endswith(("_lo", "_median", "_hi")):
            assert_allclose(estimates[column], reference[column], atol=1e-5)


def modelStar(params, index, mu, ar, rng):
    """A star built from locus point `index` at distance modulus `mu` with extinction `ar`."""
    colors = np.asarray(params.locusColors2d)[index] + ar * np.asarray(params.reddVector)
    row = {
        "objectId": index,
        "ra": 0.0,
        "dec": 0.0,
        "Ar": 2.0,
        "dustIndex": 0,
        "rmag": float(params.MrTrueFlat[index]) + mu + ar,
    }
    for name, value in zip(COLORS, colors, strict=True):
        row[name], row[name + "Err"] = float(value) + rng.normal(0, 0.01), 0.02
    return row


def test_the_map_puts_the_extinction_where_the_distance_says_it_is():
    """With the dust beyond mu = 12, a star in front of it comes out unreddened and one behind reddened."""
    _, priorGrid, flat = setup()
    shape = np.where(MU < 12.0, 0.0, 1.0).astype(np.float32)[None, :]
    _, _, wall = setup(
        ArCurves=shape,
        ArCurveMu=MU,
        ArCurveIndexColumn="dustIndex",
        ArCurveFrac=0.05,
        ArCurveFloor=0.02,
    )
    rng = np.random.default_rng(7)
    index = int(np.argmin(np.abs(np.asarray(flat.MrTrueFlat) - 6.0)))
    catalog = pd.DataFrame([modelStar(flat, index, 10.0, 0.0, rng), modelStar(flat, index, 14.0, 2.0, rng)])
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, wall, batchSize=2)
    ar = estimates.Ar_quantile_median.to_numpy()
    assert ar[0] < 0.3, f"the star in front of the dust got A_r = {ar[0]:.2f}"
    assert abs(ar[1] - 2.0) < 0.4, f"the star behind the dust got A_r = {ar[1]:.2f}"


def test_dust_in_front_pins_the_extinction_to_the_map():
    """With all the dust in front of every star the prior is a Gaussian around A_r(map), so A_r narrows."""
    catalog, priorGrid, flat = setup()
    reference, _ = makeBayesEstimates3d(catalog, priorGrid, flat, batchSize=12)
    _, _, pinned = setup(
        ArCurves=np.ones((1, MU.size), dtype=np.float32),
        ArCurveMu=MU,
        ArCurveIndexColumn="dustIndex",
        ArCurveFrac=0.05,
        ArCurveFloor=0.02,
    )
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, pinned, batchSize=12)
    width = (estimates.Ar_quantile_hi - estimates.Ar_quantile_lo).to_numpy()
    reference_width = (reference.Ar_quantile_hi - reference.Ar_quantile_lo).to_numpy()
    assert np.nanmedian(width) < 0.5 * np.nanmedian(reference_width)
    assert np.nanmedian(np.abs(estimates.Ar_quantile_median.to_numpy() - 2.0)) < np.nanmedian(
        np.abs(reference.Ar_quantile_median.to_numpy() - 2.0)
    )


def test_the_prior_does_not_change_the_cost():
    """The 3D prior is another quadratic in A_r, so the fit keeps the same shape of work."""
    catalog, priorGrid, flat = setup()
    reference, _ = makeBayesEstimates3d(catalog, priorGrid, flat, batchSize=12)
    _, _, withCurves = setup(
        ArCurves=np.full((1, MU.size), 0.5, dtype=np.float32), ArCurveMu=MU, ArCurveIndexColumn="dustIndex"
    )
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, withCurves, batchSize=12)
    assert list(estimates.columns) == list(reference.columns)
    assert len(estimates) == len(catalog)
    assert np.all(np.isfinite(estimates.Mr_true_quantile_median.to_numpy()))


def test_extinction_above_the_grid_is_pinned_to_its_top():
    """A star redder than the top of the A_r grid has its A_r pinned there, so the grid must reach the field.

    At A_r(map) = 5.5 the standard grid, which stops at 2.5 mag, pins two stars in five and their distances go
    with them. Size the grid from the extinction of the field (scripts/run_dp2.py --ar-max).
    """
    _, priorGrid, params = setup()
    rng = np.random.default_rng(11)
    index = int(np.argmin(np.abs(np.asarray(params.MrTrueFlat) - 6.0)))
    top = float(params.Ar1d[-1])
    onGrid = pd.DataFrame([modelStar(params, index, 12.0, top - 0.5, rng)])
    onGrid["Ar"] = top - 0.5
    beyond = pd.DataFrame([modelStar(params, index, 12.0, top + 1.5, rng)])
    beyond["Ar"] = top + 1.5
    good, _ = makeBayesEstimates3d(onGrid, priorGrid, params, batchSize=1)
    bad, _ = makeBayesEstimates3d(beyond, priorGrid, params, batchSize=1)
    assert abs(good.Ar_quantile_median.to_numpy()[0] - (top - 0.5)) < 0.6
    assert bad.Ar_quantile_median.to_numpy()[0] > top - 0.1  # pinned at the edge, 1.5 mag short
    assert np.isfinite(bad.Mr_true_quantile_median.to_numpy()[0])  # pinned, not NaN
