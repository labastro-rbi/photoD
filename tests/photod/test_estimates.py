import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import photod.locus as lt
import photod.bayes as bayes
from photod.bayes import (
    CHI2_POOR,
    FLAG_AR_EDGE,
    FLAG_COLOR_MISSING,
    FLAG_FEH_EDGE,
    FLAG_POOR_FIT,
    FLAG_TWO_BRANCHES,
    getEstimatesMeta,
    makeBayesEstimates3d,
)
from photod.parameters import GlobalParams
from photod.priors import getBayesConstants

COLORS = ("ug", "gr", "ri")


def setup(n=16, **columns):
    """A small tLoc locus, a flat prior grid and a handful of stars."""
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
    )
    priorGrid = np.ones((getBayesConstants()["rmagNsteps"], params.FeH1d.size * params.Mr1d.size))
    rng = np.random.default_rng(7)
    catalog = pd.DataFrame(
        {
            "objectId": np.arange(n),
            "ra": np.zeros(n),
            "dec": np.zeros(n),
            "rmag": rng.uniform(17.0, 22.0, n),
            "Ar": np.full(n, 2.0),
            "ug": rng.uniform(1.0, 2.5, n),
            "gr": rng.uniform(0.6, 1.6, n),
            "ri": rng.uniform(0.3, 0.9, n),
            "ugErr": np.full(n, 0.05),
            "grErr": np.full(n, 0.02),
            "riErr": np.full(n, 0.02),
        }
    )
    for name, value in columns.items():
        catalog[name] = value
    return catalog, priorGrid, params


def test_the_columns_are_the_ones_the_meta_promises():
    """A partition that holds stars and one that holds none have to describe themselves the same way."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    assert list(estimates.columns) == list(getEstimatesMeta(params.computeMrTrue).columns)
    assert len(makeBayesEstimates3d(catalog.iloc[:0], priorGrid, params)[0].columns) == len(estimates.columns)


def test_the_distance_modulus_is_the_magnitude_less_the_reddened_absolute_one():
    """DM = r - (Mr + A_r), and Qr is the posterior of Mr + A_r, so the quantiles swap ends."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    rmag = catalog["rmag"].to_numpy()
    for low, high in (("lo", "hi"), ("median", "median"), ("hi", "lo")):
        assert_allclose(
            estimates[f"{bayes.cc.distance_modulus}_quantile_{low}"],
            rmag - estimates[f"{bayes.cc.abs_mag_ext_r}_quantile_{high}"],
            atol=1e-6,
        )
    assert np.all(
        estimates[f"{bayes.cc.distance_modulus}_quantile_lo"]
        <= estimates[f"{bayes.cc.distance_modulus}_quantile_hi"]
    )


def test_a_star_the_locus_cannot_fit_is_flagged_and_still_answered():
    """Nothing is dropped for being unfittable: the row stays and says so."""
    catalog, priorGrid, params = setup()
    catalog.loc[0, ["ug", "gr", "ri"]] = [-5.0, 5.0, -5.0]
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert estimates[bayes.cc.chi_sq_min].to_numpy()[0] > CHI2_POOR
    assert flags[0] & FLAG_POOR_FIT
    assert np.isfinite(estimates[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()[0])
    assert len(estimates) == len(catalog)


def test_a_star_with_no_answer_is_not_mistaken_for_a_good_one():
    """A row whose fit produced nothing has to say so rather than come back with an empty flag."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    median = estimates[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert np.all(flags[~np.isfinite(median)] & FLAG_POOR_FIT)


def test_a_colour_with_no_measurement_is_flagged():
    """The fit gives an unmeasured colour no weight, and the star records that it had one."""
    catalog, priorGrid, params = setup()
    catalog.loc[1, "ugErr"] = 9.99
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert flags[1] & FLAG_COLOR_MISSING
    assert not flags[2] & FLAG_COLOR_MISSING


def test_the_edges_of_the_grid_are_flagged_as_limits():
    """A parameter pinned against the end of the model grid is a limit, not a measurement."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    feH = estimates[f"{bayes.cc.metallicity}_quantile_median"].to_numpy()
    ar = estimates[f"{bayes.cc.extinction_r}_quantile_median"].to_numpy()
    step = np.diff(params.FeH1d)[0]
    atEdge = (feH <= params.FeH1d[0] + step) | (feH >= params.FeH1d[-1] - step)
    assert_allclose((flags & FLAG_FEH_EDGE) > 0, atEdge)
    assert_allclose((flags & FLAG_AR_EDGE) > 0, ar >= params.Ar1d[-1] - params.dAr)


def test_a_lopsided_posterior_is_flagged():
    """Two surviving branches show up as a posterior far longer on one side of its median than the other."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    low, median, high = (
        estimates[f"{bayes.cc.abs_mag_r}_quantile_{q}"].to_numpy() for q in ("lo", "median", "hi")
    )
    ratio = (high - median) / np.where(median - low > 0, median - low, np.nan)
    flagged = (estimates[bayes.cc.quality_flags].to_numpy() & FLAG_TWO_BRANCHES) > 0
    assert np.all(flagged[np.isfinite(ratio) & (ratio > 3.0)])
    assert not np.any(flagged[np.isfinite(ratio) & (ratio > 0.5) & (ratio < 2.0)])
