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
    FLAG_NO_MAGNITUDE,
    FLAG_NO_PRIOR,
    FLAG_POOR_FIT,
    FLAG_TWO_BRANCHES,
    getEstimatesMeta,
    makeBayesEstimates3d,
    unfittedEstimates,
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
    """A partition that holds stars and one that holds none have to describe themselves the same way.

    Written a partition at a time, they end up in one dataset, so the types have to agree as well as the
    names: the positions are carried over from the catalog and must not be rounded to float32.
    """
    catalog, priorGrid, params = setup()
    meta = getEstimatesMeta(params.computeMrTrue)
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    empty, _ = makeBayesEstimates3d(catalog.iloc[:0], priorGrid, params)
    assert list(estimates.columns) == list(meta.columns)
    assert list(empty.columns) == list(meta.columns)
    assert estimates.dtypes.to_dict() == meta.dtypes.to_dict()
    assert empty.dtypes.to_dict() == meta.dtypes.to_dict()
    assert str(estimates[bayes.cc.right_ascension].dtype) == "float64"


def test_the_value_of_an_unmeasured_colour_barely_matters():
    """A colour marked unmeasured keeps the weight its error asks for, and 9.99 mag asks for almost none.

    Almost none, rather than none, and only against a value a colour could really take: the placeholder is
    still a measurement of a sort, so one far outside the range of stellar colours would pull the answer.
    """
    catalog, priorGrid, params = setup()
    catalog.loc[1, "ugErr"] = 9.99
    catalog.loc[1, "ug"] = 0.0  # what scripts/run_dp2.py writes for a colour it could not measure
    reference, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    columns = [c for c in reference.columns if c.endswith(("_lo", "_median", "_hi"))]
    for value in (1.5, 3.0):
        catalog.loc[1, "ug"] = value
        estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
        moved = np.abs(estimates[columns].to_numpy(float) - reference[columns].to_numpy(float))
        assert moved.max() < 0.01, f"a placeholder of {value} moved a percentile by {moved.max():.3f}"


def test_a_star_without_a_magnitude_keeps_its_row_and_no_estimate():
    """Without r there is no prior map and no distance, and the brightest map's answer would be a lie."""
    catalog, priorGrid, params = setup()
    catalog.loc[2, "rmag"] = np.nan
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert len(estimates) == len(catalog)
    assert flags[2] & FLAG_NO_MAGNITUDE and flags[2] & FLAG_POOR_FIT
    assert not np.isfinite(estimates[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()[2])
    assert not np.isfinite(estimates[bayes.cc.chi_sq_min].to_numpy()[2])
    assert np.all(np.isfinite(estimates[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()[[0, 1, 3]]))


def test_extinction_pinned_at_the_dust_map_bound_is_flagged():
    """The bound that binds is the map's, not the grid's: a star pinned against it is a limit, not a fit."""
    catalog, priorGrid, params = setup(n=1)
    catalog["Ar"] = 0.1  # so the prior stops at 1.3 * 0.1 + 0.1, far below the top of the grid
    catalog[["ug", "gr", "ri"]] = [[2.4, 1.5, 0.85]]  # a star much redder than that allows
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=1)
    ar = estimates[f"{bayes.cc.extinction_r}_quantile_median"].to_numpy()[0]
    assert ar > 0.5 * 0.23 and ar <= 0.23 + 1e-6, f"A_r came out at {ar:.3f}, not against its bound"
    assert float(params.Ar1d[-1]) > 1.0, "the grid must reach far past the bound for this to mean anything"
    assert estimates[bayes.cc.quality_flags].to_numpy()[0] & FLAG_AR_EDGE


def test_the_entropy_drop_is_in_bits():
    """The entropies are of densities, so each carries the width of its own bin."""
    catalog, priorGrid, params = setup(n=2)
    estimates, results = makeBayesEstimates3d(catalog, priorGrid, params, returnPosteriors=True)
    for star in range(len(catalog)):
        posterior = np.asarray(results.margpostAr[2][star], dtype=float)
        prior = np.asarray(results.margpostAr[0][star], dtype=float)
        bits = (
            lambda p: -np.sum(np.where(p > 0, p, 1) * np.log2(np.where(p > 0, p, 1))) * params.dAr
        )  # noqa: E731
        assert_allclose(
            estimates[bayes.cc.extinction_r_entropy_drop].to_numpy()[star],
            bits(posterior) - bits(prior),
            rtol=2e-3,
            atol=2e-3,
        )


def test_a_star_the_fit_never_saw_is_kept_and_flagged():
    """A star whose part of the sky has no prior map cannot be fitted, and is not quietly left out either."""
    catalog, _, params = setup(n=3)
    rows = unfittedEstimates(catalog, params, FLAG_NO_PRIOR)
    meta = getEstimatesMeta(params.computeMrTrue)
    assert list(rows.columns) == list(meta.columns) and rows.dtypes.to_dict() == meta.dtypes.to_dict()
    assert len(rows) == len(catalog)
    assert np.all(rows[bayes.cc.quality_flags].to_numpy() & FLAG_NO_PRIOR)
    assert np.all(rows[bayes.cc.quality_flags].to_numpy() & FLAG_POOR_FIT)
    assert_allclose(rows[bayes.cc.object_id].to_numpy(), catalog["objectId"].to_numpy())
    assert not np.any(np.isfinite(rows[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()))


def test_an_unfitted_row_still_says_what_was_measured_of_the_star():
    """Bits 16 and 32 describe the input, so counting by them must not depend on which rows were fitted."""
    catalog, _, params = setup(n=3)
    catalog.loc[0, "rmag"] = np.nan
    catalog.loc[1, "ugErr"] = 9.99
    flags = unfittedEstimates(catalog, params, FLAG_NO_PRIOR)[bayes.cc.quality_flags].to_numpy()
    assert flags[0] & FLAG_NO_MAGNITUDE and not flags[0] & FLAG_COLOR_MISSING
    assert flags[1] & FLAG_COLOR_MISSING and not flags[1] & FLAG_NO_MAGNITUDE
    assert flags[2] == FLAG_NO_PRIOR | FLAG_POOR_FIT
    assert np.all(flags & FLAG_NO_PRIOR)


def test_a_single_extinction_has_no_edge_to_be_pinned_against():
    """With A_r held at one value every star sits on it, which is not the same as being against a limit."""
    catalog, priorGrid, params = setup(n=4)
    fixed = GlobalParams(
        COLORS,
        params.locusData,
        {"ArFixed": np.array([0.2])},
        {"ArFixed": params.locus3DList["ArFixed"]},
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArGridRange="Fixed",
    )
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, fixed, batchSize=4)
    assert not np.any(estimates[bayes.cc.quality_flags].to_numpy() & FLAG_AR_EDGE)


def test_a_batch_is_kept_within_its_memory_budget():
    """batchBytes bounds one batch, whatever the number of stars asked for."""
    catalog, priorGrid, params = setup(n=16)
    cells = 12 * params.FeH1d.size * params.Mr1d.size * params.Ar1d.size
    assert bayes._batchLimit(params, params.Ar1d.size, 3 * cells) == 3, "the budget is not what bounds it"
    whole, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=16)
    tiny, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=16, batchBytes=3 * cells)
    assert_allclose(tiny.to_numpy(float), whole.to_numpy(float), rtol=1e-5, atol=1e-5, equal_nan=True)


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
    """Two surviving branches show up as a posterior far longer on one side of its median than the other.

    Measured on the absolute magnitude, not on the tLoc the locus is parametrised by, since that is where a
    giant and a dwarf solution sit far apart.
    """
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    low, median, high = (estimates[f"Mr_true_quantile_{q}"].to_numpy() for q in ("lo", "median", "hi"))
    ratio = (high - median) / np.where(median - low > 0, median - low, np.nan)
    flagged = (estimates[bayes.cc.quality_flags].to_numpy() & FLAG_TWO_BRANCHES) > 0
    assert np.all(flagged[np.isfinite(ratio) & (ratio > 3.0)])
    assert not np.any(flagged[np.isfinite(ratio) & (ratio > 0.5) & (ratio < 2.0)])
