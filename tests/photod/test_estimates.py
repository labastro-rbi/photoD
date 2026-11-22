import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import photod.bayes as bayes
import photod.locus as lt
from photod.bayes import (
    FLAG_AR_EDGE,
    FLAG_COLLAPSED,
    FLAG_COLOR_MISSING,
    FLAG_COLOR_MISSING_FIRST,
    FLAG_FEH_HIGH_EDGE,
    FLAG_FEH_LOW_EDGE,
    FLAG_NO_MAGNITUDE,
    FLAG_NO_PRIOR,
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
        datafile=lt.RUN_LOCUS_FILE,
        colnames=["tLoc", "Mr", "FeH", "ug", "gr", "ri", "iz", "zy"],
    )
    locusData = lt.subsampleLocusData(locus, 20, 3, yLabel="tLoc")
    params = GlobalParams(
        COLORS,
        locusData,
        lt.getArGridList(),
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


def test_the_multimodality_is_written_for_every_fitted_star():
    """A probability, and never more than half of one: the smaller of two peaks holds at most half."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    values = estimates[bayes.cc.multimodality].to_numpy(float)
    assert np.all((values >= 0) & (values <= 0.5))


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
    assert flags[2] & FLAG_NO_MAGNITUDE
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


def test_a_star_the_fit_never_saw_is_kept_and_flagged():
    """A star whose part of the sky has no prior map cannot be fitted, and is not quietly left out either."""
    catalog, _, params = setup(n=3)
    rows = unfittedEstimates(catalog, params, FLAG_NO_PRIOR)
    meta = getEstimatesMeta(params.computeMrTrue)
    assert list(rows.columns) == list(meta.columns) and rows.dtypes.to_dict() == meta.dtypes.to_dict()
    assert len(rows) == len(catalog)
    assert np.all(rows[bayes.cc.quality_flags].to_numpy() & FLAG_NO_PRIOR)
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
    assert flags[2] == FLAG_NO_PRIOR
    assert np.all(flags & FLAG_NO_PRIOR)


def test_a_single_extinction_has_no_edge_to_be_pinned_against():
    """With A_r held at one value every star sits on it, which is not the same as being against a limit."""
    catalog, priorGrid, params = setup(n=4)
    fixed = GlobalParams(
        COLORS,
        params.locusData,
        {"ArFixed": np.array([0.2])},
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
    # the shape columns are ratios of differences, which carry the last bits of float32 further
    shapes = [c for c in whole.columns if c.endswith(("_nongaussianity", "_asymmetry"))]
    rest = [c for c in whole.columns if c not in shapes]
    assert_allclose(
        tiny[rest].to_numpy(float), whole[rest].to_numpy(float), rtol=1e-5, atol=1e-5, equal_nan=True
    )
    assert_allclose(tiny[shapes].to_numpy(float), whole[shapes].to_numpy(float), rtol=1e-3, equal_nan=True)


def test_the_distance_modulus_is_the_magnitude_less_the_reddened_absolute_one():
    """DM = r - (Mr + A_r). Qr is that posterior, and it is no longer written, so read it inside."""
    catalog, priorGrid, params = setup()
    estimates, results = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8, returnPosteriors=True)
    rmag = catalog["rmag"].to_numpy()
    for low, high in (("lo", "hi"), ("median", "median"), ("hi", "lo")):
        assert_allclose(
            estimates[f"{bayes.cc.distance_modulus}_quantile_{low}"],
            rmag - results.statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_{high}"],
            atol=1e-6,
        )
    assert np.all(
        estimates[f"{bayes.cc.distance_modulus}_quantile_lo"]
        <= estimates[f"{bayes.cc.distance_modulus}_quantile_hi"]
    )


def test_a_star_the_locus_cannot_fit_is_answered_and_left_to_its_chi2():
    """How bad a fit is too bad depends on the science, so it is chi2min that says it, not a bit."""
    catalog, priorGrid, params = setup()
    catalog.loc[0, ["ug", "gr", "ri"]] = [-5.0, 5.0, -5.0]
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert estimates[bayes.cc.chi_sq_min].to_numpy()[0] > 100
    assert np.isfinite(estimates[f"{bayes.cc.abs_mag_r}_quantile_median"].to_numpy()[0])
    assert len(estimates) == len(catalog)
    assert not np.any(flags & 3), "a retired bit (1, a poor fit; 2, two branches) was set"


def test_a_colour_with_no_measurement_is_flagged():
    """The fit gives an unmeasured colour no weight, and the star records that it had one."""
    catalog, priorGrid, params = setup()
    catalog.loc[1, "ugErr"] = 9.99
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert flags[1] & FLAG_COLOR_MISSING
    assert flags[1] & FLAG_COLOR_MISSING_FIRST, "the bit of u-g, the first colour, is not set"
    assert not flags[1] & (FLAG_COLOR_MISSING_FIRST << 1 | FLAG_COLOR_MISSING_FIRST << 2)
    assert not flags[2] & FLAG_COLOR_MISSING


def test_a_colour_that_is_not_a_number_weighs_nothing_and_is_flagged():
    """A prepared catalog may write an unmeasured colour as NaN, and one NaN must not empty the whole star.

    The colour gets no weight at all, so the answer is the one of a colour whose error is enormous, and the
    star says it had a colour missing. A star with none of its colours is answered from its prior.
    """
    catalog, priorGrid, params = setup(n=4)
    catalog.loc[1, "ug"] = np.nan
    catalog.loc[2, "riErr"] = np.nan
    catalog.loc[3, ["ug", "gr", "ri"]] = np.nan
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=4)
    columns = [c for c in estimates.columns if c.endswith(("_lo", "_median", "_hi", "_std"))]
    assert np.all(np.isfinite(estimates[columns].to_numpy(float))), "a NaN colour emptied the star"
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    assert list(flags & FLAG_COLOR_MISSING > 0) == [False, True, True, True]
    perColour = [[(flag >> 14 + k) & 1 for k in range(len(COLORS))] for flag in flags]
    assert perColour == [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1]], "the bits name the wrong colours"

    weightless = catalog.copy()
    weightless.loc[1, ["ug", "ugErr"]] = [0.0, 1e6]
    weightless.loc[2, ["ri", "riErr"]] = [0.0, 1e6]
    reference, _ = makeBayesEstimates3d(weightless, priorGrid, params, batchSize=4)
    assert_allclose(estimates[columns].to_numpy(float)[:3], reference[columns].to_numpy(float)[:3], atol=1e-4)

    rows = unfittedEstimates(catalog, params, FLAG_NO_PRIOR)[bayes.cc.quality_flags].to_numpy()
    assert list(rows & FLAG_COLOR_MISSING > 0) == [False, True, True, True]
    colourBits = FLAG_COLOR_MISSING | sum(FLAG_COLOR_MISSING_FIRST << k for k in range(len(COLORS)))
    assert list(rows & colourBits) == list(flags & colourBits), "an unfitted row names other colours"


def test_a_star_with_no_dust_map_value_is_not_bounded_by_it():
    """A NaN in the dust-map column leaves A_r free over the whole grid, as a run without a map does."""
    catalog, priorGrid, params = setup(n=3)
    catalog.loc[1, "Ar"] = np.nan
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=3)
    unbounded = catalog.copy()
    unbounded.loc[1, "Ar"] = 1e3  # a bound far above the top of the grid
    reference, _ = makeBayesEstimates3d(unbounded, priorGrid, params, batchSize=3)
    columns = [c for c in estimates.columns if c.endswith(("_lo", "_median", "_hi", "_std"))]
    assert np.all(np.isfinite(estimates[columns].to_numpy(float)[1])), "a NaN dust-map value emptied the star"
    assert_allclose(estimates[columns].to_numpy(float), reference[columns].to_numpy(float), rtol=1e-6)


def test_the_edges_of_the_grid_are_flagged_as_limits():
    """A parameter pinned against the end of the model grid is a limit, not a measurement."""
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    flags = estimates[bayes.cc.quality_flags].to_numpy()
    feH = estimates[f"{bayes.cc.metallicity}_quantile_median"].to_numpy()
    ar = estimates[f"{bayes.cc.extinction_r}_quantile_median"].to_numpy()
    step = np.diff(params.FeH1d)[0]
    assert_allclose((flags & FLAG_FEH_LOW_EDGE) > 0, feH <= params.FeH1d[0] + step)
    assert_allclose((flags & FLAG_FEH_HIGH_EDGE) > 0, feH >= params.FeH1d[-1] - step)
    assert_allclose((flags & FLAG_AR_EDGE) > 0, ar >= params.Ar1d[-1] - params.dAr)


def test_the_shape_of_each_posterior_is_written_beside_it():
    """Non-gaussianity and asymmetry of Mr, [Fe/H], A_r and DM, read straight off the columns they come from.

    For Mr it is the true absolute magnitude, not the tLoc the locus is parametrised by, since that is where a
    giant and a dwarf solution sit far apart.
    """
    catalog, priorGrid, params = setup()
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    names = ["Mr", bayes.cc.metallicity, bayes.cc.extinction_r, bayes.cc.distance_modulus]
    assert bayes.shapeNames(params.computeMrTrue) == names
    for name in names:
        low, median, high = (estimates[f"{name}_quantile_{q}"].to_numpy(float) for q in bayes.QUANTILE_NAMES)
        std = estimates[f"{name}_std"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            assert_allclose(estimates[f"{name}_nongaussianity"], std / (0.5 * (high - low)), rtol=1e-4)
            assert_allclose(estimates[f"{name}_asymmetry"], (high - median) / (median - low), rtol=1e-4)
    columns = set(getEstimatesMeta(True).columns)
    assert {"Mr_nongaussianity", "Mr_asymmetry", "DM_nongaussianity", "DM_asymmetry"} <= columns
    assert f"{bayes.cc.abs_mag_r}_asymmetry" not in getEstimatesMeta(True).columns


def test_the_shape_of_a_gaussian_is_one_and_one():
    """A Gaussian has a non-gaussianity and an asymmetry of one; a long tail on one side does not."""
    statistics = {"x_quantile_lo": np.array([-1.0, -1.0]), "x_quantile_median": np.array([0.0, 0.0])}
    statistics |= {"x_quantile_hi": np.array([1.0, 3.0]), "x_std": np.array([1.0, 2.5])}
    shapes = bayes.shapeColumns(statistics, ["x"])
    assert_allclose(shapes["x_nongaussianity"], [1.0, 1.25])
    assert_allclose(shapes["x_asymmetry"], [1.0, 3.0])


def test_the_one_dimensional_posteriors_come_back_when_asked_for():
    """Each is normalised over its grid and gives back the median written for it."""
    catalog, priorGrid, params = setup(n=4)
    estimates, results = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=4, returnMarginals=True)
    plain, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=4)
    pd.testing.assert_frame_equal(estimates, plain)
    assert set(results.marginals) == set(results.marginalGrids)
    for name, pdf in results.marginals.items():
        grid = results.marginalGrids[name]
        assert pdf.shape == (len(catalog), grid.size)
        assert_allclose(pdf.sum(axis=1), 1.0, rtol=1e-5)
    for name in ("Mr", bayes.cc.metallicity, bayes.cc.extinction_r):
        grid, pdf = results.marginalGrids[name], results.marginals[name]
        cdf = np.cumsum(pdf, axis=1) - 0.5 * pdf
        median = np.array([np.interp(0.5, c, grid) for c in cdf])
        step = np.max(np.diff(grid))
        assert_allclose(median, estimates[f"{name}_quantile_median"].to_numpy(float), atol=step)


def test_the_r_error_widens_the_distance_modulus():
    """DM = r - Qr, so the distance is less certain than Qr by the error on r; the median does not move."""
    catalog, priorGrid, params = setup(n=8)
    exact, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=8)
    noisy = catalog.copy()
    noisy["rmagErr"] = 0.2
    widened, _ = makeBayesEstimates3d(noisy, priorGrid, params, batchSize=8)
    for end in ("lo", "hi"):
        narrow = np.abs(
            exact[f"{bayes.cc.distance_modulus}_quantile_{end}"].to_numpy(float)
            - exact[f"{bayes.cc.distance_modulus}_quantile_median"].to_numpy(float)
        )
        wide = np.abs(
            widened[f"{bayes.cc.distance_modulus}_quantile_{end}"].to_numpy(float)
            - widened[f"{bayes.cc.distance_modulus}_quantile_median"].to_numpy(float)
        )
        assert_allclose(wide, np.hypot(narrow, 0.2), rtol=1e-5)
    assert_allclose(
        exact[f"{bayes.cc.distance_modulus}_quantile_median"].to_numpy(float),
        widened[f"{bayes.cc.distance_modulus}_quantile_median"].to_numpy(float),
        rtol=1e-6,
    )
    # and a catalogue without the column is fitted as it was before the column was read
    assert "rmagErr" not in catalog.columns
    assert_allclose(bayes.magnitudeError(catalog), 0.0)


def test_a_collapsed_posterior_is_flagged():
    """A posterior inside a single grid cell looks like the best answer in the catalogue and is the worst."""
    catalog, priorGrid, params = setup(n=6)
    estimates, _ = makeBayesEstimates3d(catalog, priorGrid, params, batchSize=6)
    statistics = {
        name: estimates[name].to_numpy(float)
        for name in estimates.columns
        if name.endswith(("_lo", "_median", "_hi"))
    }
    # Qr is computed but no longer written, so the widths it carries are seeded rather than read.
    for end in ("lo", "median", "hi"):
        statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_{end}"] = np.full(len(estimates), 5.0)
    # Widths are set here rather than taken from the fit: this locus is a handful of grid points and the
    # colour errors are 0.02, so its posteriors are narrower than any real star's and would all be flagged.
    statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_lo"][:] = 5.0 - 2 * bayes.COLLAPSED_WIDTH
    statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_hi"][:] = 5.0 + 2 * bayes.COLLAPSED_WIDTH
    statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_lo"][::2] = 5.00
    statistics[f"{bayes.cc.abs_mag_ext_r}_quantile_hi"][::2] = 5.01
    flags = bayes._qualityFlags(
        statistics,
        np.full((6, len(params.fitColors)), 0.02),
        np.full(6, 8.0),
        np.zeros(6, dtype=bool),
        params,
    )
    assert np.all((flags[::2] & FLAG_COLLAPSED) > 0), "a collapsed posterior was not flagged"
    assert not np.any(flags[1::2] & FLAG_COLLAPSED), "a normal posterior was flagged"
