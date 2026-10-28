"""The white dwarf classes: the cell widths of the prior, the class probabilities and the estimates within a
class, the populations held to their shares, and the selection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from wd_loci import COLORS, NAMES, script

import photod.locus as lt
from photod.bayes import FLAG_BEYOND_RECIPE, WD_DISTANCE_MIN_P, makeBayesEstimates3d
from photod.parameters import GlobalParams, tLocCellWidths
from photod.priors import getBayesConstants
from photod.selection import RECIPE_R_MAX, bandSnr, recipeSelection


def params(path, kFeH=10):
    """GlobalParams of a locus on its full tLoc axis and a few [Fe/H] rows."""
    locus = lt.LSSTsimsLocus(fixForStripe82=False, datafile=path, colnames=NAMES)
    locusData = lt.subsampleLocusData(locus, kMr=1, kFeH=kFeH, yLabel="tLoc")
    return GlobalParams(
        COLORS,
        locusData,
        lt.getArGridList(),
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArGridRange="Small",
        colorErrFloor=0.03,
    )


def starsAt(gp, rows, rmag=20.0, err=0.02, seed=1):
    """Stars with the colours of the given locus points, A_r 0, a little noise."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    stars = pd.DataFrame(
        {"objectId": np.arange(n), "ra": np.zeros(n), "dec": np.zeros(n), "rmag": np.full(n, rmag)}
    )
    for k, c in enumerate(COLORS):
        stars[c] = gp.locusColors2d[rows, k] + rng.normal(0, err, n)
        stars[c + "Err"] = err
    return stars


def flatPrior(gp):
    """A prior of density one everywhere, for every r bin."""
    return np.ones((getBayesConstants()["rmagNsteps"], gp.FeH1d.size * gp.Mr1d.size))


def test_the_class_masks_are_the_ones_the_tLoc_tests_gave_on_this_locus(wdLoci):
    """On a white dwarf locus without a QSO track, the white dwarfs are tLoc < 0 and helium tLoc < -3."""
    gp = params(wdLoci[0])
    assert np.array_equal(gp.classAxisMasks["WD"], gp.Mr1d < 0)
    assert np.array_equal(gp.heliumAxisMask, gp.Mr1d < -3)
    assert np.array_equal(gp.classAxisMasks["MS"], gp.Mr1d >= 0)
    assert not gp.classAxisMasks["QSO"].any()


def test_the_cells_break_where_the_segment_changes_not_where_the_axis_has_a_gap():
    """Two segments with no gap between them are still two: each sums to its own number of steps."""
    # a white dwarf track that ends at -0.5 and a band of another class starting at -0.4: no gap of note
    pieces = [
        (1, np.linspace(-2.0, -0.5, 301)),
        (3, np.linspace(-0.4, 1.5, 191)),
        (0, np.linspace(1.65, 3.0, 136)),
    ]
    t = np.concatenate([values for _, values in pieces])
    segment = np.concatenate([np.full(values.size, code) for code, values in pieces])
    widths = tLocCellWidths(t, segment)
    for code, step in ((1, 0.005), (3, 0.01), (0, 0.01)):
        n = int(np.sum(segment == code))
        assert_allclose(widths[segment == code].sum(), n * step, rtol=1e-9)
    with pytest.raises(ValueError, match="single"):
        tLocCellWidths(np.array([0.0, 1.0, 1.01, 1.02]), np.array([1, 0, 0, 0]))


def test_the_prior_weights_give_each_segment_its_length(wdLoci):
    """A track of 301 points 0.005 apart is 1.505 long, and so are its weights."""
    gp = params(wdLoci[0])
    widths = gp.priorWeights.reshape(gp.FeH1d.size, gp.Mr1d.size)[0] / gp.dFeH
    for code in (1, 2):
        on = gp.segment1d == code
        assert_allclose(widths[on].sum(), on.sum() * 0.005, rtol=1e-6)
        assert_allclose(widths[on].sum(), np.ptp(gp.Mr1d[on]) + 0.005, rtol=1e-6)
    stars = gp.segment1d == 0
    assert_allclose(widths[stars].sum(), np.ptp(gp.Mr1d[stars]) + 0.01, rtol=1e-6)


def test_a_finer_stretch_of_the_axis_does_not_weigh_more():
    """The fit sums prior times likelihood over the grid points, so each point carries the width of its cell.

    The same locus with tLoc 7-8 sampled twice as finely, its colours interpolated, is the same model: with a
    flat prior density, stars at its edge get the answers they get on the even grid. Without the widths the
    finer half weighs twice as much and pulls them into it.
    """
    even = params(lt.RUN_LOCUS_FILE)
    t = even.Mr1d
    finer = np.union1d(t, np.round((t[:-1] + t[1:])[(t[:-1] >= 7.0) & (t[1:] <= 8.0)] / 2, 6))
    columns = {}
    for name in even.locusData.colnames:
        grid = np.asarray(even.locusData[name], dtype=float).reshape(even.FeH1d.size, t.size)
        columns[name] = np.stack([np.interp(finer, t, row) for row in grid]).ravel()
    from astropy.table import Table

    uneven = GlobalParams(
        COLORS,
        Table(columns),
        even.ArGridList,
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArGridRange="Small",
        colorErrFloor=0.03,
    )
    assert finer.size > t.size + 90
    feH = np.broadcast_to(even.FeH1d[:, None], (even.FeH1d.size, t.size)).ravel()
    tLoc = np.broadcast_to(t, (even.FeH1d.size, t.size)).ravel()
    rows = np.where(np.isclose(tLoc, 7.0) & np.isin(feH, even.FeH1d[[0, 1, 2]]))[0]
    stars = starsAt(even, rows, err=0.03)
    names = [f"{q}_quantile_{k}" for q in ("Mr", "tLoc", "FeH", "DM") for k in ("lo", "median", "hi")]
    a, _ = makeBayesEstimates3d(stars, flatPrior(even), even, batchSize=4)
    b, _ = makeBayesEstimates3d(stars, flatPrior(uneven), uneven, batchSize=4)
    assert_allclose(b[names], a[names], atol=2e-3)
    uneven.priorWeights = np.full_like(uneven.priorWeights, uneven.priorWeights.max())
    c, _ = makeBayesEstimates3d(stars, flatPrior(uneven), uneven, batchSize=4)
    assert np.abs(c["Mr_quantile_median"] - a["Mr_quantile_median"]).max() > 1e-2


def test_a_white_dwarf_only_locus_gives_p_wd_one(wdLoci):
    """With no other class on the locus every star is a white dwarf."""
    gp = params(wdLoci[1], kFeH=15)
    stars = starsAt(gp, [10, 200, 450])
    estimates, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=8)
    assert_allclose(estimates["p_WD"], 1.0, rtol=1e-6)
    assert_allclose(estimates["p_MS"], 0.0, atol=1e-6)


def test_the_class_probabilities_add_up_and_a_star_on_the_track_is_a_white_dwarf(wdLoci):
    """The class masses, the estimates within the class, the white dwarf distance and the prior odds."""
    gp = params(wdLoci[0])
    track = np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    # K dwarfs of solar [Fe/H], which no white dwarf resembles
    feH = np.broadcast_to(gp.FeH1d[:, None], (gp.FeH1d.size, gp.Mr1d.size)).ravel()
    dwarfs = (
        (gp.segmentFlat == 0) & gp.locusValid & (gp.MrTrueFlat > 6) & (gp.MrTrueFlat < 8) & (feH == feH.max())
    )
    mainSequence = np.where(dwarfs)[0]
    rows = np.concatenate([track[[20, 120, 200]], mainSequence[[10, -10]]])
    estimates, _ = makeBayesEstimates3d(starsAt(gp, rows), flatPrior(gp), gp, batchSize=8)
    assert_allclose(estimates["p_WD"] + estimates["p_MS"], 1.0, rtol=1e-5)
    p = estimates["p_WD"].to_numpy()
    assert np.all(p[:3] > 0.99) and np.all(p[3:] < 0.01)
    assert np.all(estimates["p_He"].to_numpy() <= p + 1e-6) and np.all(p <= 1 + 1e-6)
    lnodds = estimates["lnodds_WD"].to_numpy()
    assert np.all(lnodds[:3] > 0) and np.all(lnodds[3:] < 0)
    # the estimates are of the white dwarf branch for the first three and of the stars for the others
    tLoc = estimates["tLoc_quantile_median"].to_numpy()
    assert np.all(tLoc[:3] < 0) and np.all(tLoc[3:] > 0)
    # the white dwarf distance is given only where the class has enough of the posterior
    dmWD = estimates["DM_WD_quantile_median"].to_numpy()
    assert np.all(np.isfinite(dmWD[:3])) and np.all(np.isnan(dmWD[p <= WD_DISTANCE_MIN_P]))
    assert_allclose(dmWD[:3], estimates["DM_quantile_median"].to_numpy()[:3], atol=1e-4)
    assert estimates["nColors"].tolist() == [5] * 5
    # with a flat density the prior odds are the lengths of the two classes on the axis
    widths = gp.priorWeights * gp.locusValid
    isWD = gp.segmentFlat != 0
    assert_allclose(estimates["lnprior_odds_WD"], np.log(widths[isWD].sum() / widths[~isWD].sum()), rtol=1e-5)


def test_an_estimate_is_of_the_most_probable_class_and_not_of_the_mixture(wdLoci):
    """A star both classes describe: whatever the odds, its median sits on a branch, never between them."""
    gp = params(wdLoci[0])
    cool = np.where((gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD) & (gp.MrTrueFlat > 14.0))[0][:1]
    stars = starsAt(gp, cool, err=0.05)
    found = False
    for scale in 10.0 ** np.arange(-8, 9):
        prior = flatPrior(gp) * np.where(gp.segmentFlat == 0, 1.0, scale)
        estimates, _ = makeBayesEstimates3d(stars, prior, gp, batchSize=1)
        p = float(estimates["p_WD"].iloc[0])
        if 0.1 < p < 0.9:
            found = True
            tLoc = float(estimates["tLoc_quantile_median"].iloc[0])
            onBranch = tLoc >= gp.Mr1d[gp.segment1d == 0].min() if p < 0.5 else tLoc < 0
            assert onBranch, f"p_WD {p:.2f} and a median tLoc of {tLoc:.2f}"
            assert np.isfinite(estimates["DM_WD_quantile_median"].iloc[0])
    assert found, "no prior scale gave this star a posterior split between the classes"


def test_each_population_is_held_to_its_share():
    """Each population's mass on its own points is its share, to rounding; a share of zero is the floor."""
    run = script("run_dp2")
    gp = SimpleNamespace(
        locusValid=np.ones(12, bool),
        priorWeights=np.r_[np.full(4, 0.005), np.full(8, 0.01)],
        segmentFlat=np.r_[np.full(4, 1), np.full(8, 0)],
    )
    rng = np.random.default_rng(3)
    grid = {0: rng.uniform(0.1, 2, 12), 1: rng.uniform(0.1, 2, 12), 2: np.r_[np.zeros(4), np.ones(8)]}
    shares = np.array([[0.99, 0.01], [1.0, 0.0], [0.7, 0.3]])
    out = run.renormalisedPopulations(grid, shares, [0, 1], gp)
    for k in grid:
        for col, code in enumerate([0, 1]):
            mass = np.sum((out[k] * gp.priorWeights)[gp.segmentFlat == code])
            assert_allclose(mass, max(shares[k, col], run.SHARE_FLOOR), rtol=1e-12)
    # a share of zero is the floor, never a log of zero
    assert np.all(out[1][gp.segmentFlat == 1] > 0)
    # a population whose map is empty on its segment is spread evenly over it
    assert_allclose(out[2][:4], 0.3 / 0.02)


def test_the_selection_recipe_reads_the_same_from_fluxes_and_from_colour_errors():
    """The S/N of g and i recovered from the colour errors selects what the fluxes select."""
    rng = np.random.default_rng(11)
    n = 2000
    snr = {b: 10 ** rng.uniform(0.0, 2.0, n) for b in "gri"}
    flux = {b: np.full(n, 1e4) for b in "gri"}
    fluxes = {f"{b}_psfFlux": flux[b] for b in "gri"} | {f"{b}_psfFluxErr": flux[b] / snr[b] for b in "gri"}
    err = {b: 1.0857 / snr[b] for b in "gri"}
    prepared = {
        "rmagErr": err["r"],
        "grErr": np.hypot(err["g"], err["r"]),
        "riErr": np.hypot(err["r"], err["i"]),
    }
    rmag = rng.uniform(18, 26, n)
    a, beyondA = recipeSelection(rmag, bandSnr(fluxes))
    b, beyondB = recipeSelection(rmag, bandSnr(prepared))
    expected = (snr["r"] > 10) & (snr["g"] > 3) & (snr["i"] > 3)
    assert np.array_equal(a, expected) and np.array_equal(b, expected)
    assert np.array_equal(beyondA, expected & (rmag >= RECIPE_R_MAX)) and np.array_equal(beyondA, beyondB)
    # a missing error is not a pass
    prepared["rmagErr"][:5] = np.nan
    assert not recipeSelection(rmag, bandSnr(prepared))[0][:5].any()


def test_a_prepared_catalogue_is_held_to_the_recipe_and_the_faint_end_is_flagged():
    """Low S/N rows are not fitted; the faint end past r 23.5 is, flagged; the run records the choice."""
    run = script("run_dp2")
    frame = pd.DataFrame({c: np.zeros(4) for c in run.FIT_COLUMNS})
    frame["objectId"] = np.arange(4)
    frame["rmag"] = [20.0, 24.0, 22.0, 22.0]
    frame["rmagErr"] = [0.01, 0.05, 0.2, 0.01]  # S/N 109, 22, 5.4, 109
    frame["grErr"] = [0.02, 0.1, 0.25, 0.5]  # g S/N 56, 12, -, 2.2
    frame["riErr"] = 0.03
    kept = run.prepareStars(frame, {}, None, selection="recipe")
    assert kept["objectId"].tolist() == [0, 1]
    assert len(run.prepareStars(frame, {}, None, selection="none")) == 4, "--selection none dropped rows"
    gp = params(lt.RUN_LOCUS_FILE, kFeH=15)
    stars = kept.assign(**{c: 0.5 for c in COLORS}, **{c + "Err": 0.05 for c in COLORS})
    estimates, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=2)
    flags = estimates["fit_flags"].to_numpy()
    assert not flags[0] & FLAG_BEYOND_RECIPE and flags[1] & FLAG_BEYOND_RECIPE
    settings = dict(catalog="c", priors="p", floor=0.03, ar_column="Ar", ar_scale=1.0, ar_max=8.0)
    settings |= dict(no_dust_map=False, dust_curves="", cone=None, no_input_columns=False)
    for selection in run.SELECTIONS:
        recorded = run.runConfiguration(SimpleNamespace(**settings, selection=selection))
        assert recorded["selection"] == selection


def test_maps_for_a_population_the_locus_lacks_are_refused(tmp_path):
    """A population with no segment of its own on the locus would have nowhere to go."""
    run = script("run_dp2")
    path = tmp_path / "priors.npz"
    np.savez(path, popSegment=np.array([0, 1, 2]), model=np.array('{"locus": "LSSTlocus_10Gyr_fix.txt"}'))
    with pytest.raises(SystemExit, match=r"\[1, 2\]"):
        run.checkPriorLocus(path, lt.RUN_LOCUS_FILE)


def test_a_colour_with_no_value_is_not_counted_as_measured_on_an_unfitted_row():
    """nColors and bit 16 of an unfitted row read a colour as the fit does: error at most 1, and a value."""
    from photod.bayes import FLAG_COLOR_MISSING, FLAG_NO_PRIOR, unfittedEstimates

    gp = params(lt.RUN_LOCUS_FILE, kFeH=15)
    stars = pd.DataFrame(
        {"objectId": [1, 2], "ra": 10.0, "dec": 0.0, "rmag": 20.0}
        | {c: 0.5 for c in COLORS}
        | {c + "Err": 0.05 for c in COLORS}
    )
    stars.loc[1, "zy"] = np.nan
    rows = unfittedEstimates(stars, gp, FLAG_NO_PRIOR)
    assert rows["nColors"].tolist() == [5, 4]
    flags = rows["fit_flags"].to_numpy()
    assert not flags[0] & FLAG_COLOR_MISSING and flags[1] & FLAG_COLOR_MISSING
