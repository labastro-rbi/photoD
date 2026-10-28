"""The stellar estimates of one class on every row (GlobalParams.estimateClass), the class column, and the run
that keeps every row of its input (run_dp2 --selection flag)."""

import argparse
import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from test_wd_classes import flatPrior, params, starsAt
from wd_loci import COLORS, script

import photod.locus as lt
from photod.bayes import (
    CLASS_COLUMN,
    FLAG_BEYOND_RECIPE,
    FLAG_EXTRAGALACTIC,
    FLAG_FAILS_RECIPE,
    FLAG_NO_PRIOR,
    WD_DISTANCE,
    WD_DISTANCE_MIN_P,
    WD_MASS,
    makeBayesEstimates3d,
    unfittedEstimates,
)
from photod.quality import QC_CLASSES
from photod.selection import FAILS_RECIPE, bandSnr, recipeSelection

STELLAR = ("tLoc_", "Mr_", "FeH_", "Ar_", "DM_quantile", "DM_std")


def stellarColumns(frame):
    """The columns of the stellar estimates, of whichever class they are."""
    return [c for c in frame.columns if c.startswith(STELLAR)] + ["multimodality"]


@pytest.fixture(scope="module")
def mixed(qsoLoci):
    """The mass and QSO locus with estimates of the stars, and stars, white dwarfs and QSOs placed on it."""
    gp = params(qsoLoci[1])
    gp = dataclasses.replace(gp, estimateClass="MS", wdMasses=np.linspace(0.4, 1.3, gp.FeH1d.size))
    feH = np.broadcast_to(gp.FeH1d[:, None], (gp.FeH1d.size, gp.Mr1d.size)).ravel()
    dwarfs = np.where(
        (gp.segmentFlat == lt.SEGMENT_STARS) & gp.locusValid & (gp.MrTrueFlat > 5) & (feH == feH.max())
    )[0]
    hydrogen = np.where(gp.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    qso = np.where((gp.segmentFlat == lt.SEGMENT_QSO) & np.isclose(feH, -0.5))[0]
    rows = np.concatenate([dwarfs[[5, 40, 120]], hydrogen[[30, 150, 400]], qso[[40, 80, 120]]])
    stars = starsAt(gp, rows, err=0.01)
    return gp, stars, rows


def test_the_stellar_estimates_are_those_of_a_fit_of_the_stars_alone(mixed):
    """Even where p_WD or p_QSO is 1, every stellar column is what a fit with a prior on stars alone gives."""
    gp, stars, rows = mixed
    prior = flatPrior(gp)
    perClass, _ = makeBayesEstimates3d(stars, prior, gp, batchSize=16)
    onlyStars = dataclasses.replace(gp, estimateClass=None)
    reference, _ = makeBayesEstimates3d(
        stars, prior * (gp.segmentFlat == lt.SEGMENT_STARS), onlyStars, batchSize=16
    )
    assert np.all(perClass["p_MS"].to_numpy()[:3] > 0.99)
    assert np.all(perClass["p_WD"].to_numpy()[3:6] > 0.999)
    assert np.all(perClass["p_QSO"].to_numpy()[6:] > 0.99)
    assert_allclose(reference["p_MS"], 1.0, atol=1e-6)
    for name in stellarColumns(perClass):
        assert np.isfinite(perClass[name]).all(), name
        assert_allclose(perClass[name], reference[name], rtol=1e-5, atol=1e-5, err_msg=name)
    keep = ~FLAG_EXTRAGALACTIC
    assert np.array_equal(perClass["fit_flags"] & keep, reference["fit_flags"] & keep)
    assert np.array_equal(perClass["fit_flags"] & FLAG_EXTRAGALACTIC > 0, perClass["p_QSO"].to_numpy() > 0.5)


def test_the_class_probabilities_are_those_of_the_fit_of_the_most_probable_class(mixed):
    """Scaling each class to its own peak changes the probabilities by rounding only, where none underflow."""
    gp, stars, rows = mixed
    prior = flatPrior(gp)
    perClass, _ = makeBayesEstimates3d(stars, prior, gp, batchSize=16)
    likely, _ = makeBayesEstimates3d(stars, prior, dataclasses.replace(gp, estimateClass=None), batchSize=16)
    for name in ("p_MS", "p_WD", "p_He", "p_QSO"):
        assert_allclose(perClass[name], likely[name], rtol=1e-4, atol=1e-6, err_msg=name)
    assert_allclose(perClass["p_MS"] + perClass["p_WD"] + perClass["p_QSO"], 1.0, rtol=1e-5)
    sure = np.abs(likely["lnodds_WD"].to_numpy()) < 15
    assert_allclose(perClass["lnodds_WD"][sure], likely["lnodds_WD"][sure], atol=1e-3)
    # the white dwarf distance and the redshift, which are of their own classes, do not move either
    for name in (f"{WD_DISTANCE}_quantile_median", WD_MASS):
        assert_allclose(perClass[name], likely[name], rtol=1e-5, atol=1e-5, equal_nan=True, err_msg=name)
    # the estimates of the most probable class are of the white dwarfs, or empty for a QSO
    assert np.isnan(likely["DM_quantile_median"].to_numpy()[6:]).all()


def test_the_white_dwarf_columns_are_written_where_the_class_is_likely_and_photoz_always(mixed):
    """DM_WD and WD_mass where p_WD > WD_DISTANCE_MIN_P; photoz, the probabilities and class everywhere."""
    gp, stars, rows = mixed
    out, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=16)
    likely = out["p_WD"].to_numpy() > WD_DISTANCE_MIN_P
    assert likely.any() and (~likely).any()
    for name in [f"{WD_DISTANCE}_quantile_{q}" for q in ("lo", "median", "hi")] + [WD_MASS]:
        assert np.array_equal(np.isfinite(out[name].to_numpy()), likely), name
    for name in ("p_MS", "p_WD", "p_He", "p_QSO", "lnodds_WD", "photoz"):
        assert np.isfinite(out[name]).all(), name
    assert_allclose(out["photoz"].to_numpy()[6:], lt.qsoRedshift(gp.Mr1d[rows[6:] % gp.Mr1d.size]), atol=0.1)


def test_the_class_column_is_the_most_probable_class(mixed):
    """class is the index in QC_CLASSES of the largest of p_MS, p_WD and p_QSO, -1 on a row never fitted."""
    gp, stars, rows = mixed
    out, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=16)
    p = out[[f"p_{name}" for name in QC_CLASSES]].to_numpy()
    assert out[CLASS_COLUMN].dtype == np.int8
    assert np.array_equal(out[CLASS_COLUMN].to_numpy(), np.argmax(p, axis=1))
    assert out[CLASS_COLUMN].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    empty = unfittedEstimates(stars, gp, FLAG_NO_PRIOR)
    assert (empty[CLASS_COLUMN] == -1).all() and empty[CLASS_COLUMN].dtype == np.int8


def preparedFrame(run):
    """Four rows of a prepared catalogue: two that meet the recipe (one past r 23.5) and two that do not."""
    frame = pd.DataFrame({c: np.zeros(4) for c in run.FIT_COLUMNS})
    frame["objectId"] = np.arange(4)
    frame["rmag"] = [20.0, 24.0, 22.0, 22.0]
    frame["rmagErr"] = [0.01, 0.05, 0.2, 0.01]  # S/N 109, 22, 5.4, 109
    frame["grErr"] = [0.02, 0.1, 0.25, 0.5]  # g S/N 56, 12, -, 2.2
    frame["riErr"] = 0.03
    return frame


def test_the_flag_selection_keeps_every_row_and_marks_the_ones_the_recipe_drops():
    """Every row is fitted, and FLAG_FAILS_RECIPE is set exactly where the recipe would leave the row out."""
    run = script("run_dp2")
    frame = preparedFrame(run)
    passes = recipeSelection(frame["rmag"].to_numpy(), bandSnr(frame))[0]
    assert passes.tolist() == [True, True, False, False]
    flagged = run.prepareStars(frame, {}, None, selection="flag")
    assert flagged["objectId"].tolist() == [0, 1, 2, 3]
    assert np.array_equal(flagged[FAILS_RECIPE].to_numpy(), ~passes)
    recipe = run.prepareStars(frame, {}, None, selection="recipe")
    none = run.prepareStars(frame, {}, None, selection="none")
    assert FAILS_RECIPE not in recipe.columns and FAILS_RECIPE not in none.columns
    gp = params(lt.RUN_LOCUS_FILE, kFeH=15)
    stars = flagged.assign(**{c: 0.5 for c in COLORS}, **{c + "Err": 0.05 for c in COLORS})
    estimates, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=4)
    flags = estimates["fit_flags"].to_numpy()
    assert np.array_equal(flags & FLAG_FAILS_RECIPE > 0, ~passes)
    assert np.array_equal(flags & FLAG_BEYOND_RECIPE > 0, [False, True, False, False])
    # a row the fit is never run on carries the mark too
    empty = unfittedEstimates(stars, gp, FLAG_NO_PRIOR)
    assert np.array_equal(empty["fit_flags"].to_numpy() & FLAG_FAILS_RECIPE > 0, ~passes)
    # a fit of the recipe's rows alone gives those rows the same answers
    alone, _ = makeBayesEstimates3d(stars[passes].drop(columns=FAILS_RECIPE), flatPrior(gp), gp, batchSize=4)
    assert_allclose(alone["DM_quantile_median"], estimates["DM_quantile_median"][passes], rtol=1e-6)


def test_the_flag_selection_from_fluxes_keeps_the_point_sources_that_fail_the_recipe():
    """From the survey table the point-source cut still applies; the recipe only marks."""
    run = script("run_dp2")
    n = 5
    df = pd.DataFrame(
        {
            "objectId": np.arange(n),
            "coord_ra": np.zeros(n),
            "coord_dec": np.zeros(n),
            "refExtendedness": [0.0, 0.0, 0.0, 1.0, 0.0],
            "ebv": np.full(n, 0.02),
        }
    )
    for b in "ugrizy":
        df[f"{b}_psfFlux"] = 10 ** (-0.4 * (20.0 - 31.4))
        df[f"{b}_psfFluxErr"] = df[f"{b}_psfFlux"] / 50.0
    df.loc[1, "r_psfFluxErr"] = df.loc[1, "r_psfFlux"] / 5.0  # r S/N 5: fails
    df.loc[2, "g_psfFluxErr"] = df.loc[2, "g_psfFlux"] / 2.0  # g S/N 2: fails
    df.loc[4, "r_psfFlux"] = 10 ** (-0.4 * (16.0 - 31.4))  # brighter than 16.5: not a point source fitted
    flagged = run.starsFromFluxes(df, "flag")
    assert flagged["objectId"].tolist() == [0, 1, 2]
    assert flagged[FAILS_RECIPE].tolist() == [False, True, True]
    assert run.starsFromFluxes(df, "recipe")["objectId"].tolist() == [0]


def test_a_run_of_the_stars_estimates_records_them_and_the_selection():
    """A run records the stars' estimates when asked for them, and its selection and fit model always."""
    run = script("run_dp2")
    settings = dict(
        catalog="/data/dp2",
        priors="/data/priors.npz",
        floor=0.03,
        ar_column="Ar",
        ar_scale=1.0,
        ar_max=8.0,
        no_dust_map=False,
        dust_curves="",
        cone=None,
        no_input_columns=False,
    )
    stars = run.runConfiguration(SimpleNamespace(**settings, selection="flag", estimate_class="MS"))
    assert stars["selection"] == "flag" and stars["estimateClass"] == "MS"
    assert stars["model"] == run.FIT_MODEL and "qcTable" not in stars
    likely = run.runConfiguration(SimpleNamespace(**settings, selection="recipe", estimate_class="likely"))
    assert likely["selection"] == "recipe" and likely["model"] == run.FIT_MODEL
    assert not {"estimateClass", "qcTable"} & likely.keys()
    assert run.runConfiguration(SimpleNamespace(**settings))["selection"] == run.SELECTION
    assert run.ESTIMATE_CLASSES["MS"] == "MS" and run.ESTIMATE_CLASSES["likely"] is None


def test_every_selection_default_is_the_command_line_one(monkeypatch):
    """A function handed no selection selects as run_dp2.py does when --selection is not given."""
    run = script("run_dp2")

    class Parsed(Exception):
        pass

    def stop(parser, *args, **kwargs):
        raise Parsed(parser.get_default("selection"))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    with pytest.raises(Parsed) as parsed:
        run.main()
    assert parsed.value.args[0] == run.SELECTION == "recipe"
    for function in (run.prepareStars, run.starsFromFluxes, run.fitAndWrite, run.startWorker):
        default = inspect.signature(function).parameters["selection"].default
        assert default == run.SELECTION, function.__name__


def test_an_unknown_estimate_class_is_refused(qsoLoci):
    """Only the stars, or the most probable class, have estimates of their own to give."""
    with pytest.raises(ValueError, match="estimateClass"):
        dataclasses.replace(params(qsoLoci[1]), estimateClass="QSO")


def test_the_class_probabilities_sum_to_one_on_a_star_no_model_comes_near(mixed):
    """Far from every class, at a chi2 of tens of thousands, two classes about as likely still sum to 1.

    The stars lie on the line between two white dwarfs of the locus, moved off it by 3 mag in every colour,
    where the most probable class goes over from QSO to star; the line is narrowed down to that place.
    """
    gp, stars, rows = mixed
    a, b = gp.locusColors2d[rows[4]], gp.locusColors2d[rows[7]]
    offset = np.where(np.arange(len(COLORS)) % 2 == 0, 3.0, -3.0)

    def fit(t):
        colours = (1 - t[:, None]) * a + t[:, None] * b + offset
        n = len(t)
        far = pd.DataFrame({"objectId": np.arange(n), "ra": np.zeros(n), "dec": np.zeros(n), "rmag": 20.0})
        for k, c in enumerate(COLORS):
            far[c], far[c + "Err"] = colours[:, k], 0.01
        return makeBayesEstimates3d(far, flatPrior(gp), gp, batchSize=16)[0]

    lo, hi = 0.0, 1.0
    for _ in range(4):
        t = np.linspace(lo, hi, 16)
        out = fit(t)
        changed = (out[CLASS_COLUMN] != out[CLASS_COLUMN].iloc[0]).to_numpy()
        assert changed[-1], "the most probable class is the same along the whole line"
        k = np.argmax(changed)
        lo, hi = t[k - 1], t[k]
    p = out[[f"p_{name}" for name in QC_CLASSES]].to_numpy()
    assert (out["chi2min"] > 1e4).all()
    assert (np.sort(p, axis=1)[:, -2] > 0.05).any(), "no star with two classes about as likely"
    assert_allclose(p.sum(axis=1), 1.0, atol=2e-6)
    assert (out["p_He"] <= out["p_WD"] * (1 + 1e-5) + 1e-12).all()


def test_a_star_with_no_colour_measured_has_the_estimates_of_its_prior(mixed):
    """No colour and no dust curve leave the likelihood flat in A_r too: the answer is the prior's, not empty.

    Only a run that keeps every row (--selection flag) fits such stars; the recipe needs two colours.
    """
    gp, stars, rows = mixed
    blind = stars.iloc[:3].copy()
    for c in COLORS:
        blind[c], blind[c + "Err"] = 0.0, 9.99
    out, _ = makeBayesEstimates3d(blind, flatPrior(gp), gp, batchSize=16)
    assert (out["nColors"] == 0).all() and (out["chi2min"] == 0).all()
    for name in stellarColumns(out) + ["p_MS", "p_WD", "p_QSO", "photoz"]:
        assert np.isfinite(out[name]).all(), name
    assert_allclose(out["p_WD"], 1 / (1 + np.exp(-out["lnprior_odds_WD"].to_numpy(float))), rtol=1e-4)
    assert (out[CLASS_COLUMN] >= 0).all()
