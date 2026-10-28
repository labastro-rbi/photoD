"""The QSO class: its place outside everything built on Mr, the class columns and photoz, and a run that
refuses maps without a QSO population."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_wd_classes import flatPrior, params, starsAt
from wd_loci import COLORS, read, script

import photod.locus as lt
from photod.bayes import FLAG_EXTRAGALACTIC, WD_DISTANCE_MIN_P, makeBayesEstimates3d
from photod.parameters import GlobalParams


def test_the_qso_points_stay_out_of_everything_built_on_mr(qsoLoci):
    """The Qr and true Mr grids are those of the locus without the track; the track keeps its redshifts."""
    without, with_ = params(qsoLoci[0]), params(qsoLoci[1])
    assert np.array_equal(without.QrGrid, with_.QrGrid)
    assert np.array_equal(without.MrTrueGrid, with_.MrTrueGrid)
    assert with_.QrGrid.min() > lt.QSO_MR + 10
    assert_allclose(with_.qsoRedshift1d[[0, -1]], lt.QSO_REDSHIFT)
    assert with_.qsoAxisIndex.size == 149 and without.qsoAxisIndex.size == 0
    assert with_.classAxisMasks["QSO"].sum() == 149
    # the track's cells are its own, one step of 0.01 each
    widths = with_.priorWeights.reshape(with_.FeH1d.size, -1)[0] / with_.dFeH
    assert_allclose(widths[with_.qsoAxisIndex].sum(), 149 * 0.01, rtol=1e-9)


def test_star_estimates_do_not_move_when_the_qso_class_has_no_posterior_mass(qsoLoci):
    """With a negligible QSO prior the stars and white dwarfs come out as they did without the track."""
    without, with_ = params(qsoLoci[0]), params(qsoLoci[1])
    feH = np.broadcast_to(without.FeH1d[:, None], (without.FeH1d.size, without.Mr1d.size)).ravel()
    dwarfs = np.where(
        (without.segmentFlat == 0)
        & without.locusValid
        & (without.MrTrueFlat > 5)
        & (without.MrTrueFlat < 9)
        & (feH == feH.max())
    )[0]
    track = np.where(without.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    rows = np.concatenate([dwarfs[[5, 40, -5]], track[[30, 150]]])
    stars = starsAt(without, rows)
    priorWith = flatPrior(with_) * np.where(with_.segmentFlat == lt.SEGMENT_QSO, 1e-30, 1.0)
    a, _ = makeBayesEstimates3d(stars, flatPrior(without), without, batchSize=8)
    b, _ = makeBayesEstimates3d(stars, priorWith, with_, batchSize=8)
    assert np.all(b["p_QSO"] < 1e-20)
    names = [
        c for c in a.columns if c.startswith(("DM", "Mr_", "tLoc", "FeH", "Ar_")) or c in ("p_WD", "p_He")
    ]
    for name in names + ["multimodality"]:
        assert_allclose(b[name], a[name], rtol=1e-5, atol=1e-6, err_msg=name)
    assert np.array_equal(a["fit_flags"], b["fit_flags"])
    assert np.isnan(b["photoz"]).all()


def test_a_qso_is_called_one_with_its_redshift_and_no_distance(qsoLoci):
    """A QSO's colours give p_QSO, photoz and an empty distance modulus, and the flag says why."""
    gp = params(qsoLoci[1])
    onTrack = np.where((gp.segmentFlat == lt.SEGMENT_QSO) & np.isclose(gp.FeH1d.repeat(gp.Mr1d.size), -0.5))[
        0
    ]
    z = lt.qsoRedshift(gp.Mr1d[onTrack % gp.Mr1d.size])
    rows = onTrack[[np.argmin(np.abs(z - target)) for target in (1.0, 2.0, 4.0)]]
    feH = np.broadcast_to(gp.FeH1d[:, None], (gp.FeH1d.size, gp.Mr1d.size)).ravel()
    dwarf = np.where((gp.segmentFlat == 0) & gp.locusValid & (gp.MrTrueFlat > 6) & (feH == feH.max()))[0][10]
    stars = starsAt(gp, np.append(rows, dwarf), err=0.01)
    estimates, _ = makeBayesEstimates3d(stars, flatPrior(gp), gp, batchSize=8)
    p = estimates["p_QSO"].to_numpy()
    assert np.all(p[:3] > 0.5) and p[3] < 0.01
    assert_allclose(estimates["p_MS"] + estimates["p_WD"] + estimates["p_QSO"], 1.0, rtol=1e-5)
    assert_allclose(
        estimates["photoz"].to_numpy()[:3], z[[np.argmin(np.abs(z - t)) for t in (1, 2, 4)]], atol=0.1
    )
    assert np.isnan(estimates["photoz"].iloc[3])
    flags = estimates["fit_flags"].to_numpy()
    assert np.all(flags[:3] & FLAG_EXTRAGALACTIC) and not flags[3] & FLAG_EXTRAGALACTIC
    for name in ("DM_quantile_lo", "DM_quantile_median", "DM_quantile_hi", "DM_std"):
        assert np.isnan(estimates[name].to_numpy()[:3]).all() and np.isfinite(estimates[name].iloc[3])
    # the stellar estimates are of the most probable class of star, never of the track, and empty where no
    # star comes within float range of the QSO's fit
    tLoc = estimates["tLoc_quantile_median"].to_numpy()
    assert np.isfinite(tLoc[3]) and not np.any((tLoc >= lt.QSO_TLOC[0]) & (tLoc <= lt.QSO_TLOC[1]))
    assert np.all(np.isfinite(tLoc) | (p > 1 - 1e-6))
    dmWD = estimates["DM_WD_quantile_median"].to_numpy()
    assert np.all(np.isnan(dmWD[estimates["p_WD"].to_numpy() <= WD_DISTANCE_MIN_P]))


def test_a_qso_track_laid_out_otherwise_is_refused(qsoLoci):
    """The redshifts are read off QSO_TLOC, so a track on other tLoc would be read as other redshifts."""
    table = read(qsoLoci[1])
    table = lt.subsampleLocusData(table, kMr=1, kFeH=10, yLabel="tLoc")
    keep = ~((np.asarray(table["segment"]) == lt.SEGMENT_QSO) & (np.asarray(table["tLoc"]) > 1.45))
    table = table[keep]
    with pytest.raises(ValueError, match="QSO track"):
        GlobalParams(COLORS, table, lt.getArGridList(), yLabel="tLoc", MrColumn="tLoc", computeMrTrue=True)


def test_a_qso_locus_without_a_qso_population_is_refused(qsoLoci, tmp_path):
    """Without a population of its own the track would have only the stars' smoothing tails for a prior."""
    run = script("run_dp2")
    path = tmp_path / "priors.npz"
    grid = qsoLoci[2]
    model = {"locus": qsoLoci[1].name, "wdMass": {"grid": "g", "masses": grid["masses"], "md5": ""}}
    nTLoc = np.unique(np.asarray(read(qsoLoci[1])["tLoc"])).size
    np.savez(
        path,
        popSegment=np.array([0, 1, 2]),
        wdMassWeight=np.ones((31, nTLoc)),
        model=np.array(json.dumps(model)),
    )
    with pytest.raises(SystemExit, match="QSO population"):
        run.checkPriorLocus(path, qsoLoci[1])
