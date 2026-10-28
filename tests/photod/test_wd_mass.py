"""The white dwarf mass axis in the fit: the prior on mass of a pixel, the WD_mass column, and a run that
refuses a locus and maps that do not go together."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_wd_classes import flatPrior, params, starsAt
from wd_loci import COLORS, read, script

import photod.locus as lt
from photod.bayes import WD_DISTANCE_MIN_P, makeBayesEstimates3d
from photod.parameters import GlobalParams
from photod.priors import getBayesConstants


def test_the_prior_of_a_pixel_follows_the_mass_prior_and_keeps_the_shares(massLocus):
    """A flat map with shares: each population at its share, the white dwarfs spread over mass as weighed."""
    path, grid = massLocus
    run = script("run_dp2")
    gp = params(path, kFeH=1)
    bc = getBayesConstants(lt.HELIUM_TLOC[0])
    xGrid = np.linspace(bc["FeHmin"], bc["FeHmax"], bc["FeHNpts"])
    yGrid = np.linspace(bc["MrFaint"], bc["MrBright"], bc["MrNpts"])
    weight = script("make_priors").massWeights(grid, path)
    nR = bc["rmagNsteps"]
    shares = np.tile([0.9, 0.08, 0.02], (nR, 1))
    shares[0] = [1.0, 0.0, 0.0]
    priors = dict(
        kde=np.ones((1, nR, yGrid.size, xGrid.size), np.float32),
        rmag=np.linspace(bc["rmagMin"], bc["rmagMax"], nR),
        xGrid=xGrid,
        yGrid=yGrid,
        popShare=shares[None],
        popSegment=np.array([0, 1, 2]),
        wdMassWeight=weight.astype(np.float32),
    )
    out = run.locusPrior(priors, 0, gp)
    cells = gp.priorWeights * gp.locusValid
    for k in (0, 5):
        for code, share in zip((0, 1, 2), shares[k], strict=True):
            on = gp.segmentFlat == code
            assert_allclose(np.sum((out[k] * cells)[on]), max(share, run.SHARE_FLOOR), rtol=1e-9)
    density = out[5].reshape(weight.shape)
    hydrogen = gp.segment1d == lt.SEGMENT_HYDROGEN_WD
    assert_allclose(density[:, hydrogen] / density[:, hydrogen].mean(axis=0), weight[:, hydrogen], rtol=1e-5)
    # a population whose map is empty on its segment is spread over it by the same weights
    empty = dict(priors, kde=np.where(yGrid[None, None, :, None] < -3, 0, priors["kde"]).astype(np.float32))
    helium = gp.segment1d == lt.SEGMENT_HELIUM_WD
    spread = run.locusPrior(empty, 0, gp)[5].reshape(weight.shape)[:, helium]
    assert_allclose(spread / spread.mean(axis=0), weight[:, helium], rtol=1e-5)


def test_wd_mass_is_the_mass_of_the_white_dwarf_posterior(massLocus):
    """With the prior on one mass row the column is that mass; it is empty off the class and off the axis."""
    path, grid = massLocus
    gp = params(path, kFeH=5)
    masses = np.asarray(grid["masses"])[::5]
    withMass = GlobalParams(
        COLORS,
        gp.locusData,
        gp.ArGridList,
        yLabel="tLoc",
        MrColumn="tLoc",
        computeMrTrue=True,
        ArGridRange="Small",
        colorErrFloor=0.03,
        wdMasses=masses,
    )
    track = np.where(withMass.segmentFlat == lt.SEGMENT_HYDROGEN_WD)[0]
    feH = np.broadcast_to(withMass.FeH1d[:, None], (withMass.FeH1d.size, withMass.Mr1d.size)).ravel()
    row = 3
    dwarfs = (withMass.segmentFlat == 0) & withMass.locusValid & (withMass.MrTrueFlat > 6)
    dwarfs &= (withMass.MrTrueFlat < 8) & (feH == feH.max())
    rows = np.concatenate([track[(feH[track] == withMass.FeH1d[row])][[60, 150]], np.where(dwarfs)[0][[5]]])
    stars = starsAt(withMass, rows)
    prior = flatPrior(withMass) * np.where(
        (withMass.segmentFlat == 0) | (feH == withMass.FeH1d[row]), 1.0, 0.0
    )
    estimates, _ = makeBayesEstimates3d(stars, prior, withMass, batchSize=4)
    p, mass = estimates["p_WD"].to_numpy(), estimates["WD_mass"].to_numpy()
    assert np.all(p[:2] > 0.99)
    assert_allclose(mass[:2], masses[row], atol=1e-4)
    assert np.all(np.isnan(mass[p <= WD_DISTANCE_MIN_P]))
    # with a flat prior over mass the answer is still a mass on the grid
    flat, _ = makeBayesEstimates3d(stars, flatPrior(withMass), withMass, batchSize=4)
    assert np.all((flat["WD_mass"].to_numpy()[:2] >= 0.4) & (flat["WD_mass"].to_numpy()[:2] <= 1.3))
    # a locus without a mass axis has no mass to give
    plain, _ = makeBayesEstimates3d(stars, prior, gp, batchSize=4)
    assert plain["WD_mass"].isna().all()
    with pytest.raises(ValueError, match="one mass per"):
        GlobalParams(
            COLORS, gp.locusData, gp.ArGridList, yLabel="tLoc", MrColumn="tLoc",
            computeMrTrue=True, ArGridRange="Small", wdMasses=masses[:-1],
        )  # fmt: skip


def test_a_mass_locus_and_maps_without_a_prior_on_mass_are_refused(massLocus, singleLocus, tmp_path):
    """A run needs the maps that say how much of each mass there is, and those maps need a mass axis."""
    run = script("run_dp2")
    path, grid = massLocus
    nFeH, nT = 31, np.unique(np.asarray(read(path)["tLoc"])).size
    plain = tmp_path / "plain.npz"
    np.savez(plain, model=np.array(json.dumps({"locus": path.name})))
    with pytest.raises(SystemExit, match="are masses"):
        run.checkPriorLocus(plain, path)
    good = tmp_path / "mass.npz"
    wdMass = {"grid": "WD_massgrid.json", "masses": grid["masses"], "md5": "x"}
    np.savez(
        good,
        model=np.array(json.dumps({"locus": path.name, "wdMass": wdMass})),
        wdMassWeight=np.ones((nFeH, nT)),
    )
    assert run.checkPriorLocus(good, path) == tuple(grid["masses"])
    # maps with a prior on mass for a locus without the axis
    other = tmp_path / "other.npz"
    np.savez(other, model=np.array(json.dumps({"wdMass": wdMass})), wdMassWeight=np.ones((nFeH, nT)))
    with pytest.raises(SystemExit, match="are not masses"):
        run.checkPriorLocus(other, singleLocus)
    wrong = tmp_path / "wrong.npz"
    np.savez(
        wrong,
        model=np.array(json.dumps({"locus": path.name, "wdMass": wdMass})),
        wdMassWeight=np.ones((nFeH, 5)),
    )
    with pytest.raises(SystemExit, match="grid"):
        run.checkPriorLocus(wrong, path)
    # and maps without a prior on mass, for a locus without the axis, go through with no masses
    bare = tmp_path / "bare.npz"
    np.savez(bare, model=np.array(json.dumps({"locus": singleLocus.name})))
    assert run.checkPriorLocus(bare, singleLocus) is None
