import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import photod.locus as lt
from photod.bayes import getColorsAndPriorIndices
from photod.parameters import GlobalParams

COLNAMES = ["tLoc", "Mr", "FeH", "ug", "gr", "ri", "iz", "zy"]


def load(name):
    """A locus table from the data directory."""
    return lt.LSSTsimsLocus(
        fixForStripe82=False, datafile=lt.DEFAULT_LOCUS_FILE.parent / name, colnames=COLNAMES
    )


def loadPair():
    """The original locus, the DP2 one, and the rows of the DP2 table that lie on the original grid.

    The Mr correction moves the reddest dwarfs half a magnitude fainter than the last row of the original
    grid, so the DP2 table continues the grid there; everything the two have in common is the rest of it, in
    the same order, because both tables run over [Fe/H] and then over tLoc.
    """
    original, dp2 = load("LSSTlocus_10Gyr_fix.txt"), load("LSSTlocus_10Gyr_DP2.txt")
    onGrid = np.array(dp2["tLoc"]) <= np.array(original["tLoc"]).max() + 1e-9
    assert int(onGrid.sum()) == len(original)
    return original, dp2, onGrid


def test_dp2_locus_keeps_the_grid_of_the_original_and_extends_the_faint_end():
    """The shared part of the grid, and Mr on it, are the original's; the new rows only go fainter."""
    original, dp2, onGrid = loadPair()
    tOriginal, tDp2 = np.unique(original["tLoc"]), np.unique(dp2["tLoc"])
    assert_allclose(tDp2[: tOriginal.size], tOriginal)
    assert tDp2.size > tOriginal.size and tDp2[tOriginal.size] > tOriginal[-1]
    assert_allclose(np.unique(dp2["FeH"]), np.unique(original["FeH"]))
    assert len(dp2) == np.unique(dp2["FeH"]).size * tDp2.size
    for c in ("tLoc", "Mr", "FeH"):
        assert_allclose(np.array(dp2[c])[onGrid], np.array(original[c]))
    for c in ("ug", "gr", "ri", "iz", "zy"):
        assert np.all(np.isfinite(np.array(dp2[c])))
        assert np.abs(np.array(dp2[c])[onGrid] - np.array(original[c])).max() < 0.6
    giants = np.array(original["Mr"]) != np.array(original["tLoc"])
    for c in ("gr", "ri", "iz", "zy"):
        assert_allclose(np.array(dp2[c])[onGrid][giants], np.array(original[c])[giants])
    assert np.all(np.array(dp2["Mr"])[~onGrid] == np.array(dp2["tLoc"])[~onGrid])


def test_dp2_locus_moves_the_main_sequence_the_measured_way():
    """G and K dwarfs (0.6 < g-i < 1.0) are redder at fixed tLoc, the red end (g-i > 1.8) bluer."""
    original, dp2, onGrid = loadPair()
    ms = np.abs(np.array(original["Mr"]) - np.array(original["tLoc"])) < 1e-3
    gi0 = np.array(original["gr"]) + np.array(original["ri"])
    delta = (np.array(dp2["gr"]) + np.array(dp2["ri"]))[onGrid] - gi0
    assert np.median(delta[ms & (gi0 > 0.6) & (gi0 < 1.0)]) > 0.02
    assert np.median(delta[ms & (gi0 > 1.8) & (gi0 < 2.6)]) < -0.02


def test_dp2_locus_keeps_the_reddest_main_sequence_colour():
    """The red end is moved, not cut: the reddest dwarf of the original is still on the grid, fainter.

    Half a magnitude of Mr offset at the red end used to fall off the faint end of the grid, which left the
    reddest colour of the locus 0.3 mag bluer than the model it came from and gave red M dwarfs nothing to
    fit but the last row of the grid.
    """
    original, dp2, _ = loadPair()
    for locus in (original, dp2):
        ms = np.abs(np.array(locus["Mr"]) - np.array(locus["tLoc"])) < 1e-3
        locus["reddest"] = np.where(ms, np.array(locus["gi"]), -np.inf)
    assert_allclose(np.max(dp2["reddest"]), np.max(original["reddest"]), atol=0.01)
    faintest = np.array(dp2["tLoc"])[np.argmax(dp2["reddest"])]
    assert faintest > np.array(original["tLoc"]).max() + 0.4


def test_color_error_floor_is_added_in_quadrature():
    """GlobalParams(colorErrFloor=f) gives sqrt(err^2 + f^2) to the fit; the default changes nothing."""
    locus = load("LSSTlocus_10Gyr_DP2.txt")
    locusData = lt.subsampleLocusData(locus, 20, 3, yLabel="tLoc")
    colors = ("ug", "gr", "ri")
    ArGridList, locus3DList = lt.get3DmodelList(locusData, colors, yLabel="tLoc")
    catalog = pd.DataFrame(
        {
            "rmag": [18.0, 21.0],
            "ug": [1.2, 1.5],
            "gr": [0.5, 0.9],
            "ri": [0.2, 0.4],
            "ugErr": [0.01, 0.10],
            "grErr": [0.005, 0.02],
            "riErr": [0.005, 0.03],
        }
    )
    for floor in (0.0, 0.03):
        params = GlobalParams(
            colors,
            locusData,
            ArGridList,
            locus3DList,
            yLabel="tLoc",
            MrColumn="tLoc",
            computeMrTrue=True,
            colorErrFloor=floor,
        )
        colorsErr = getColorsAndPriorIndices(catalog, params)[1]
        expected = np.sqrt(catalog[[c + "Err" for c in colors]].to_numpy() ** 2 + floor**2)
        assert_allclose(colorsErr, expected)
