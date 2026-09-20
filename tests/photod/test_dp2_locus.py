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


def test_dp2_locus_keeps_the_grid_of_the_original():
    """Same (FeH, tLoc) grid and Mr as the original; only the colours differ, by at most the corrections."""
    original, dp2 = load("LSSTlocus_10Gyr_fix.txt"), load("LSSTlocus_10Gyr_DP2.txt")
    assert len(dp2) == len(original)
    for c in ("tLoc", "Mr", "FeH"):
        assert_allclose(np.array(dp2[c]), np.array(original[c]))
    for c in ("ug", "gr", "ri", "iz", "zy"):
        diff = np.abs(np.array(dp2[c]) - np.array(original[c]))
        assert np.all(np.isfinite(np.array(dp2[c])))
        assert diff.max() < 0.6
    giants = np.array(original["Mr"]) != np.array(original["tLoc"])
    for c in ("gr", "ri", "iz", "zy"):
        assert_allclose(np.array(dp2[c])[giants], np.array(original[c])[giants])


def test_dp2_locus_moves_the_main_sequence_the_measured_way():
    """G and K dwarfs (0.6 < g-i < 1.0) are redder at fixed tLoc, the red end (g-i > 1.8) bluer."""
    original, dp2 = load("LSSTlocus_10Gyr_fix.txt"), load("LSSTlocus_10Gyr_DP2.txt")
    ms = np.abs(np.array(original["Mr"]) - np.array(original["tLoc"])) < 1e-3
    gi0 = np.array(original["gr"]) + np.array(original["ri"])
    delta = np.array(dp2["gr"]) + np.array(dp2["ri"]) - gi0
    assert np.median(delta[ms & (gi0 > 0.6) & (gi0 < 1.0)]) > 0.02
    assert np.median(delta[ms & (gi0 > 1.8) & (gi0 < 2.6)]) < -0.02


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
            colors, locusData, ArGridList, locus3DList, yLabel="tLoc", MrColumn="tLoc", colorErrFloor=floor
        )
        _, colorsErr, _, _ = getColorsAndPriorIndices(catalog, params)
        expected = np.sqrt(catalog[[c + "Err" for c in colors]].to_numpy() ** 2 + floor**2)
        assert_allclose(colorsErr, expected)
