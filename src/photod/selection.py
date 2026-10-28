"""Which objects a run fits: the point-source recipe of scripts/run_dp2.py, as one function for every reader.

The recipe asks for S/N above 10 in r and above 3 in g and in i, and puts the limit of the catalogue at
r = 23.5. An object past that limit that still meets the S/N cuts is fitted and flagged as beyond the recipe
(bayes.FLAG_BEYOND_RECIPE) rather than dropped without a word: what the fit makes of it is reported apart.
A run can also keep every row and mark the ones the recipe would drop (FAILS_RECIPE, bayes.FLAG_FAILS_RECIPE),
so that its catalogue lines up row for row with its input.

The S/N of a band is 1.0857 over its magnitude error, read from the PSF fluxes of the survey table or, for a
prepared catalogue that carries only r and the colours, from the colour errors: the error on g is what is
left of the error on g-r once the error on r is taken out, and the same for i from r-i. A catalogue that
added a systematic term to its colour errors (the DP2 preparation adds 5e-5 in quadrature) gets a band error
larger by that term, which at the cut moves S/N 3 to 2.9994.
"""

import numpy as np

RECIPE_SNR = {"r": 10.0, "g": 3.0, "i": 3.0}
RECIPE_R_MAX = 23.5
MAG_PER_SNR = 1.0857  # 2.5 / ln 10: a magnitude error of 1.0857 / S/N
# The column in which a run that keeps every row (scripts/run_dp2.py --selection flag) marks the rows the
# recipe would have dropped, and from which the fit sets bayes.FLAG_FAILS_RECIPE.
FAILS_RECIPE = "failsRecipe"


def bandSnr(columns):
    """S/N in r, g and i, from whichever the columns hold: PSF fluxes and their errors, or r and the colours.

    columns is anything indexed by name (a DataFrame, or a dict of arrays). The flux names are those of the
    DP2 object table (r_psfFlux, r_psfFluxErr); the prepared names are rmagErr, grErr and riErr.
    """
    names = set(getattr(columns, "columns", columns))
    if {f"{b}_psf{s}" for b in "gri" for s in ("Flux", "FluxErr")} <= names:
        return {b: _array(columns, f"{b}_psfFlux") / _array(columns, f"{b}_psfFluxErr") for b in "gri"}
    rErr = _array(columns, "rmagErr")
    with np.errstate(invalid="ignore"):
        errors = {
            "r": rErr,
            "g": np.sqrt(np.maximum(_array(columns, "grErr") ** 2 - rErr**2, 0.0)),
            "i": np.sqrt(np.maximum(_array(columns, "riErr") ** 2 - rErr**2, 0.0)),
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        return {b: np.where(e >= 0, MAG_PER_SNR / e, np.nan) for b, e in errors.items()}


def recipeSelection(rmag, snr):
    """The objects that meet the recipe, and of those the ones past its r limit, as two boolean arrays.

    A missing value fails every cut it enters, so an object with no r error, or a colour error that is NaN,
    is not selected.
    """
    rmag = np.asarray(rmag, dtype=float)
    keep = np.isfinite(rmag)
    with np.errstate(invalid="ignore"):
        for band, least in RECIPE_SNR.items():
            keep &= np.asarray(snr[band], dtype=float) > least
    return keep, keep & beyondRecipe(rmag)


def beyondRecipe(rmag):
    """Whether each magnitude is past the r limit of the recipe."""
    with np.errstate(invalid="ignore"):
        return np.asarray(rmag, dtype=float) >= RECIPE_R_MAX


def failsRecipe(frame):
    """Whether each row of a frame is marked as failing the recipe; False on every row of one with no mark."""
    if FAILS_RECIPE not in getattr(frame, "columns", ()):
        return np.zeros(len(frame), dtype=bool)
    return np.asarray(frame[FAILS_RECIPE], dtype=bool)


def _array(columns, name):
    values = columns[name]
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(values, dtype=float)
