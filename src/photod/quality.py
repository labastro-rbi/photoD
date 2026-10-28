"""The goodness-of-fit test of a fit against its own model: is the chi2 of the class a star is given too large
for a star the model would have drawn?

chi2min of the claimed class is not chi2-distributed with a known number of degrees of freedom: the A_r
bound, the reddening vector running along the white dwarf track in g-r and r-i, the mass axis and the colour
floor all change it, and differently for each set of measured colours. So the null distribution is taken
from the model itself (scripts/make_qc_table.py): stars drawn from the prior, given the noise of the survey,
and fitted again by the production fit. Its 99th percentile (QC_ALPHA) for each class, colour pattern and
|b| class is the table read here, and a star whose chi2min for the class it is given is above that is
flagged (bayes.FLAG_POOR_FIT_P).

The colour pattern is which colours were measured, as a bitmask over the fitted colours in their order (bit
0 for the first). The |b| classes are the latitude bins of the evaluation protocol, 0-20, 20-40 and 40-90
degrees, which is also where the A_r bound, and with it the null, changes most.
"""

import numpy as np
import pandas as pd

from photod.locus import CLASS_SEGMENTS

QC_CLASSES = tuple(CLASS_SEGMENTS)
QC_ALPHA = 0.01
LATITUDE_EDGES = (20.0, 40.0)
MIN_COLOURS = 3  # with two colours the white dwarf track and the reddening vector are degenerate
QC_COLUMNS = ("class", "latitude", "pattern", "colours", "nColors", "chi2", "level", "n", "nEff")


def colourPattern(colorsErr):
    """The measured colours of each star as a bitmask: a colour is measured where its error is finite."""
    measured = np.isfinite(np.asarray(colorsErr, dtype=float))
    return (measured * (1 << np.arange(measured.shape[1]))).sum(axis=1).astype(np.int64)


def patternColours(pattern, colors):
    """The colours a pattern says were measured, as one comma-separated string."""
    return ",".join(c for k, c in enumerate(colors) if int(pattern) >> k & 1)


def latitudeClass(ra, dec):
    """The |b| class of each position, 0 for |b| < 20 deg, 1 for 20-40 and 2 above."""
    from astropy.coordinates import SkyCoord

    b = SkyCoord(ra=np.asarray(ra, dtype=float), dec=np.asarray(dec, dtype=float), unit="deg").galactic.b.deg
    return np.searchsorted(LATITUDE_EDGES, np.abs(b), side="right").astype(np.int64)


def readQcTable(path, colors):
    """The thresholds of a QC table, as an array by class, |b| class and pattern; NaN where it has none.

    A table written for other colours, or in another order, is refused: its patterns would name other colours.
    """
    table = pd.read_csv(path, comment="#")
    missing = [c for c in QC_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} is not a QC table: it lacks {missing}")
    for pattern, said in zip(table["pattern"], table["colours"].fillna(""), strict=True):
        if patternColours(pattern, colors) != said:
            raise ValueError(f"{path} names pattern {pattern} {said!r}, which for {colors} is another set")
    out = np.full((len(QC_CLASSES), len(LATITUDE_EDGES) + 1, 1 << len(colors)), np.nan)
    unknown = sorted(set(table["class"]) - set(QC_CLASSES))
    if unknown:
        raise ValueError(f"{path} holds classes {unknown}, not among {QC_CLASSES}")
    cells = zip(table["class"], table["latitude"], table["pattern"], table["chi2"], strict=True)
    for name, lat, pattern, chi2 in cells:
        out[QC_CLASSES.index(name), int(lat), int(pattern)] = chi2
    return out


def claimedClass(statistics):
    """Index into QC_CLASSES of the most probable class of each star; -1 where the fit gave no answer."""
    p = np.stack([np.asarray(statistics[f"p_{name}"], dtype=float) for name in QC_CLASSES], axis=1)
    known = np.isfinite(p).all(axis=1)
    return np.where(known, np.argmax(np.nan_to_num(p, nan=-1.0), axis=1), -1)


def poorFit(thresholds, statistics, pattern, latitude):
    """Whether chi2min of each star's claimed class is above the table's threshold for its pattern and |b|.

    A star with no claimed class, or one whose cell the table leaves empty (no colours measured), is not
    flagged here: the first is FLAG_POOR_FIT already, and the second has nothing to test.
    """
    claimed = claimedClass(statistics)
    chi2 = np.stack([np.asarray(statistics[f"chi2min{name}"], dtype=float) for name in QC_CLASSES], axis=1)
    rows = np.arange(claimed.size)
    own = np.where(claimed >= 0, chi2[rows, np.maximum(claimed, 0)], np.nan)
    limit = np.where(claimed >= 0, thresholds[np.maximum(claimed, 0), latitude, pattern], np.nan)
    with np.errstate(invalid="ignore"):
        return own > limit
