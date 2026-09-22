"""Build data/LSSTlocus_10Gyr_DP2.txt, the locus for Rubin DP2, from LSSTlocus_10Gyr_fix.txt.

Two things are wrong with the original locus on DP2 photometry, both measured on DP2 stars:

* u-g. At fixed spectroscopic [Fe/H] (DESI DR1) the observed u-g is offset from the locus by +0.03 mag for
  [Fe/H] < -1.5 and -0.16 mag at solar metallicity, the same in two fields. The offset is added to the
  locus u-g.
* Mr along the main sequence. With Gaia DR3 parallaxes for 331,000 stars in five fields (b = -11 to -29),
  the mean of parallax - E[parallax] per bin of g-i shows the locus too faint by 0.05-0.26 mag for G and K
  dwarfs and too bright by 0.14-0.5 mag for the red end. This is measured over all stars with a parallax
  (a signal-to-noise cut keeps the stars whose parallax scattered high and biases the result). The fit takes
  Mr = tLoc on the main sequence, so the correction moves the colours of each tLoc row rather than the Mr
  column: the row that carried the colours of a star now sits at a tLoc fainter or brighter by the offset.
  Half a magnitude of that at the red end reaches past the last row of the original tLoc grid, so the grid
  grows at the faint end rather than the reddest colours being cut off it.

  python scripts/make_locus.py                        # writes data/LSSTlocus_10Gyr_DP2.txt
  python scripts/make_locus.py --measure stars.parquet estimates.parquet
                                                       # prints the Mr table for another catalog
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
ORIGINAL = DATA / "LSSTlocus_10Gyr_fix.txt"
OUTPUT = DATA / "LSSTlocus_10Gyr_DP2.txt"

# u-g offset versus [Fe/H] (DESI DR1, COSMOS and l = 285, b = +69)
UG_FEH = np.array([-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
UG_OFFSET = np.array(
    [0.026, 0.026, 0.026, 0.026, 0.015, -0.043, -0.079, -0.110, -0.145, -0.155, -0.161, -0.161, -0.161]
)
# Mr offset versus dereddened g-i (Gaia DR3 parallaxes, five DP2 fields); positive = the locus is too bright.
# One entry per bin of measureMrOffsets, at the centre of the bin, so that a re-measurement prints this table
# row by row; the bins it prints are read back off GI, which keeps the two from drifting apart. The last
# entries sit at the clip of measureMrOffsets rather than at a measurement, and the redder colours of the
# locus take the same value, the interpolation holding the last entry.
GI = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1])
MR_OFFSET = np.array(
    [0.040, -0.052, -0.177, -0.264, 0.025, 0.142, 0.132, 0.260, 0.292, 0.275, 0.358, 0.471, 0.5, 0.5, 0.5]
)

COLS = ("tLoc", "Mr", "FeH", "ug", "gr", "ri", "iz", "zy")


def readLocus(path):
    """The locus table as a DataFrame (comment and header lines skipped)."""
    lines = [line for line in open(path).read().splitlines() if line.strip() and not line.startswith("#")]
    rows = [[float(x) for x in line.split()] for line in lines if line.split()[0] != "tLoc"]
    return pd.DataFrame(rows, columns=COLS)


def writeLocus(locus, path, comment):
    """Write the locus in the format of the data files, with one comment line."""
    with open(path, "w") as fh:
        fh.write(f"# {comment}\n   tLoc    Mr    FeH     ug      gr      ri      iz     zy \n")
        for row in locus.itertuples(index=False):
            fh.write(
                f"{row.tLoc:7.2f} {row.Mr:6.2f} {row.FeH:5.2f} "
                + " ".join(f"{v:7.3f}" for v in row[3:])
                + "\n"
            )


def shiftedTLoc(track):
    """The tLoc that each row of one main-sequence track belongs at, given the measured offsets.

    A row the parallaxes find too bright by dMr carries the colours of a star of tLoc + dMr. Forced to
    increase with tLoc, because the sequence has to stay one: the offsets are measured in bins of colour and
    a bin whose offset drops by more than the bin is wide would otherwise fold the sequence back on itself.
    """
    return np.maximum.accumulate(track.tLoc.to_numpy() + np.interp(track.gr + track.ri, GI, MR_OFFSET))


def extendFaintEnd(locus, tMax):
    """The locus with its tLoc grid continued at the faint end, up to tMax, in rows of main sequence.

    The reddest dwarfs move half a magnitude fainter, past the last row of the original grid, and a colour
    with no row to land on is one the fit can no longer reach: red M dwarfs then pile up on the last row of
    the grid with a large chi2 instead of fitting. The grid is the same for every [Fe/H], so all of the
    tracks are extended by the same rows, carrying Mr = tLoc like the faint end they continue and the
    colours of the row they follow until the resampling overwrites them. The prior maps are interpolated
    onto whatever grid the locus has and their own grid reaches tLoc = 17, so there is room to grow.
    """
    t = np.unique(locus.tLoc)
    step = round(float(t[-1] - t[-2]), 2)
    nRows = int(np.floor((tMax - t[-1]) / step + 1e-9))
    if nRows < 1:
        return locus
    extra = np.round(t[-1] + step * np.arange(1, nRows + 1), 2)
    blocks = []
    for _, track in locus.groupby("FeH", sort=False):
        rows = track.iloc[[-1] * nRows].reset_index(drop=True)
        rows["tLoc"] = extra
        rows["Mr"] = extra
        blocks += [track, rows]
    return pd.concat(blocks, ignore_index=True)


def buildLocus(locus):
    """Apply both corrections; returns a new table, on the (FeH, tLoc) grid extended at the faint end."""
    locus = locus.copy()
    locus["ug"] += np.interp(locus.FeH, UG_FEH, UG_OFFSET)
    colors = ["ug", "gr", "ri", "iz", "zy"]
    mainSequence = np.abs(locus.Mr - locus.tLoc) < 1e-3
    tracks = {}
    for feH, track in locus[mainSequence].groupby("FeH"):
        track = track.sort_values("tLoc")
        tracks[feH] = (shiftedTLoc(track), {c: track[c].to_numpy() for c in colors})
    locus = extendFaintEnd(locus, max(shifted[-1] for shifted, _ in tracks.values()))
    mainSequence = (np.abs(locus.Mr - locus.tLoc) < 1e-3).to_numpy()
    for feH, (shifted, track) in tracks.items():
        rows = mainSequence & (locus.FeH.to_numpy() == feH)
        # Outside the shifted sequence np.interp holds its end values, which is what the handful of rows
        # above its blue end need: the correction moves that end fainter and the main sequence has nothing
        # brighter than it to resample from, the turnoff branch above carrying those magnitudes instead.
        for c in colors:
            locus.loc[rows, c] = np.interp(locus.tLoc.to_numpy()[rows], shifted, track[c])
    return locus


def measureMrOffsets(stars, estimates, binEdges=None, minStars=300, nDraws=300):
    """The Mr table from a catalog with Gaia parallaxes (parallax, parallaxErr) and the fit estimates for it.

    stars needs rmag, gr, ri, parallax and, if the colours are to be dereddened, Ar; estimates needs the
    Mr_true and Ar quantiles of the fit, in the same row order. Every star with a parallax enters. The bins
    are the bins of GI unless given, so that the table printed is the table MR_OFFSET holds.
    """
    if binEdges is None:
        half = (GI[1] - GI[0]) / 2
        binEdges = np.append(GI - half, GI[-1] + half)
    plx = stars.parallax.to_numpy(float) + 0.017
    mu = {
        q: stars.rmag.to_numpy()
        - estimates[f"Mr_true_quantile_{o}"].to_numpy()
        - estimates.Ar_quantile_median.to_numpy()
        for q, o in (("lo", "hi"), ("med", "median"), ("hi", "lo"))
    }
    sigma = np.maximum((mu["hi"] - mu["lo"]) / 2, 0.02)
    draws = mu["med"][:, None] + sigma[:, None] * np.random.default_rng(0).standard_normal((len(plx), nDraws))
    expected = np.mean(10 ** (2 - draws / 5), axis=1)
    gi = stars.gr.to_numpy() + stars.ri.to_numpy() - (0.641 * stars.Ar.to_numpy() if "Ar" in stars else 0.0)
    print(f"{'g-i':>10} {'N':>7} {'<plx>':>7} {'<E>':>7} {'dMr':>7}")
    for lo, hi in zip(binEdges[:-1], binEdges[1:], strict=True):
        m = (gi >= lo) & (gi < hi) & np.isfinite(plx) & np.isfinite(expected)
        if m.sum() >= minStars:
            d = np.clip(5 * np.log10(plx[m].mean() / expected[m].mean()), -0.5, 0.5)
            print(
                f"{lo:4.1f}-{hi:<5.1f} {m.sum():7d} {plx[m].mean():7.4f} {expected[m].mean():7.4f} {d:+7.3f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--measure", nargs=2, metavar=("STARS", "ESTIMATES"), help="parquet files; print the Mr table"
    )
    args = ap.parse_args()
    if args.measure:
        measureMrOffsets(pd.read_parquet(args.measure[0]), pd.read_parquet(args.measure[1]))
    else:
        writeLocus(
            buildLocus(readLocus(ORIGINAL)),
            OUTPUT,
            "LSSTlocus_10Gyr_fix.txt with the u-g and Mr(g-i) corrections measured on Rubin DP2 "
            "(scripts/make_locus.py)",
        )
        print(f"written {OUTPUT}")
