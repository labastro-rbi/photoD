"""Build the white dwarf loci with a mass axis: on the white dwarf rows the [Fe/H] axis is the mass.

At a fixed colour, which fixes the effective temperature, a white dwarf's absolute magnitude depends on its
radius and so on its mass: about 0.3 mag across the observed spread of 0.1-0.15 Msun. A single 0.6 Msun track
makes the distance posterior far too narrow. A white dwarf's colours do not depend on [Fe/H], so the 31 rows
the stars use for it can carry 31 masses at no cost in grid size, and the prior and the marginalisation then
integrate over mass as they do over [Fe/H].

The white dwarf tracks start from Zeljko Ivezic's DP2 white dwarf colour sequences (ivezic/PhotoD 589e9f7,
data/WD_Hydrogen_MrColorSequences_DP2.txt and data/WD_Helium_MrColorSequences_DP2.txt, the second spelt
WD_Heliuum_... there). They give LSST colours as a function of Mr for hydrogen (DA) and helium (DB, DC) white
dwarfs, Bergeron models adjusted to fit DP2, Mr 8.50-14.50 in steps of 0.02. Each goes on the tLoc axis below
everything the stellar locus uses, at locus.HELIUM_TLOC and locus.HYDROGEN_TLOC, one tLoc step of 0.005 per
0.02 in Mr. Every row carries its segment (photod.locus.SEGMENT_*): the stars 0, the hydrogen track 1, the
helium track 2, which is what the fit reads the classes from.

The colours of each mass come from the Montreal cooling tables, Table_Mass_0.2 to 1.3 (Bedard et al. 2020
cooling sequences, Holberg & Bergeron 2006 synthetic photometry; downloaded 2026-10-25 from
https://www.astro.umontreal.ca/~bergeron/CoolingModels/Tables/), kept in data/WD_Montreal_Table_Mass.txt:
SDSS ugriz and Pan-STARRS z, y, interpolated linearly in mass at a fixed log Teff. Ivezic's sequences are the
0.6 Msun model adjusted to DP2, and both match that model best (colour rms 0.03 against 0.06 at 0.5 or 0.7
Msun). The adjustment, his sequence minus the 0.6 Msun model, is carried over to every other mass at the same
Teff, since filter and calibration corrections depend on the spectrum. The 0.6 Msun row therefore reproduces
his sequence, and the filter systems of the tables need not be LSST's.

The masses are 31 values from 0.40 to 1.30 Msun (the range of the tables), spaced uniformly in
0.5 CDF + 0.5 uniform of the hydrogen prior: dense at its peak and still covering the tails. The prior on mass
is the literature's, not fitted to any data here:

  hydrogen  N(0.59, 0.035) + N(0.76, 0.15), 100 pc SDSS DAs with Teff >= 6000 K (Kilic et al. 2020,
            arXiv:2006.00323), with a weight of 0.40 on the broad part, which makes the mean mass 0.66 Msun,
            the 40 pc mean of McCleery et al. 2020 as quoted by O'Brien et al. 2024 (arXiv:2312.02735)
  helium    N(0.60, 0.05), between the DB, DC and DZ means of Kilic et al. 2025 (arXiv:2412.04611), with no
            massive tail

Two files are written, into the directory --out:

  LSSTlocus_10Gyr_WDmass.txt   LSSTlocus_10Gyr_fix.txt with both mass-resolved white dwarf tracks
  WD_massgrid.json             the mass of each [Fe/H] row, the prior density of each track per unit [Fe/H]
                               (wH, wHe) and the Mr range each row's model covers (coverH, coverHe), which
                               make_priors.py --wd-mass-grid reads; a massive white dwarf brighter than its
                               hottest model (150 kK) has no model and gets no prior

scripts/make_qso_locus.py adds the QSO track to them, which gives the locus a run fits; it builds them in a
temporary directory of its own, so neither is kept in data/.

Usage:
  python scripts/make_wd_mass_locus.py --out DIR
  python scripts/make_wd_mass_locus.py --vendor DIR_OF_Table_Mass_FILES    (rewrites the tables file)
"""

import argparse
import json
from pathlib import Path

import numpy as np

import photod.locus as lt

DATA = Path(__file__).resolve().parents[1] / "data"
STARS = DATA / "LSSTlocus_10Gyr_fix.txt"
TRACKS = {
    lt.SEGMENT_HELIUM_WD: (lt.HELIUM_TLOC, DATA / "WD_Helium_MrColorSequences_DP2.txt"),
    lt.SEGMENT_HYDROGEN_WD: (lt.HYDROGEN_TLOC, DATA / "WD_Hydrogen_MrColorSequences_DP2.txt"),
}
TABLES = DATA / "WD_Montreal_Table_Mass.txt"
LOCUS_NAME = "LSSTlocus_10Gyr_WDmass.txt"
MASSGRID_NAME = "WD_massgrid.json"
TABLE_MASSES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
MASS_RANGE = (0.40, 1.30)
ANCHOR_MASS = 0.6  # the model Ivezic's sequences were adjusted from
KINDS = {lt.SEGMENT_HYDROGEN_WD: "H", lt.SEGMENT_HELIUM_WD: "He"}
# columns of a Table_Mass row kept here, by their place in the 43 of the file
TABLE_COLUMNS = {
    "Teff": 0,
    "logg": 1,
    "u": 24,
    "g": 25,
    "r": 26,
    "i": 27,
    "z": 28,
    "zPS1": 32,
    "yPS1": 33,
    "age": 42,
}
CITATION = (
    "Montreal white dwarf cooling tables Table_Mass_* "
    "(https://www.astro.umontreal.ca/~bergeron/CoolingModels/Tables/, downloaded 2026-10-25): "
    "Bedard et al. 2020, ApJ 901, 93 (cooling); Holberg & Bergeron 2006, AJ 132, 1221 (photometry); "
    "Bergeron et al. 1995; Kowalski & Saumon 2006; Tremblay et al. 2011; Bergeron et al. 2011; "
    "Blouin et al. 2018 (atmospheres)"
)


def write(path, rows, title):
    """Locus rows (tLoc, Mr, FeH, colours, segment) as a locus file, in the order the fit reads a grid: tLoc
    within blocks of [Fe/H]."""
    rows = rows[np.lexsort((rows[:, 0], rows[:, 2]))]
    with open(path, "w") as out:
        out.write(f"# {title}\n")
        out.write("   tLoc      Mr     FeH      ug       gr       ri       iz       zy  segment\n")
        for r in rows:
            colours = " ".join(f"{c:8.4f}" for c in r[3:8])
            out.write(f"{r[0]:8.3f} {r[1]:7.3f} {r[2]:6.2f} {colours} {int(r[8]):8d}\n")
    print(f"{path}: {len(rows)} rows")


def gaussian(m, mean, sigma):
    """The normal density."""
    return np.exp(-0.5 * ((m - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def hydrogenMassPrior(m, broad=0.40):
    """0.60 N(0.59, 0.035) + 0.40 N(0.76, 0.15), Kilic et al. 2020; the 0.40 makes the mean 0.66 Msun."""
    return (1 - broad) * gaussian(m, 0.59, 0.035) + broad * gaussian(m, 0.76, 0.15)


def heliumMassPrior(m):
    """N(0.60, 0.05): Kilic et al. 2025, DB 0.586 +- 0.036, DC 0.61 +- 0.06, DZ 0.635 +- 0.062."""
    return gaussian(m, 0.60, 0.05)


MASS_PRIORS = {"H": hydrogenMassPrior, "He": heliumMassPrior}


def readRawTable(path):
    """The hydrogen and helium blocks of one Table_Mass_* file, the kept columns, in order of Teff."""
    lines = Path(path).read_text().splitlines()
    blocks, i = {}, 0
    while i < len(lines):
        words = lines[i].split()
        if len(words) > 3 and words[2] == "Mo" and "grid" in lines[i]:
            n, kind = int(words[0]), "H" if "hydrogen" in lines[i] else "He"
            rows = np.array([[float(x) for x in line.split()] for line in lines[i + 2 : i + 2 + n]])
            if rows.shape != (n, 43):
                raise ValueError(f"{path}: the {kind} block holds {rows.shape}, not {n} rows of 43")
            rows = rows[np.argsort(rows[:, 0])]
            blocks[kind] = rows[:, list(TABLE_COLUMNS.values())]
            i += 2 + n
        else:
            i += 1
    return blocks


def vendor(source, out=TABLES):
    """The columns used here of every Table_Mass_* in the directory source, as one text file."""
    with open(out, "w") as handle:
        handle.write(f"# {CITATION}\n")
        handle.write("# SDSS ugriz and Pan-STARRS z, y absolute magnitudes; age in years\n")
        handle.write("kind mass " + " ".join(TABLE_COLUMNS) + "\n")
        for mass in TABLE_MASSES:
            for kind, rows in readRawTable(Path(source) / f"Table_Mass_{mass}").items():
                for row in rows:
                    values = " ".join(f"{v:.3f}" for v in row[:-1])
                    handle.write(f"{kind} {mass:.1f} {values} {row[-1]:.4e}\n")
    print(f"{out}: {len(TABLE_MASSES)} masses")


def readTables(path=TABLES):
    """{(kind, mass): rows of TABLE_COLUMNS} from the vendored file."""
    kinds = np.loadtxt(path, comments="#", skiprows=3, usecols=0, dtype=str)
    values = np.loadtxt(path, comments="#", skiprows=3, usecols=range(1, 2 + len(TABLE_COLUMNS)))
    return {
        (kind, mass): values[(kinds == kind) & np.isclose(values[:, 0], mass), 1:]
        for kind in KINDS.values()
        for mass in TABLE_MASSES
    }


def modelColours(rows):
    """Mr and the colours ug, gr, ri, iz (SDSS) and zy (Pan-STARRS) of table rows."""
    c = {name: rows[:, k] for k, name in enumerate(TABLE_COLUMNS)}
    colours = [c["u"] - c["g"], c["g"] - c["r"], c["r"] - c["i"], c["i"] - c["z"], c["zPS1"] - c["yPS1"]]
    return c["r"], np.column_stack(colours)


def modelAtMass(tables, kind, mass, logT):
    """Mr and colours at mass on the points logT, linear in mass between the two tables either side of it.

    NaN where either of the two tables does not reach that temperature.
    """
    lo = max(m for m in TABLE_MASSES if m <= mass + 1e-9)
    hi = min(m for m in TABLE_MASSES if m >= mass - 1e-9)
    both = []
    for m in (lo, hi):
        rows = tables[(kind, m)]
        logTeff = np.log10(rows[:, 0])
        mr, colours = modelColours(rows)
        v = np.column_stack([np.interp(logT, logTeff, x) for x in (mr, *colours.T)])
        v[(logT < logTeff[0]) | (logT > logTeff[-1])] = np.nan
        both.append(v)
    w = 0.0 if hi == lo else (mass - lo) / (hi - lo)
    v = (1 - w) * both[0] + w * both[1]
    return v[:, 0], v[:, 1:]


def onMr(mr, colours, grid):
    """Colours tabulated against Mr, interpolated onto grid, and the Mr range of the model.

    Beyond that range the end colours are held; those points get no prior (WD_massgrid.json cover*).
    """
    ok = np.isfinite(mr) & np.isfinite(colours).all(axis=1)
    order = np.argsort(mr[ok])
    x, c = mr[ok][order], colours[ok][order]
    return np.column_stack([np.interp(grid, x, c[:, k]) for k in range(c.shape[1])]), (
        float(x[0]),
        float(x[-1]),
    )


def massGrid(n, reference=hydrogenMassPrior):
    """n masses uniform in u = 0.5 CDF + 0.5 uniform of the reference prior on MASS_RANGE, and du/dM there."""
    m = np.linspace(*MASS_RANGE, 20001)
    cdf = np.cumsum(reference(m))
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    u = 0.5 * cdf + 0.5 * (m - MASS_RANGE[0]) / (MASS_RANGE[1] - MASS_RANGE[0])
    masses = np.interp(np.linspace(0, 1, n), u, m)
    return masses, np.interp(masses, m, np.gradient(u, m))


def massTracks(tables, kind, sequence, mrGrid, masses, logT):
    """The colours of every mass on mrGrid, with the DP2 adjustment carried over at the same Teff.

    Returns (masses, mrGrid, 5) colours and the Mr range each mass's model covers.
    """
    mr06, c06 = modelAtMass(tables, kind, ANCHOR_MASS, logT)
    delta = sequence - onMr(mr06, c06, mrGrid)[0]  # the adjustment, as a function of the anchor's Mr
    # at each temperature, the adjustment the anchor model has there
    deltaAtT = np.column_stack([np.interp(mr06, mrGrid, delta[:, k]) for k in range(delta.shape[1])])
    colours, cover = [], []
    for mass in masses:
        mr, c = modelAtMass(tables, kind, mass, logT)
        row, covered = onMr(mr, c + deltaAtT, mrGrid)
        colours.append(row)
        cover.append(covered)
    return np.stack(colours), cover


def build(out, tables=TABLES):
    """Both files, into the directory out."""
    out = Path(out)
    tables = readTables(tables)
    stars = np.loadtxt(STARS, comments="#", skiprows=2)
    stars = np.column_stack([stars, np.full(len(stars), lt.SEGMENT_STARS)])
    feH = np.unique(stars[:, 2])
    masses, slope = massGrid(feH.size)
    logT = np.linspace(np.log10(3000.0), np.log10(150000.0), 4000)
    grid = {"locus": LOCUS_NAME, "feH": feH.round(2).tolist(), "masses": masses.round(4).tolist()}
    grid["transfer"] = "Ivezic DP2 sequence minus the Montreal 0.6 Msun model, carried over at the same Teff"
    grid["priors"] = {kind: MASS_PRIORS[kind].__doc__ for kind in KINDS.values()}
    grid["tables"] = CITATION
    rows = []
    for code, (tLoc, path) in TRACKS.items():
        kind = KINDS[code]
        sequence = np.loadtxt(path, comments="#")
        mrGrid = sequence[:, 0]
        colours, cover = massTracks(tables, kind, sequence[:, 1:6], mrGrid, masses, logT)
        t = tLoc[0] + (mrGrid - mrGrid[0]) * (tLoc[1] - tLoc[0]) / (mrGrid[-1] - mrGrid[0])
        for f, c in zip(feH, colours, strict=True):
            n = mrGrid.size
            rows.append(np.column_stack([t, mrGrid, np.full(n, f), c, np.full(n, code)]))
        # density of the prior per unit [Fe/H]: p(M) dM/d[Fe/H], with [Fe/H] linear in the spacing variable
        weight = MASS_PRIORS[kind](masses) / ((feH[-1] - feH[0]) * slope)
        grid[f"w{kind}"] = (weight / (weight.sum() * (feH[1] - feH[0]))).round(6).tolist()
        grid[f"cover{kind}"] = [[round(lo, 4), round(hi, 4)] for lo, hi in cover]
    tracks = np.vstack(rows)
    source = "mass-resolved white dwarf tracks (Montreal tables, Ivezic's DP2 adjustment; WD_massgrid.json)"
    write(out / LOCUS_NAME, np.vstack([stars, tracks]), f"LSSTlocus_10Gyr_fix.txt with the {source}")
    (out / MASSGRID_NAME).write_text(json.dumps(grid, indent=1) + "\n")
    print(f"{out / MASSGRID_NAME}: {masses.size} masses {masses[0]:.2f}-{masses[-1]:.2f} Msun")
    return grid


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="directory to write the two files to")
    ap.add_argument("--vendor", default="", help="directory of the Table_Mass_* files, to rewrite the tables")
    args = ap.parse_args()
    if args.vendor:
        vendor(args.vendor)
        return
    if not args.out:
        raise SystemExit("--out names the directory to write the two files to")
    build(args.out)


if __name__ == "__main__":
    main()
