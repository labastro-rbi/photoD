"""Build the QSO track of the locus, and the white dwarf locus with it.

A class missing from the model has its objects absorbed by whichever class fits them best, and for the
UV-excess point sources of DP2 that is the white dwarfs. A QSO track with a prior share of its own is the same
remedy that the white dwarf tracks are for the stars.

The colours are those of the qsogen SED model (Temple, Hewett & Banerji 2021, MNRAS 508, 737) with the
parameters they fitted to the SDSS DR16Q median colours, emission lines, host light and Lyman forest
included, through the LSST baseline total throughputs. Both are pinned: qsogen at MJTemple/qsogen d2f9abf1
(the five files the model reads) and lsst/throughputs 1.9 (baseline/total_*.dat), checked by md5 before use.
The luminosity follows qsogen's DR16Q redshift relation shifted by MI_SHIFT = +2.5 mag, since the QSOs of
DP2 are about 2.5 mag fainter than the 18.6 < i < 19.1 sample it describes, and the model scales the lines
(Baldwin effect) and the host with M_i. The [Fe/H] rows, which a QSO has no use for, carry the scatter of
the continuum slope: dalpha = 0.6 ([Fe/H] + 1) on both power-law slopes, so the 31 rows span +-3 sigma of
sigma_alpha = 0.3 (Richards et al. 2003, Vanden Berk et al. 2001) and a prior N([Fe/H]; -1, 0.5) is that
scatter. The redshift runs over 0.2-5.0, linear in tLoc on locus.QSO_TLOC in steps of 0.01: below 0.2 a QSO
is host-dominated and extended, above 5 there are none at r < 25. Its Mr is locus.QSO_MR, a placeholder the
fit never reads, and its segment is locus.SEGMENT_QSO.

Files:

  data/QSO_track_qsogen_LSST.txt    the track (tLoc, z, FeH, colours), rewritten by --vendor from qsogen
  LSSTlocus_10Gyr_WDmass_QSO.txt    the mass-resolved white dwarf locus of make_wd_mass_locus.py, with the
                                    track: the locus of a run that tells stars, white dwarfs and QSOs apart
                                    (photod.locus.WD_LOCUS_FILE)
  WD_massgrid_QSO.json              its mass grid, as make_wd_mass_locus.py writes it but naming this locus, for
                                    make_priors.py --wd-mass-grid

The two in data/ are what this script writes from the files in data/.

Usage:
  python scripts/make_qso_locus.py [--out DIR]
  python scripts/make_qso_locus.py --vendor --qsogen DIR --throughputs DIR    (rewrites the track file)
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_wd_mass_locus as wdm  # noqa: E402

import photod.locus as lt  # noqa: E402

DATA = wdm.DATA
TRACK = DATA / "QSO_track_qsogen_LSST.txt"
LOCUS_NAME = "LSSTlocus_10Gyr_WDmass_QSO.txt"
MASSGRID_NAME = "WD_massgrid_QSO.json"
QSOGEN = "MJTemple/qsogen d2f9abf1ad23c489da8857f7e3c1bca862105d22"
QSOGEN_MD5 = {
    "config.py": "85dafa7fe26f6e738987a352c143f1f5",
    "qsosed.py": "f75fdc1d708054cb4933f8457440774c",
    "qsosed_emlines_20210625.dat": "25931a1a2f57008ce8fa65228576c534",
    "S0_template_norm.sed": "fee2bbac2446e77716ac31850ad5213a",
    "pl_ext_comp_03.sph": "4cf5d61db4cc4ba56a5238a0f6f0ceb6",
}
THROUGHPUTS = "lsst/throughputs 1.9 baseline"
THROUGHPUTS_MD5 = {
    "total_u.dat": "08555d14c690ad38a602b90a13b66c40",
    "total_g.dat": "29a47ff5fab48ac29baa73a59e5daf20",
    "total_r.dat": "a93c4bd0a901e5dda42d23770cc2c7cf",
    "total_i.dat": "9facc9343d8ed111c445a0296210b338",
    "total_z.dat": "ddba9384e06b6698b183a9b38c15906d",
    "total_y.dat": "e545721daa9f6029a1097f149d2423a6",
}
MI_SHIFT = 2.5
SLOPE_PER_FEH = 0.6  # dalpha per unit of the [Fe/H] axis, zero at [Fe/H] = -1
TLOC_STEP = 0.01


def citation(miShift=MI_SHIFT):
    """Where the track comes from, for the file headers."""
    return (
        f"qsogen (Temple, Hewett & Banerji 2021, MNRAS 508, 737; {QSOGEN}) through {THROUGHPUTS}; "
        f"M_i shift {miShift:+.1f}; dalpha = {SLOPE_PER_FEH} (FeH + 1); "
        f"z {lt.QSO_REDSHIFT[0]}-{lt.QSO_REDSHIFT[1]} on tLoc {lt.QSO_TLOC[0]}-{lt.QSO_TLOC[1]}"
    )


def trackTLoc():
    """The tLoc values of the track."""
    n = int(round((lt.QSO_TLOC[1] - lt.QSO_TLOC[0]) / TLOC_STEP)) + 1
    return np.round(np.linspace(*lt.QSO_TLOC, n), 3)


def checkPinned(directory, md5s, what):
    """Refuse files that are not the pinned ones: the track would be another model's without saying so."""
    for name, expected in md5s.items():
        found = hashlib.md5((Path(directory) / name).read_bytes()).hexdigest()
        if found != expected:
            raise SystemExit(f"{Path(directory) / name} is not the file of {what} (md5 {found})")


def abMagnitude(wavelength, flam, band):
    """AB magnitude of a spectrum f_lambda(wavelength in Angstrom) through a (wavelength, throughput) band."""
    w, throughput = band
    f = np.interp(w, wavelength, flam, left=0.0, right=0.0)
    num = np.trapezoid(f * throughput * w, w)
    den = np.trapezoid(throughput / w, w) * 2.998e18
    return -2.5 * np.log10(num / den) - 48.6


def trackColours(qsogen, throughputs, feH, z, miShift=MI_SHIFT):
    """The colours (ug, gr, ri, iz, zy) of the model at every [Fe/H] row and redshift, shape (feH, z, 5)."""
    checkPinned(qsogen, QSOGEN_MD5, QSOGEN)
    checkPinned(throughputs, THROUGHPUTS_MD5, THROUGHPUTS)
    bands = {}
    for b in "ugrizy":
        table = np.loadtxt(Path(throughputs) / f"total_{b}.dat", comments="#")
        bands[b] = (table[:, 0] * 10.0, table[:, 1])  # nm to Angstrom
    # qsogen reads its templates from the working directory when it is imported
    here = os.getcwd()
    os.chdir(qsogen)
    sys.path.insert(0, str(qsogen))
    try:
        from config import params as defaults
        from qsosed import Quasar_sed
    finally:
        os.chdir(here)
    wavelength = np.logspace(2.95, 4.48, num=20001)
    out = np.zeros((len(feH), len(z), 5))
    for i, f in enumerate(feH):
        params = dict(defaults)
        dalpha = SLOPE_PER_FEH * (f + 1.0)
        params["plslp1"] = defaults["plslp1"] + dalpha
        params["plslp2"] = defaults["plslp2"] + dalpha
        for j, zz in enumerate(z):
            mi = np.interp(zz, *defaults["zlum_lumval"]) + miShift
            sed = Quasar_sed(z=zz, LogL3000=46, wavlen=wavelength, params=params, M_i=mi)
            m = [abMagnitude(sed.wavred, sed.flux, bands[b]) for b in "ugrizy"]
            out[i, j] = -np.diff(m)
    if not np.all(np.isfinite(out)):
        raise SystemExit("the model gave no flux in a band somewhere on the track")
    return out


def vendor(qsogen, throughputs, miShift=MI_SHIFT, path=TRACK):
    """Rewrite the track file from the model."""
    feH = np.unique(np.loadtxt(wdm.STARS, comments="#", skiprows=2)[:, 2])
    tLoc = trackTLoc()
    z = lt.qsoRedshift(tLoc)
    colours = trackColours(qsogen, throughputs, feH, z, miShift)
    with open(path, "w") as out:
        out.write(f"# QSO track: {citation(miShift)}\n")
        out.write("   tLoc       z     FeH      ug       gr       ri       iz       zy\n")
        for i, f in enumerate(feH):
            for j, t in enumerate(tLoc):
                values = " ".join(f"{c:8.4f}" for c in colours[i, j])
                out.write(f"{t:8.3f} {z[j]:7.4f} {f:6.2f} {values}\n")
    print(f"{path}: {feH.size} x {tLoc.size} points")


def trackRows(path=TRACK):
    """The track as locus rows: (tLoc, QSO_MR, FeH, colours, SEGMENT_QSO)."""
    track = np.loadtxt(path, comments="#", skiprows=2)
    n = len(track)
    return np.column_stack(
        [track[:, 0], np.full(n, lt.QSO_MR), track[:, 2], track[:, 3:8], np.full(n, lt.SEGMENT_QSO)]
    )


def withTrack(rows, track):
    """Locus rows with the track added, refused where the two do not share the [Fe/H] rows or overlap."""
    if not np.array_equal(np.unique(rows[:, 2]), np.unique(track[:, 2])):
        raise SystemExit("the QSO track is not on the [Fe/H] rows of the locus")
    taken = (rows[:, 0] >= lt.QSO_TLOC[0]) & (rows[:, 0] <= lt.QSO_TLOC[1])
    if taken.any():
        raise SystemExit(f"the locus already has points on the tLoc of the QSO track, {lt.QSO_TLOC}")
    return np.vstack([rows, track])


def build(out=DATA, track=TRACK):
    """The mass-resolved white dwarf locus with the QSO track, and its mass grid, into the directory out.

    The locus without the track is made by make_wd_mass_locus.py in a temporary directory and read back from
    its file there; the directory is deleted with both files of make_wd_mass_locus.py in it.
    """
    out = Path(out)
    with tempfile.TemporaryDirectory() as scratch:
        grid = wdm.build(scratch)
        base = np.loadtxt(Path(scratch) / wdm.LOCUS_NAME, comments="#", skiprows=2)
    merged = withTrack(base, trackRows(track))
    wdm.write(out / LOCUS_NAME, merged, f"{wdm.LOCUS_NAME} with the QSO track of {Path(track).name}")
    with open(track) as handle:
        source = handle.readline().lstrip("# ").strip()
    grid = dict(grid, locus=LOCUS_NAME, derivedFrom=wdm.LOCUS_NAME, qso=source)
    (out / MASSGRID_NAME).write_text(json.dumps(grid, indent=1) + "\n")
    print(f"{out / MASSGRID_NAME}: the mass grid of {wdm.LOCUS_NAME}, for {LOCUS_NAME}")
    return grid


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA), help="directory to write the loci and the mass grid to")
    ap.add_argument("--vendor", action="store_true", help="rewrite the track file from qsogen")
    ap.add_argument("--qsogen", default="", help="directory of qsogen at the pinned commit (with --vendor)")
    ap.add_argument(
        "--throughputs", default="", help="directory of the LSST total_*.dat files (with --vendor)"
    )
    ap.add_argument(
        "--mi-shift", type=float, default=MI_SHIFT, help="shift of the DR16Q M_i(z) relation (with --vendor)"
    )
    ap.add_argument("--track", default=str(TRACK), help="the track file to build the locus with")
    args = ap.parse_args()
    if args.vendor:
        if not (args.qsogen and args.throughputs):
            raise SystemExit("--vendor needs --qsogen and --throughputs")
        vendor(Path(args.qsogen).resolve(), Path(args.throughputs).resolve(), args.mi_shift, Path(args.track))
        return
    build(args.out, args.track)


if __name__ == "__main__":
    main()
