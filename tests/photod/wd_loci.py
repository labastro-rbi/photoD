"""The loci with white dwarf and QSO tracks that the tests fit and build prior maps on, and the scripts that
build them, as modules."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import photod.locus as lt

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
COLORS = ("ug", "gr", "ri", "iz", "zy")
NAMES = ["tLoc", "Mr", "FeH", *COLORS]


def script(name):
    """One of the scripts as a module."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    """A locus file as the fit reads it."""
    return lt.LSSTsimsLocus(fixForStripe82=False, datafile=path, colnames=NAMES)


def singleMassLoci(out):
    """LSSTlocus_10Gyr_fix.txt with Ivezic's two white dwarf sequences as tracks of one mass, on the tLoc of
    scripts/make_wd_mass_locus.py and repeated at every [Fe/H], and the two tracks alone, written to out."""
    make = script("make_wd_mass_locus")
    stars = np.loadtxt(make.STARS, comments="#", skiprows=2)
    stars = np.column_stack([stars, np.full(len(stars), lt.SEGMENT_STARS)])
    feH = np.unique(stars[:, 2])
    tracks = []
    for segment, (tLoc, path) in make.TRACKS.items():
        seq = np.loadtxt(path, comments="#")
        mr, colours = seq[:, 0], seq[:, 1:6]
        t = tLoc[0] + (mr - mr[0]) * (tLoc[1] - tLoc[0]) / (mr[-1] - mr[0])
        for f in feH:
            tracks.append(np.column_stack([t, mr, np.full(mr.size, f), colours, np.full(mr.size, segment)]))
    tracks = np.vstack(tracks)
    make.write(out / "LSSTlocus_10Gyr_WD.txt", np.vstack([stars, tracks]), "stars and single-mass tracks")
    make.write(out / "LSSTlocus_WD_only.txt", tracks, "single-mass white dwarf tracks alone")
    return out / "LSSTlocus_10Gyr_WD.txt", out / "LSSTlocus_WD_only.txt"
