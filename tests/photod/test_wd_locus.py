"""The locus with white dwarf and QSO tracks: Ivezic's sequences on the tLoc axis, the mass-resolved tracks
of scripts/make_wd_mass_locus.py, the QSO track of scripts/make_qso_locus.py, and the files in data/ they
build."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from wd_loci import COLORS, NAMES, read, script

import photod.locus as lt


def test_the_white_dwarf_loci_carry_their_segments(wdLoci):
    """Every row of a white dwarf locus says what it models; a locus without the column is all stars."""
    merged, only = wdLoci
    locus = lt.LSSTsimsLocus(fixForStripe82=False, datafile=merged, colnames=NAMES)
    assert "segment" in locus.colnames
    segment, tLoc = np.asarray(locus["segment"]), np.asarray(locus["tLoc"])
    assert set(np.unique(segment)) == {lt.SEGMENT_STARS, lt.SEGMENT_HYDROGEN_WD, lt.SEGMENT_HELIUM_WD}
    assert np.all(
        (tLoc[segment == lt.SEGMENT_HYDROGEN_WD] >= -2.0) & (tLoc[segment == lt.SEGMENT_HYDROGEN_WD] <= -0.5)
    )
    assert np.all(
        (tLoc[segment == lt.SEGMENT_HELIUM_WD] >= -6.0) & (tLoc[segment == lt.SEGMENT_HELIUM_WD] <= -4.5)
    )
    assert set(np.unique(lt.LSSTsimsLocus(datafile=only, colnames=NAMES)["segment"])) == {1, 2}
    # the white dwarf and QSO locus that ships says so too
    shipped = lt.LSSTsimsLocus(datafile=lt.WD_LOCUS_FILE, colnames=NAMES)
    assert set(np.unique(shipped["segment"])) == {0, 1, 2, 3}
    # and a locus without the column is all stars
    assert "segment" not in lt.LSSTsimsLocus(datafile=lt.RUN_LOCUS_FILE, colnames=NAMES).colnames


def test_a_segment_that_changes_with_feh_is_refused(wdLoci):
    """A class has to sit on the same stretch of the axis at every [Fe/H]."""
    table = lt.LSSTsimsLocus(datafile=wdLoci[0], colnames=NAMES)
    table = lt.subsampleLocusData(table, kMr=1, kFeH=15, yLabel="tLoc")
    table["segment"][0] = 3
    with pytest.raises(ValueError, match="segment"):
        lt.segmentAxis(table)


def test_the_mass_locus_is_the_single_mass_axis_with_a_colour_per_mass(massLocus, singleLocus):
    """Same rows, tLoc, Mr and segments as the single-mass locus; only the white dwarf colours differ."""
    mass, single = read(massLocus[0]), read(singleLocus)
    assert len(mass) == len(single)
    for name in ("tLoc", "Mr", "FeH", "segment"):
        assert np.array_equal(np.asarray(mass[name]), np.asarray(single[name])), name
    stars = np.asarray(mass["segment"]) == lt.SEGMENT_STARS
    for color in COLORS:
        assert np.array_equal(np.asarray(mass[color])[stars], np.asarray(single[color])[stars])
    assert lt.hasWhiteDwarfMassAxis(mass) and not lt.hasWhiteDwarfMassAxis(single)
    assert not lt.hasWhiteDwarfMassAxis(read(lt.DEFAULT_LOCUS_FILE.parent / "LSSTlocus_10Gyr_fix.txt"))


def test_the_anchor_mass_reproduces_ivezics_sequences():
    """The 0.6 Msun model with the DP2 adjustment carried over is his sequence, to interpolation precision."""
    m = script("make_wd_mass_locus")
    tables = m.readTables()
    logT = np.linspace(np.log10(3000.0), np.log10(150000.0), 4000)
    for code, (_, path) in m.TRACKS.items():
        sequence = np.loadtxt(path, comments="#")
        colours, cover = m.massTracks(tables, m.KINDS[code], sequence[:, 1:6], sequence[:, 0], [0.6], logT)
        assert np.abs(colours[0] - sequence[:, 1:6]).max() < 0.005
        assert cover[0][0] < 8.5 and cover[0][1] > 14.5
    # and a heavier white dwarf of the same colour is fainter
    seq = np.loadtxt(m.TRACKS[lt.SEGMENT_HYDROGEN_WD][1], comments="#")
    heavy = m.massTracks(tables, "H", seq[:, 1:6], seq[:, 0], [0.9], logT)[0][0]
    at12 = np.argmin(np.abs(seq[:, 0] - 12.0))
    gi = heavy[:, 1] + heavy[:, 2]
    order = np.argsort(gi)
    assert np.interp(seq[at12, 2] + seq[at12, 3], gi[order], seq[order, 0]) > 12.3


def test_the_mass_grid_is_the_literature_prior_on_mass(massLocus):
    """31 masses from 0.4 to 1.3 Msun, and the prior density of each track on them."""
    grid = massLocus[1]
    masses = np.asarray(grid["masses"])
    assert masses.size == 31 and masses[0] == 0.4 and masses[-1] == 1.3 and np.all(np.diff(masses) > 0)
    for key in ("wH", "wHe"):
        assert_allclose(np.sum(grid[key]) * 0.1, 1.0, rtol=1e-4)
    # the prior means: 0.66 Msun for hydrogen (the 40 pc mean), 0.60 for helium
    step = 1 / (3 * script("make_wd_mass_locus").massGrid(31)[1])
    for key, mean in (("wH", 0.66), ("wHe", 0.60)):
        density = np.asarray(grid[key]) / step
        assert abs(np.sum(density * masses * step) / np.sum(density * step) - mean) < 0.01


def test_the_qso_locus_is_the_white_dwarf_locus_with_a_track_of_its_own(qsoLoci):
    """Every row of the white dwarf locus is kept as it was; the track sits on QSO_TLOC with segment 3."""
    base, merged = read(qsoLoci[0]), read(qsoLoci[1])
    segment = np.asarray(merged["segment"])
    old = segment != lt.SEGMENT_QSO
    for name in ("tLoc", "Mr", "FeH", "segment", *COLORS):
        assert np.array_equal(np.asarray(merged[name])[old], np.asarray(base[name])), name
    tLoc = np.asarray(merged["tLoc"])[~old]
    assert_allclose([tLoc.min(), tLoc.max()], lt.QSO_TLOC)
    assert np.all(np.asarray(merged["Mr"])[~old] == lt.QSO_MR)
    assert np.unique(tLoc).size == 149 and (~old).sum() == 149 * 31
    assert qsoLoci[2]["locus"] == qsoLoci[1].name
    assert lt.hasWhiteDwarfMassAxis(merged)


def test_the_locus_and_mass_grid_that_ship_are_what_the_scripts_build(qsoLoci):
    """data/ holds the output of make_qso_locus.py on the files in data/, to the byte."""
    assert qsoLoci[1].read_bytes() == lt.WD_LOCUS_FILE.read_bytes()
    assert (qsoLoci[1].parent / "WD_massgrid_QSO.json").read_bytes() == (
        lt.WD_LOCUS_FILE.parent / "WD_massgrid_QSO.json"
    ).read_bytes()
    assert sorted(p.name for p in qsoLoci[1].parent.iterdir()) == [
        "LSSTlocus_10Gyr_WDmass.txt",
        "LSSTlocus_10Gyr_WDmass_QSO.txt",
        "WD_massgrid.json",
        "WD_massgrid_QSO.json",
    ]
