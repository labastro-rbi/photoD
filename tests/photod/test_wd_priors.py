"""The prior maps of a locus with white dwarf and QSO tracks (scripts/make_priors.py): the grid they reach
down to, the segments the model stars are placed on, the track maps, the prior on mass, and the QSO
population."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from wd_loci import read, script

import photod.locus as lt
from photod.priors import getBayesConstants

# the totals of Palanque-Delabrouille et al. 2016, Table 7, per r bin: PLE, then PLE+LEDE
TABLE7_TOTALS = (
    [73, 194, 544, 1644, 5185, 15669, 40203, 82311, 137938, 203090, 280774, 376990, 497720, 649595, 841048]
    + [1083262, 1390804],
    [88, 227, 630, 1830, 5404, 15314, 38401, 80354, 139813, 209907, 286955, 373634, 475834, 600426, 754956]
    + [948101, 1190284],
)


def test_the_grid_reaches_down_to_the_white_dwarf_tracks_on_the_step_of_the_stars():
    """A stellar locus keeps the grid the maps have always had; a locus with tracks below -2 extends it."""
    stars = getBayesConstants()
    assert (stars["MrFaint"], stars["MrBright"], stars["MrNpts"]) == (17.0, -2.0, 381)
    assert getBayesConstants(1.65) == stars
    tracks = getBayesConstants(lt.HELIUM_TLOC[0])
    assert (tracks["MrBright"], tracks["MrNpts"]) == (-6.0, 461)
    assert {k: v for k, v in tracks.items() if k not in ("MrBright", "MrNpts")} == {
        k: v for k, v in stars.items() if k not in ("MrBright", "MrNpts")
    }
    a = np.linspace(stars["MrFaint"], stars["MrBright"], stars["MrNpts"])
    b = np.linspace(tracks["MrFaint"], tracks["MrBright"], tracks["MrNpts"])
    assert_allclose(b[: a.size], a, atol=1e-12)
    assert json.loads(json.dumps(tracks)) == tracks


def test_a_locus_without_white_dwarf_tracks_keeps_the_label_map_as_it_is(wdLoci):
    """LSSTlocus_10Gyr_fix.txt places the model stars by SEGMENT_LABEL_MAP itself, and a white dwarf fainter
    than the turn-off at tLoc = Mr, as on a locus that never had tracks; on a locus with them the map moves up
    by the two tracks, which take the white dwarfs alone."""
    priors = script("make_priors")
    feH, plain, *_ = priors.locusAxes(lt.RUN_LOCUS_FILE)
    for i, feh in enumerate(feH):
        labels = lt.SEGMENT_LABEL_MAP[round(float(feh), 1)]
        toSeg = plain[i]["labelToSeg"]
        assert toSeg.shape[1] > max(labels) and not plain[i]["whiteDwarfTracks"]
        for k in range(toSeg.shape[1]):
            assert set(np.where(toSeg[:, k])[0]) == set(labels.get(k, ())), (feh, k)
    stars = pd.DataFrame(
        {"FeH": [-1.0, -1.0, -0.5, -0.5], "Mr": [12.0, 6.0, 12.0, 2.0], "label": [9, 1, 9, 9]}
    )
    placed = lt.assignTLocPartition(stars, plain, feH)["tLoc"].to_numpy()
    assert_allclose(placed[:3], [12.0, 6.0, 12.0])
    assert np.isfinite(placed[3]) and placed[3] <= 4.0
    feH, shifted, *_ = priors.locusAxes(wdLoci[0])
    for i, feh in enumerate(feH):
        labels = lt.SEGMENT_LABEL_MAP[round(float(feh), 1)]
        toSeg = shifted[i]["labelToSeg"]
        assert shifted[i]["whiteDwarfTracks"] and toSeg.shape[1] == plain[i]["labelToSeg"].shape[1] + 2
        assert set(np.where(toSeg[:, 0])[0]) == {lt.HELIUM_WHITE_DWARF_LABEL}
        assert set(np.where(toSeg[:, 1])[0]) == {lt.WHITE_DWARF_LABEL}
        for k in range(2, toSeg.shape[1]):
            assert set(np.where(toSeg[:, k])[0]) == set(labels.get(k - 2, ())) - {9}, (feh, k)
    onTrack = lt.assignTLocPartition(stars, shifted, feH)["tLoc"].to_numpy()
    assert -2.0 <= onTrack[0] <= -0.5 and onTrack[1] == 6.0


def test_the_white_dwarf_tracks_are_unit_mass_shapes_and_the_weights_split_each_dwarf():
    """A track map has unit mass and stays on its track; each white dwarf is split whole between the two."""
    priors = script("make_priors")
    bc = getBayesConstants(lt.HELIUM_TLOC[0])
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    xGrid = np.linspace(bc["FeHmin"], bc["FeHmax"], bc["FeHNpts"])
    yGrid = np.linspace(bc["MrFaint"], bc["MrBright"], bc["MrNpts"])
    track = (np.linspace(8.5, 14.5, 301), np.linspace(-2.0, -0.5, 301))
    rng = np.random.default_rng(5)
    dwarfs = pd.DataFrame({"rmag": rng.uniform(19, 25, 400), "Mr": rng.uniform(10, 14, 400)})
    for code in (lt.SEGMENT_HYDROGEN_WD, lt.SEGMENT_HELIUM_WD):
        maps = priors.trackMaps(dwarfs, True, code, track, rGrid, bc, xGrid, yGrid)
        dx, dy = xGrid[1] - xGrid[0], abs(yGrid[1] - yGrid[0])
        mass = maps.sum(axis=(1, 2)) * dx * dy * (xGrid.size - 1) / xGrid.size
        assert_allclose(mass, 1.0, rtol=0.03)
        assert np.all(
            maps[:, (yGrid > -0.3) | (yGrid < -2.2), :].sum(axis=(1, 2)) < 0.02 * maps.sum(axis=(1, 2))
        )
    mr = np.linspace(8.5, 14.5, 50)
    weights = sum(priors.trackWeights(mr, True, code) for code in lt.WHITE_DWARF_SEGMENTS)
    assert_allclose(weights, 1.0)
    assert_allclose(priors.trackWeights(mr, False, lt.SEGMENT_HYDROGEN_WD), 1.0)
    flat = priors.trackMaps(None, True, 1, track, rGrid, bc, xGrid, yGrid)
    assert np.all(flat[:, (yGrid >= -2.0) & (yGrid <= -0.5), 0] > 0)


def test_a_white_dwarf_past_the_end_of_the_tracks_is_not_in_the_shares(monkeypatch):
    """The shares count the white dwarfs the track maps are made of: a Mr 15.5 one counts nowhere."""
    priors = script("make_priors")
    rng = np.random.default_rng(3)
    n = 40000
    labels = rng.choice([1, 9], n, p=[0.9, 0.1])
    mr = np.where(labels == 9, rng.choice([12.0, 15.5], n), rng.uniform(0, 10, n))
    cone = pd.DataFrame({"rmag": rng.uniform(14, 27, n), "label": labels, "FeH": -1.0, "Mr": mr})
    monkeypatch.setattr(priors.lsdb, "open_catalog", lambda *a, **k: SimpleNamespace(compute=lambda: cone))
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    track = (np.linspace(8.5, 14.5, 301), np.linspace(-2.0, -0.5, 301))
    tracks = {lt.SEGMENT_HYDROGEN_WD: track, lt.SEGMENT_HELIUM_WD: (track[0], track[1] - 4.0)}
    args = ("trilegal", 0.0, 0.0, 4.0, rGrid, 0.5, np.array([-2.5, 0.5]), -4.0, 16.0, tracks)
    shares, wide = priors.populationShares(*args)
    held = cone[(cone.label != 9) | (cone.Mr <= 14.5)]
    i = int(np.argmin(np.abs(rGrid - 22.0)))
    near = np.abs(held.rmag.to_numpy() - rGrid[i]) <= 0.5
    dwarfs = near & (held.label.to_numpy() == 9)
    assert_allclose(shares[i, 1:].sum(), dwarfs.sum() / near.sum(), rtol=1e-12)
    assert_allclose(shares.sum(axis=1), 1.0)
    assert wide["Mr"].max() <= 14.5


def test_the_mass_weights_spread_each_tLoc_over_the_masses_it_has_models_for(massLocus):
    """Mean one over the rows at every tLoc of a track, zero where a mass has no model, one elsewhere."""
    path, grid = massLocus
    weights = script("make_priors").massWeights(grid, path)
    locus = read(path)
    segment = lt.segmentAxis(locus)
    mr = np.asarray(locus["Mr"]).reshape(weights.shape)[0]
    assert np.all(weights[:, segment == lt.SEGMENT_STARS] == 1)
    for code, key in ((lt.SEGMENT_HYDROGEN_WD, "H"), (lt.SEGMENT_HELIUM_WD, "He")):
        on = segment == code
        assert_allclose(weights[:, on].mean(axis=0), 1.0, rtol=1e-9)
        cover = np.asarray(grid[f"cover{key}"])
        outside = (mr[None, on] < cover[:, :1]) | (mr[None, on] > cover[:, 1:])
        assert outside.any() and np.all(weights[:, on][outside] == 0)
        # where every mass has a model the rows follow the prior on mass
        full = np.where(~outside.any(axis=0))[0][0]
        w = np.asarray(grid[f"w{key}"])
        assert_allclose(weights[:, on][:, full], w / w.mean(), rtol=1e-9)


def test_an_ms_star_lands_on_the_same_tLoc_with_and_without_the_qso_rows(qsoLoci):
    """The QSO track takes no model stars and does not shift the stellar segments of the label map."""
    priors = script("make_priors")
    rng = np.random.default_rng(7)
    n = 20000
    stars = pd.DataFrame(
        {
            "FeH": rng.uniform(-2.5, 0.5, n),
            "Mr": rng.uniform(-3.5, 15.0, n),
            "label": rng.choice([1, 2, 3, 4, 7, 8, 9], n),
        }
    )
    placed = []
    for path in qsoLoci[:2]:
        feH, segments, mrMin, mrMax, _ = priors.locusAxes(path)
        placed.append((lt.assignTLocPartition(stars, segments, feH)["tLoc"].to_numpy(), mrMin, mrMax))
    (a, loA, hiA), (b, loB, hiB) = placed
    assert np.array_equal(a, b, equal_nan=True)
    assert (loA, hiA) == (loB, hiB) and loB > lt.QSO_MR
    assert np.isfinite(a).mean() > 0.5
    # and nothing lands on the track
    assert not np.any((b >= lt.QSO_TLOC[0]) & (b <= lt.QSO_TLOC[1]))


def test_the_counts_are_the_table_of_the_paper():
    """Each row of both models adds up to the paper's total to its rounding, and dimming shifts the counts."""
    priors = script("make_priors")
    for table, totals in zip((priors.QSO_PLE, priors.QSO_PLE_LEDE), TABLE7_TOTALS, strict=True):
        assert np.abs(table.sum(axis=1) - np.asarray(totals)).max() <= 1
    # r < 23 over all redshifts is 304 per deg2 in the paper (mean of the two models), and the track keeps
    # all but the lowest redshifts of the first bin
    n, pz = priors.qsoCounts(15.5, 23.0, 0.0)
    assert 0.8 * 304 < n < 304
    assert_allclose(np.trapezoid(pz, priors.QSO_ZFINE), 1.0, rtol=1e-6)
    assert np.all(pz[(priors.QSO_ZFINE < 0.2) | (priors.QSO_ZFINE > 5.0)] == 0)
    assert_allclose(priors.qsoCounts(20.5, 21.5, 0.7)[0], priors.qsoCounts(19.8, 20.8, 0.0)[0], rtol=1e-9)


def test_the_qso_maps_live_in_the_gap_only_the_track_reads(qsoLoci):
    """The gap is bounded by the map points the neighbouring segments read; the maps have unit mass in it."""
    priors = script("make_priors")
    assert priors.qsoGap(qsoLoci[1], compact=True) == pytest.approx((-0.5, 1.6))
    assert priors.qsoGap(qsoLoci[1], compact=False) == pytest.approx((-0.5, 1.65))
    bc = getBayesConstants(lt.HELIUM_TLOC[0])
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    xGrid = np.linspace(bc["FeHmin"], bc["FeHmax"], bc["FeHNpts"])
    yGrid = np.linspace(bc["MrFaint"], bc["MrBright"], bc["MrNpts"])
    gap = priors.qsoGap(qsoLoci[1], compact=True)
    maps = priors.qsoMaps(rGrid, 0.5, 0.2, xGrid, yGrid, gap)
    dx, dy = xGrid[1] - xGrid[0], abs(yGrid[1] - yGrid[0])
    assert_allclose(maps.sum(axis=(1, 2)) * dx * dy, 1.0, rtol=1e-9)
    outside = (yGrid <= gap[0]) | (yGrid >= gap[1])
    assert np.all(maps[:, outside, :] == 0)


def test_the_qso_shape_never_sets_the_peak_of_a_map():
    """In the gap the other maps are zero and the QSO shape is added, scaled to at most their peak."""
    priors = script("make_priors")
    yGrid = np.linspace(17.0, -6.0, 461)
    gap = (-0.5, 1.6)
    inGap = (yGrid > gap[0] + 1e-9) & (yGrid < gap[1] - 1e-9)
    rng = np.random.default_rng(4)
    maps = rng.uniform(0.0, 0.2, (3, yGrid.size, 4))
    shape = np.zeros_like(maps)
    shape[:, inGap, :] = rng.uniform(0.0, 1.0, (3, int(inGap.sum()), 4))
    shape[2] *= 0.01  # a QSO shape lower than the rest is left as it is
    out = priors.withQsoMaps(maps, shape, yGrid, gap)
    assert np.array_equal(out[:, ~inGap, :], maps[:, ~inGap, :])
    assert_allclose(out.max(axis=(1, 2)), maps[:, ~inGap, :].max(axis=(1, 2)))
    assert_allclose(out[2][inGap], shape[2][inGap])
    for k in range(2):
        ratio = out[k][inGap] / shape[k][inGap]
        assert_allclose(ratio, ratio.flat[0])


def test_the_qso_share_comes_off_the_other_populations(monkeypatch):
    """With QSOs the shares of each r bin add to one, and q is the counts against the model stars per deg2."""
    priors = script("make_priors")
    rng = np.random.default_rng(2)
    n = 50000
    cone = pd.DataFrame(
        {
            "rmag": rng.uniform(14, 27, n),
            "label": rng.choice([1, 3, 9], n, p=[0.9, 0.08, 0.02]),
            "FeH": rng.uniform(-2, 0, n),
            "Mr": rng.uniform(0, 14, n),
        }
    )
    monkeypatch.setattr(priors.lsdb, "open_catalog", lambda *a, **k: SimpleNamespace(compute=lambda: cone))
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    track = (np.array([0.0, 14.0]), np.array([-2.0, -0.5]))
    tracks = {lt.SEGMENT_HYDROGEN_WD: track, lt.SEGMENT_HELIUM_WD: (track[0], track[1] - 4.0)}
    args = ("trilegal", 0.0, 0.0, 4.0, rGrid, 0.5, np.array([-2.5, 0.5]), -4.0, 15.0, tracks)
    plain, _ = priors.populationShares(*args)
    withQso, _ = priors.populationShares(*args, qsoAr=0.3)
    assert_allclose(plain.sum(axis=1), 1.0)
    assert_allclose(withQso.sum(axis=1), 1.0)
    q = withQso[:, -1]
    assert_allclose(withQso[:, :-1], plain * (1 - q)[:, None], rtol=1e-12)
    area = 2 * np.pi * (1 - np.cos(np.radians(4.0))) * (180 / np.pi) ** 2
    i = int(np.argmin(np.abs(rGrid - 21.0)))
    stars = np.count_nonzero(np.abs(cone["rmag"].to_numpy() - rGrid[i]) <= 0.5) / area
    nQ = priors.qsoCounts(rGrid[i] - 0.5, rGrid[i] + 0.5, 0.3)[0]
    assert_allclose(q[i], nQ / (nQ + stars), rtol=1e-12)


def test_the_qso_share_counts_the_stars_over_the_area_the_catalogue_covers(monkeypatch, tmp_path):
    """A cone half outside the model catalogue holds half the stars per deg2 of cone, not of covered sky."""
    import cdshealpix
    from astropy.io import fits

    priors = script("make_priors")
    order = 9
    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.arange(12 * 4**order), order)
    counts = np.where(lat.deg > 0, 7, 0).astype(np.int64)
    column = fits.Column(name="VALUE", format="K", array=counts)
    hdu = fits.BinTableHDU.from_columns([column])
    hdu.header.update(ORDERING="NESTED", NSIDE=2**order)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(tmp_path / "point_map.fits")
    covered = priors.catalogueCoverage(tmp_path)
    assert covered.size == 12 * 4**priors.COVERAGE_ORDER and 0.49 < covered.mean() < 0.51
    fraction = priors.coneCoverage(covered, 0.0, 0.0, 4.0)
    assert 0.45 < fraction < 0.55
    rng = np.random.default_rng(4)
    n = 20000
    cone = pd.DataFrame(
        {"rmag": rng.uniform(14, 27, n), "label": 1, "FeH": -1.0, "Mr": rng.uniform(0, 10, n)}
    )
    monkeypatch.setattr(priors.lsdb, "open_catalog", lambda *a, **k: SimpleNamespace(compute=lambda: cone))
    bc = getBayesConstants()
    rGrid = np.linspace(bc["rmagMin"], bc["rmagMax"], bc["rmagNsteps"])
    track = (np.array([8.5, 14.5]), np.array([-2.0, -0.5]))
    tracks = {lt.SEGMENT_HYDROGEN_WD: track, lt.SEGMENT_HELIUM_WD: (track[0], track[1] - 4.0)}
    args = ("trilegal", 0.0, 0.0, 4.0, rGrid, 0.5, np.array([-2.5, 0.5]), -4.0, 15.0, tracks)
    shares, _ = priors.populationShares(*args, qsoAr=0.0, covered=covered)
    whole, _ = priors.populationShares(*args, qsoAr=0.0)
    area = 2 * np.pi * (1 - np.cos(np.radians(4.0))) * (180 / np.pi) ** 2
    i = int(np.argmin(np.abs(rGrid - 22.0)))
    w = 2.0  # no white dwarfs in the cone, so the r window widens to its widest
    stars = np.count_nonzero(np.abs(cone["rmag"].to_numpy() - rGrid[i]) <= w)
    nQ = priors.qsoCounts(rGrid[i] - w, rGrid[i] + w, 0.0)[0]
    assert_allclose(shares[i, -1], nQ / (nQ + stars / (area * fraction)), rtol=1e-12)
    assert_allclose(whole[i, -1], nQ / (nQ + stars / area), rtol=1e-12)


def test_the_sfd_lookup_reads_the_projection_of_the_map_headers(tmp_path):
    """The poles are the centres of the two maps and b = 0 at l = 0 is SCALE pixels out along x."""
    from astropy.io import fits

    priors = script("make_priors")
    size, scale = 65, 30
    for name, n in (("ngp", 1), ("sgp", -1)):
        y, x = np.mgrid[:size, :size]
        data = (x + 1000 * y + (0 if n == 1 else 5e5)).astype(np.float32)
        header = fits.Header({"CRPIX1": 33.0, "CRPIX2": 33.0, "LAM_SCAL": scale, "LAM_NSGP": n})
        fits.PrimaryHDU(data, header).writeto(tmp_path / f"SFD_dust_4096_{name}.fits")
    ar = priors.sfdExtinction([0.0, 0.0, 0.0, 90.0], [90.0, -90.0, 0.0, 30.0], tmp_path, arPerEbv=1.0)
    assert ar[0] == 32 + 1000 * 32 and ar[1] == 32 + 1000 * 32 + 5e5
    assert ar[2] == 32 + scale + 1000 * 32
    rho = np.sqrt(1 - np.sin(np.radians(30.0))) * scale
    assert ar[3] == 32 + 1000 * (32 - round(rho))
