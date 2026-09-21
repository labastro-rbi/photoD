"""Shapes of the extinction along the line of sight, A_r(mu) / A_r(total), from 3D dust maps.

The fit needs the *shape* only: each star scales it by its own A_r from the 2D dust map, so the band and the
calibration of the 3D map cancel and maps on different scales can be mixed freely. Curves are tabulated on a
HEALPix grid (nested, equatorial) and stored with an index that maps a HEALPix pixel to a row, -1 where no map
has data. Stars in those pixels keep the flat A_r prior.

  python scripts/make_dust_curves.py --out dust_curves.npz
  python scripts/make_dust_curves.py --footprint .../object_lc/skymap.6.fits --nside 128 --out dust_dp2.npz

No single map covers the sky, so they are tried in order and the first one with data wins each sightline:

  marshall   |l| < 100, |b| < 10, to 10 kpc, any declination
  chen2018   the Galactic plane including the anticentre, to 6 kpc
  bayestar   three quarters of the sky, dec > -30, to 60 kpc
  edenhofer  all sky but only to 2 kpc, so it is used above |b| = 10 where the dust is all nearby

The deep maps come first because a curve has to reach past the stars: a map that stops short would put the
last of the extinction at its own edge rather than where the dust is. That is why the shallow all-sky map is
kept out of the plane, where sightlines carry dust well beyond its range.
"""

import argparse

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

MU = np.arange(4.0, 16.01, 0.25)
# name: (highest |b| it is used at, lowest |b| it is used at, A_r per unit of the map, reaches past the disc)
# The conversion matters only for the total column, which is written out so that a run can bound a star's
# extinction by a measured column instead of by the whole Galaxy. A map that stops short of the far side of
# the disc cannot bound anything, so its total is not written.
MAPS = {
    "marshall": (10.0, 0.0, 7.21, True),  # A_r / A_Ks for R_V = 3.1
    "chen2018": (12.0, 0.0, 1.0, True),  # already an r band extinction
    "bayestar": (90.0, 0.0, 2.483, True),  # A_r / E, Green et al. (2019)
    "edenhofer": (90.0, 10.0, 2.35, False),  # A_r / E of Zhang, Green and Rix (2023); stops at 2 kpc
}


def mapQuery(name):
    """A callable returning the extinction, up to a constant, at SkyCoords carrying distances."""
    if name == "marshall":
        from dustmaps.marshall import MarshallQuery

        query = MarshallQuery()
        return lambda c: np.asarray(query(c), dtype=float)
    if name == "chen2018":
        from dustmaps.chen2018 import Chen2018Query

        query = Chen2018Query()
        return lambda c: np.asarray(query(c), dtype=float)
    if name == "bayestar":
        from dustmaps.bayestar import BayestarQuery

        query = BayestarQuery(version="bayestar2019")
        return lambda c: np.asarray(query(c), dtype=float)
    if name == "edenhofer":
        from dustmaps.edenhofer2023 import Edenhofer2023Query

        try:
            query = Edenhofer2023Query(integrated=True, flavor="less_data_but_2kpc")
        except Exception:
            query = Edenhofer2023Query(integrated=True)
        return lambda c: np.asarray(query(c), dtype=float)
    raise ValueError(f"unknown map {name}")


def footprintPixels(path, nside):
    """Pixels of the given nside that the footprint covers, from the sky map of the object catalog."""
    if not path:
        return np.arange(12 * nside**2, dtype=np.int64)
    with fits.open(path) as hdus:
        columns = [c.name for c in hdus[1].columns]
        if "PIXEL" in columns:
            pixel = np.asarray(hdus[1].data.field("PIXEL")).ravel().astype(np.int64)
            value = np.asarray(hdus[1].data.field("VALUE")).ravel().astype(float)
            mapNside = int(hdus[1].header["NSIDE"])
        else:
            value = np.asarray(hdus[1].data.field(0)).ravel().astype(float)
            pixel = np.arange(value.size, dtype=np.int64)
            mapNside = int(round(np.sqrt(value.size / 12)))
    pixel = pixel[value > 0]
    if mapNside >= nside:
        return np.unique(pixel >> (2 * int(round(np.log2(mapNside / nside)))))
    spread = int((nside // mapNside) ** 2)
    return np.unique(np.concatenate([pixel * spread + i for i in range(spread)]))


def pixelCentres(pixels, nside):
    """Equatorial coordinates of the centres of the given nested HEALPix pixels."""
    import cdshealpix

    lon, lat = cdshealpix.nested.healpix_to_lonlat(pixels, int(np.log2(nside)))
    return SkyCoord(ra=lon, dec=lat, frame="icrs")


def curveShapes(query, coords, mu=MU, chunk=2000):
    """A_r(mu) / A_r(total) for each coordinate, and the total in the map's own units.

    The map is queried for every sightline and distance at once, which is what makes an all-sky grid
    affordable. A curve is kept when the map has at least two finite samples and any extinction at all; it is
    forced to increase with distance, because a cumulative column cannot fall.
    """
    distance = 10 ** (mu / 5 - 2)  # kpc
    galactic = coords.galactic
    lon, lat = galactic.l.deg, galactic.b.deg
    shapes = np.zeros((len(coords), mu.size), dtype=np.float32)
    totals = np.zeros(len(coords), dtype=np.float32)
    for start in range(0, len(coords), chunk):
        stop = min(start + chunk, len(coords))
        n = stop - start
        grid = SkyCoord(
            l=np.repeat(lon[start:stop], mu.size) * u.deg,
            b=np.repeat(lat[start:stop], mu.size) * u.deg,
            distance=np.tile(distance, n) * u.kpc,
            frame="galactic",
        )
        try:
            values = np.atleast_1d(query(grid)).astype(float).reshape(n, mu.size)
        except Exception:
            continue
        for i, row in enumerate(values):
            good = np.isfinite(row)
            if good.sum() < 2 or not np.any(row[good] > 0):
                continue
            filled = np.maximum.accumulate(np.maximum(np.interp(mu, mu[good], row[good]), 0))
            if filled[-1] > 0:
                shapes[start + i] = filled / filled[-1]
                totals[start + i] = filled[-1]
    return shapes, totals


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--maps",
        default="marshall,chen2018,bayestar,edenhofer",
        help="maps to try, deepest first; the first with data wins each sightline",
    )
    ap.add_argument("--nside", type=int, default=128, help="HEALPix nside of the grid (128 is 27 arcmin)")
    ap.add_argument("--footprint", default="", help="sky map of the object catalog, to build only its pixels")
    ap.add_argument("--bmax", type=float, default=90.0, help="highest |b| to tabulate")
    ap.add_argument("--bmin", type=float, default=0.0, help="lowest |b| to tabulate")
    ap.add_argument("--data-dir", default="", help="dustmaps data directory")
    ap.add_argument("--out", default="dust_curves.npz")
    args = ap.parse_args()

    if args.data_dir:
        from dustmaps.config import config

        config["data_dir"] = args.data_dir

    pixels = footprintPixels(args.footprint, args.nside)
    coords = pixelCentres(pixels, args.nside)
    latitude = np.abs(coords.galactic.b.deg)
    wanted = (latitude <= args.bmax) & (latitude >= args.bmin)
    pixels, coords, latitude = pixels[wanted], coords[wanted], latitude[wanted]
    print(f"{pixels.size} pixels of nside {args.nside} between |b| = {args.bmin} and {args.bmax}")

    shapes = np.zeros((pixels.size, MU.size), dtype=np.float32)
    total = np.zeros(pixels.size, dtype=np.float32)
    source = np.zeros(pixels.size, dtype=np.int8) - 1
    names = [n.strip() for n in args.maps.split(",") if n.strip()]
    for number, name in enumerate(names):
        high, low, perUnit, deep = MAPS.get(name, (90.0, 0.0, 1.0, False))
        todo = np.where((source < 0) & (latitude <= high) & (latitude >= low))[0]
        if not todo.size:
            continue
        try:
            query = mapQuery(name)
        except Exception as error:  # a map whose data is not on this machine simply does not contribute
            print(f"{name:>10}: unavailable ({type(error).__name__})")
            continue
        found, columns = curveShapes(query, coords[todo])
        covered = found[:, -1] > 0
        shapes[todo[covered]] = found[covered]
        source[todo[covered]] = number
        if deep:
            total[todo[covered]] = perUnit * columns[covered]
        half = np.array([np.interp(0.5, s, MU) for s in found[covered]]) if covered.any() else np.zeros(1)
        print(
            f"{name:>10}: {int(covered.sum())} of {todo.size} tried, half the extinction by mu "
            f"{np.median(half):.1f}, {int((source < 0).sum())} sightlines still uncovered"
        )

    covered = source >= 0
    index = np.full(12 * args.nside**2, -1, dtype=np.int32)
    index[pixels[covered]] = np.arange(int(covered.sum()), dtype=np.int32)
    np.savez_compressed(
        args.out,
        shapes=shapes[covered],
        total=total[covered],
        mu=MU,
        index=index,
        nside=args.nside,
        maps=np.array(names),
        source=source[covered],
    )
    bounded = total[covered] > 0
    print(
        f"{int(bounded.sum())} sightlines carry a measured total column, median A_r "
        f"{np.median(total[covered][bounded]) if bounded.any() else 0:.2f}"
    )
    area = 4 * np.pi * (180 / np.pi) ** 2 / (12 * args.nside**2)
    print(
        f"{args.out}: {int(covered.sum())} of {pixels.size} sightlines covered "
        f"({100 * covered.mean():.1f} %, {covered.sum() * area:.0f} deg2)"
    )


if __name__ == "__main__":
    main()
