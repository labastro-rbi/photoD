"""Shapes of the extinction along the line of sight, A_r(mu) / A_r(total), from a 3D dust map.

The fit needs the *shape* only: each star scales it by its own A_r from the 2D dust map, so the band and the
calibration of the 3D map cancel and any of them can be used. Curves are tabulated on a HEALPix grid (nested,
equatorial) and stored with an index that maps a HEALPix pixel to a row, -1 where the map has no data. Stars
in those pixels keep the flat A_r prior.

  python scripts/make_dust_curves.py --map marshall --out dust_marshall.npz
  python scripts/make_dust_curves.py --map chen2018 --bmax 10 --nside 128 --out dust_chen.npz
  python scripts/make_dust_curves.py --map bayestar --bmin 10 --out dust_bayestar.npz

The maps (dustmaps package, which downloads them on first use) differ in where they have data:
  marshall   |l| < 100, |b| < 10, to ~10 kpc, any declination
  chen2018   the Galactic plane including the anticentre, to ~6 kpc
  bayestar   three quarters of the sky, dec > -30, to ~60 kpc
A 3D prior is worth having where stars sit inside the dust, which in practice means |b| < 10; above that the
extinction is nearly all in front of the stars and the flat prior loses nothing.
"""

import argparse

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

MU = np.arange(4.0, 16.01, 0.25)


def mapQuery(name):
    """A callable returning A_r (up to a constant) at SkyCoords with distances, for the named 3D dust map."""
    from dustmaps import bayestar, chen2018, marshall

    if name == "marshall":
        q = marshall.MarshallQuery()
        return lambda c: 7.21 * np.asarray(q(c), dtype=float)  # A_r / A_Ks for R_V = 3.1
    if name == "chen2018":
        q = chen2018.Chen2018Query()
        return lambda c: np.asarray(q(c), dtype=float)
    if name == "bayestar":
        q = bayestar.BayestarQuery(version="bayestar2019")
        return lambda c: 2.483 * np.asarray(q(c), dtype=float)  # A_r / E, Green et al. (2019)
    raise ValueError(f"unknown map {name}")


def pixelCentres(nside):
    """Equatorial coordinates of the centres of all nested HEALPix pixels of this nside."""
    import cdshealpix

    depth = int(np.log2(nside))
    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.arange(12 * nside**2), depth)
    return SkyCoord(ra=lon, dec=lat, frame="icrs")


def curveShapes(query, coords, mu=MU):
    """A_r(mu) / A_r(total) for each coordinate; rows of zeros where the map has no data."""
    dist = 10 ** (mu / 5 - 2)  # kpc
    shapes = np.zeros((len(coords), mu.size), dtype=np.float32)
    for i, c in enumerate(coords):
        grid = SkyCoord(
            l=np.full(mu.size, c.galactic.l.deg) * u.deg,
            b=np.full(mu.size, c.galactic.b.deg) * u.deg,
            distance=dist * u.kpc,
            frame="galactic",
        )
        a = np.atleast_1d(query(grid)).astype(float)
        good = np.isfinite(a)
        if good.sum() < 2 or not np.any(a[good] > 0):
            continue
        a = np.maximum.accumulate(np.maximum(np.interp(mu, mu[good], a[good]), 0))
        shapes[i] = a / a[-1]
    return shapes


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="marshall", choices=("marshall", "chen2018", "bayestar"))
    ap.add_argument(
        "--nside", type=int, default=128, help="HEALPix nside of the curve grid (128 is 27 arcmin)"
    )
    ap.add_argument("--bmax", type=float, default=12.0, help="highest |b| to tabulate")
    ap.add_argument("--bmin", type=float, default=0.0, help="lowest |b| to tabulate")
    ap.add_argument("--data-dir", default="", help="dustmaps data directory")
    ap.add_argument("--out", default="dust_curves.npz")
    args = ap.parse_args()

    if args.data_dir:
        from dustmaps.config import config

        config["data_dir"] = args.data_dir
    coords = pixelCentres(args.nside)
    b = np.abs(coords.galactic.b.deg)
    wanted = np.where((b <= args.bmax) & (b >= args.bmin))[0]
    print(f"{len(wanted)} HEALPix pixels of nside {args.nside} with {args.bmin} < |b| < {args.bmax}")
    shapes = curveShapes(mapQuery(args.map), coords[wanted])
    covered = shapes[:, -1] > 0
    index = np.full(12 * args.nside**2, -1, dtype=np.int32)
    index[wanted[covered]] = np.arange(int(covered.sum()), dtype=np.int32)
    np.savez(args.out, shapes=shapes[covered], mu=MU, index=index, nside=args.nside, dustMap=args.map)
    half = np.array([np.interp(0.5, s, MU) for s in shapes[covered]]) if covered.any() else np.array([np.nan])
    print(
        f"{args.out}: {int(covered.sum())} of {len(wanted)} pixels covered by {args.map}; "
        f"half of the extinction is reached by a median distance modulus of {np.median(half):.1f}"
    )


if __name__ == "__main__":
    main()
