"""Dust curves for run_dp2.py --dust-curves of an exponential dust layer: the fraction of the 2D column in
front of a star at distance d and Galactic latitude b,

    f(d, b) = 1 - exp(-d sin|b| / h),

for a layer of scale height h with the Sun in its mid-plane. It is an assumed geometry times the run's 2D
column, not a 3D measurement: with --dust-curve-mode band it turns the A_r box into a band whose edges follow
the dust in front of each locus point, and far behind the layer it is the box again.

  python scripts/make_expo_dust_curves.py --out expo_h125.npz

h = 125 pc is Marshall et al. (2006); the literature range is 119 pc (Jones et al. 2011) to 134 pc (Drimmel &
Spergel 2001). The file has the format run_dp2.readCurves reads: shapes (rows by mu), mu, index (the row of
every nested HEALPix pixel at nside), nside, and no total, so the box top stays the 2D map. The rows are |b|
bins of --b-step degrees, the pixel centre's |b| taken to the nearest; |b| = 0 is held at half a step, where
the layer would otherwise put no dust in front of anything.
"""

import argparse
import json

import numpy as np

SCALE_HEIGHT = 125.0  # pc, Marshall et al. 2006


def curves(scaleHeight=SCALE_HEIGHT, nside=128, muMin=2.0, muMax=20.0, muStep=0.1, bStep=0.1):
    """The shapes, mu grid, pixel index and the |b| of each row."""
    import astropy.units as u
    import cdshealpix
    from astropy.coordinates import SkyCoord

    order = int(round(np.log2(nside)))
    if 1 << order != nside:
        raise SystemExit(f"nside {nside} is not a power of two")
    lon, lat = cdshealpix.nested.healpix_to_lonlat(np.arange(12 * nside**2), order)
    absb = np.abs(SkyCoord(ra=lon.to(u.deg), dec=lat.to(u.deg), frame="icrs").galactic.b.deg)
    mu = np.round(np.arange(muMin, muMax + 0.5 * muStep, muStep), 6)
    rows = np.arange(0.0, 90.0 + 0.5 * bStep, bStep)
    centre = np.clip(rows, 0.5 * bStep, None)
    d = 10 ** (mu / 5 + 1)  # pc
    shapes = 1 - np.exp(-d[None, :] * np.sin(np.radians(centre))[:, None] / scaleHeight)
    index = np.clip(np.round(absb / bStep).astype(np.int32), 0, rows.size - 1)
    return shapes.astype(np.float32), mu, index, rows


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale-height", type=float, default=SCALE_HEIGHT, help="pc")
    ap.add_argument("--nside", type=int, default=128)
    ap.add_argument("--mu-min", type=float, default=2.0)
    ap.add_argument("--mu-max", type=float, default=20.0)
    ap.add_argument("--mu-step", type=float, default=0.1)
    ap.add_argument("--b-step", type=float, default=0.1, help="degrees")
    args = ap.parse_args()
    shapes, mu, index, rows = curves(
        args.scale_height, args.nside, args.mu_min, args.mu_max, args.mu_step, args.b_step
    )
    note = {
        "model": "exponential dust layer, f = 1 - exp(-d sin|b| / h)",
        "scaleHeight": args.scale_height,
        "bStep": args.b_step,
        "source": "Marshall+2006 (h ~ 125 pc); range 119 (Jones+2011) - 134 pc (Drimmel & Spergel 2001)",
    }
    np.savez(
        args.out,
        shapes=shapes,
        mu=mu,
        index=index,
        nside=np.int64(args.nside),
        note=np.array(json.dumps(note)),
    )
    for b in (10, 20, 40):
        k = int(round(b / args.b_step))
        at = [float(np.interp(5 * np.log10(x) - 5, mu, shapes[k])) for x in (100, 300, 1000)]
        print(f"|b| {b}: fraction in front at 100, 300, 1000 pc {np.round(at, 3).tolist()}")
    print(f"written {args.out}: {shapes.shape[0]} |b| rows by {mu.size} mu, nside {args.nside}")


if __name__ == "__main__":
    main()
