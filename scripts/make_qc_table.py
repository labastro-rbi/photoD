"""The QC table of run_dp2.py --qc-table: the 99th percentile of chi2min for the class a star is given, under
the fit's own model, for every colour pattern and |b| class (photod.quality).

The null is a Monte Carlo of the final model, fitted by the production fit itself, in three steps:

  python scripts/make_qc_table.py draw --sim sim_full.parquet --errmodel errmodel.npz --locus L --priors P \\
         [--dust-curves C --dust-curve-mode band] --out mc_hats
  python scripts/run_dp2.py --catalog mc_hats --ar-column Ar_map --selection recipe ...  (the run's options)
  python scripts/make_qc_table.py table --run <run_dp2 output>/<name> --out qc_table.csv

draw puts stars where the TRILEGAL sim has them (its positions, r and A_r column, the same number in each |b|
class), and draws each one's locus point from the run's prior in its pixel and r bin, its A_r flat over that
point's allowed range (the box, or the band, bayes.arBandEdges), and its colours as the locus colours reddened
by A_r with the model's own error added, the colour floor on every colour. The photometry is then that of the
sim: the flux error against magnitude and the band coverage of errmodel.npz in the star's pixel, noise in
flux, a colour kept where both bands have S/N > 3, r S/N > 10 and g, i S/N > 3, 16.5 < r < 25. No survey data
enter.

The two inputs of draw are not in this repository, and nothing in it writes them, so the table cannot be
rebuilt from the repository alone:

  sim_full.parquet  a TRILEGAL simulation of DP2 fields, a row per star, of which draw reads ra, dec, p5 (the
                    star's HEALPix pixel at order 5), Ar_map (the dust-map A_r at its position) and rmag_obs
                    (its r magnitude)
  errmodel.npz      mbins, magnitude bins; sig_<p5>_<band>, the flux error in each bin (zero point 25); and
                    cover_<p5>_<band>, the fraction of stars the band measures, for every pixel of the sim

The class is drawn with equal odds rather than with the prior's, so that the rare classes get enough stars,
and each star carries the weight prior / proposal of its class: the table is of the model's prior mixture.
table takes the claimed class (the most probable) of every fitted star and the weighted 99th percentile of
its chi2min in that class, by class, pattern and |b| class. A cell with fewer than --min-eff effective stars
takes the value of the first level of (class, pattern), (class, colour count, |b|), (class, colour count)
that has them, and the level is written. chi2.isf(alpha, max(n - 2, 1)) is written beside it as a check only.

data/qc_table_dp2_inclusive_wd.csv, the table of the DP2 white dwarf and QSO catalogue, is a draw of
--n 3000000 --seed 20261025 with --floor 0.03 --ar-max 8.0 and the run's locus, prior maps and dust curves,
fitted with the run's --locus, --priors, --floor, --dust-curves, --dust-curve-mode band and
--estimate-class MS, and --selection recipe --ar-column Ar_map --ar-max 8.0 (scripts/run_dp2.py). The sim
was 1,474,612 stars in five pixels (sim_full.parquet md5 f66a656df5ec12c720305fc908ccedd8, errmodel.npz md5
3dfac4f11171bc8b3806a923b946140e). The table's first line names the directory the Monte Carlo was fitted in,
on the machine it was made on.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from photod.bayes import MISSING_COLOR_ERR, arBandEdges
from photod.priors import getPriorMapIndex
from photod.quality import (
    LATITUDE_EDGES,
    QC_ALPHA,
    QC_CLASSES,
    claimedClass,
    colourPattern,
    latitudeClass,
    patternColours,
)

COLORS = ("ug", "gr", "ri", "iz", "zy")
BANDS = "ugrizy"
MIN_EFF = 1000.0  # effective stars for a 99th percentile: about ten above it
SEED = 20261025
R_RANGE = (16.5, 25.0)  # the sim's selection in r


def runScript():
    """scripts/run_dp2.py as a module, for the fit setup it builds."""
    path = Path(__file__).resolve().parent / "run_dp2.py"
    spec = importlib.util.spec_from_file_location("run_dp2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_dp2"] = module
    spec.loader.exec_module(module)
    return module


def fitSetup(run, args):
    """The GlobalParams of the run the table is for, as run_dp2 builds them."""
    masses = run.checkPriorLocus(args.priors, args.locus)
    curves = run.readCurves(args.dust_curves) if args.dust_curves else None
    gp = run.globalParameters(
        args.floor,
        True,
        curves,
        args.ar_max,
        args.locus,
        masses,
        args.dust_curve_mode,
        "",
        None,
    )
    return gp, curves


def drawPoints(rng, mass, classMasks, n):
    """n locus points from the prior masses of one pixel and r bin, the class drawn with equal odds.

    Returns the points, their class index and weight, the prior probability of the class over 1 / number of
    classes with any mass.
    """
    present = [k for k, m in enumerate(classMasks) if mass[m].sum() > 0]
    total = mass.sum()
    cls = rng.choice(present, size=n)
    points = np.empty(n, dtype=np.int64)
    weight = np.empty(n)
    for k in present:
        take = cls == k
        p = np.where(classMasks[k], mass, 0.0)
        points[take] = rng.choice(mass.size, size=take.sum(), p=p / p.sum())
        weight[take] = p.sum() / total * len(present)
    return points, cls, weight


def bandEdges(gp, points, rmag, curveIndex, arMap):
    """The allowed A_r range of each drawn point, by the fit's arBandEdges, cut to the box and the grid."""
    arMax = np.minimum(gp.ArPriorScale * arMap + gp.ArPriorOffset, gp.Ar1d[-1])
    if gp.bandFlat is None:
        return np.zeros_like(arMax), arMax
    import jax

    edges = jax.jit(jax.vmap(arBandEdges, in_axes=(0, 0, 0, None, 0, None, None, 0, 0)))
    lo, hi = [], []
    for part in np.array_split(np.arange(points.size), max(1, points.size // 200_000)):
        i = points[part]
        a, b = edges(
            gp.MrTrueFlat[i][:, None],
            rmag[part],
            gp.ArCurves[curveIndex[part]],
            gp.ArCurveMu,
            arMap[part],
            gp.ArPriorScale,
            gp.ArPriorOffset,
            gp.bandFlat[i][:, None],
            gp.extragalacticFlat[i][:, None],
        )
        lo.append(np.asarray(a)[:, 0])
        hi.append(np.asarray(b)[:, 0])
    lo, hi = np.concatenate(lo), np.minimum(np.concatenate(hi), arMax)
    return np.minimum(lo, hi), hi


def photometry(rng, mags, pixel, errmodel):
    """Observed magnitudes and S/N per band under the sim's error model of each star's pixel."""
    bins = errmodel["mbins"]
    obs, snr = {}, {}
    for b in BANDS:
        sf = np.empty(pixel.size)
        cover = np.ones(pixel.size, bool)
        for p in np.unique(pixel):
            here = pixel == p
            k = np.clip(np.searchsorted(bins, mags[b][here]) - 1, 0, bins.size - 2)
            sf[here] = errmodel[f"sig_{p}_{b}"][k]
            cover[here] = rng.random(here.sum()) < float(errmodel[f"cover_{p}_{b}"])
        flux = 10 ** (-0.4 * (mags[b] - 25.0))
        fo = flux + rng.normal(size=pixel.size) * sf
        snr[b] = np.where(cover, fo / sf, -1.0)
        obs[b] = np.where(fo > 0, 25.0 - 2.5 * np.log10(np.where(fo > 0, fo, 1.0)), np.nan)
    return obs, snr


def draw(args):
    """Write the Monte Carlo catalogue."""
    run = runScript()
    rng = np.random.default_rng(args.seed)
    gp, curves = fitSetup(run, args)
    nFeH, nMr = gp.FeH1d.size, gp.Mr1d.size
    classMasks = [np.broadcast_to(gp.classAxisMasks[name], (nFeH, nMr)).ravel() for name in QC_CLASSES]
    sim = pd.read_parquet(args.sim, columns=["ra", "dec", "p5", "Ar_map", "rmag_obs"])
    errmodel = dict(np.load(args.errmodel))
    lat = latitudeClass(sim.ra, sim.dec)
    classes = np.unique(lat)
    rows = np.concatenate([rng.choice(np.where(lat == k)[0], args.n // classes.size) for k in classes])
    stars = sim.iloc[rows].reset_index(drop=True)
    rTrue = stars.rmag_obs.to_numpy(float)
    arMap = stars.Ar_map.to_numpy(float)
    curveIndex = (
        run.withDust(stars[["ra", "dec"]].copy(), curves)["dustIndex"].to_numpy()
        if curves
        else np.zeros(len(stars), dtype=np.int32)
    )
    priors = run.loadPriors(args.priors)
    priorRow = priors["index"][run.priorPixels(stars.ra, stars.dec, int(priors["order"]))]
    if (priorRow < 0).any():
        raise SystemExit("the sim has stars where the prior file has no map")
    rBin = getPriorMapIndex(rTrue)
    points = np.empty(len(stars), dtype=np.int64)
    cls = np.empty(len(stars), dtype=np.int64)
    weight = np.empty(len(stars))
    weights = gp.locusValid * gp.priorWeights
    for row in np.unique(priorRow):
        grid = run.locusPrior(priors, row, gp)
        for k in np.unique(rBin[priorRow == row]):
            take = np.where((priorRow == row) & (rBin == k))[0]
            mass = np.asarray(grid[k], dtype=float) * weights
            points[take], cls[take], weight[take] = drawPoints(rng, mass, classMasks, take.size)
    lo, hi = bandEdges(gp, points, rTrue, curveIndex, arMap)
    Ar = lo + rng.random(lo.size) * (hi - lo)
    colours = gp.locusColors2d[points] + Ar[:, None] * gp.reddVector
    sigma = np.full(colours.shape, gp.colorErrFloor)
    colours = colours + rng.normal(size=colours.shape) * sigma
    mags = {"r": rTrue}
    mags["g"] = mags["r"] + colours[:, 1]
    mags["u"] = mags["g"] + colours[:, 0]
    mags["i"] = mags["r"] - colours[:, 2]
    mags["z"] = mags["i"] - colours[:, 3]
    mags["y"] = mags["z"] - colours[:, 4]
    obs, snr = photometry(rng, mags, stars.p5.to_numpy(), errmodel)
    with np.errstate(invalid="ignore"):
        keep = (
            (snr["r"] > 10)
            & (snr["g"] > 3)
            & (snr["i"] > 3)
            & (obs["r"] > R_RANGE[0])
            & (obs["r"] < R_RANGE[1])
        )
    out = pd.DataFrame(
        {
            "objectId": np.arange(len(stars), dtype=np.int64),
            "ra": stars.ra.to_numpy(float),
            "dec": stars.dec.to_numpy(float),
            "rmag": obs["r"],
            "rmagErr": 1.0857 / snr["r"],
            "Ar_map": arMap,
        }
    )
    for c in COLORS:
        a, b = c
        ok = (snr[a] > 3) & (snr[b] > 3)
        out[c] = np.where(ok, obs[a] - obs[b], 0.0)
        out[c + "Err"] = np.where(ok, np.hypot(1.0857 / snr[a], 1.0857 / snr[b]), 9.99)
    truth = {
        "t_class": cls,
        "t_point": points,
        "t_Ar": Ar,
        "t_w": weight,
        "t_lat": lat[rows],
        "t_rtrue": rTrue,
    }
    for name, value in truth.items():
        out[name] = value
    out = out[keep].reset_index(drop=True)
    print(f"drawn {len(stars):,}, selected {len(out):,}; by class drawn / selected:")
    for k, name in enumerate(QC_CLASSES):
        print(f"  {name}: {int((cls == k).sum()):,} / {int((out.t_class == k).sum()):,}")
    # the catalogue is written by a process of its own, which has not run JAX: written from this one, after
    # the band edges ran on the GPU, lsdb's writer crashed the interpreter on 3M rows
    rows = Path(f"{args.out}.parquet")
    out.to_parquet(rows, index=False)
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    subprocess.run(
        [sys.executable, __file__, "hats", "--rows", str(rows), "--out", args.out], check=True, env=env
    )
    rows.unlink()


def hats(args):
    """Write a table of rows as a HATS catalogue."""
    import lsdb

    out = pd.read_parquet(args.rows)
    cat = lsdb.from_dataframe(
        out, ra_column="ra", dec_column="dec", partition_rows=150_000, margin_threshold=None
    )
    cat.write_catalog(args.out, catalog_name=Path(args.out).name, as_collection=False, overwrite=True)
    print(f"written {args.out}: {len(out):,} rows")


def weightedQuantile(x, w, q):
    """The smallest x whose weighted cumulative share reaches q."""
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cum, q * cum[-1])])


def thresholds(frame, alpha=QC_ALPHA, minEff=MIN_EFF):
    """The QC table of a fitted Monte Carlo: claimed class, pattern and |b| of every star, and its weight."""
    from scipy import stats

    nLat = len(LATITUDE_EDGES) + 1
    frame = frame[frame.claimed >= 0]
    levels = [
        ("cell", ["claimed", "lat", "pattern"]),
        ("pattern", ["claimed", "pattern"]),
        ("nColors", ["claimed", "nColors", "lat"]),
        ("nColorsAll", ["claimed", "nColors"]),
    ]
    found = []
    for level, keys in levels:
        table = {}
        for key, g in frame.groupby(keys):
            w = g.w.to_numpy()
            nEff = w.sum() ** 2 / np.sum(w**2)
            if nEff >= minEff:
                table[key if isinstance(key, tuple) else (key,)] = (
                    weightedQuantile(g.chi2.to_numpy(), w, 1 - alpha),
                    len(g),
                    nEff,
                )
        found.append((level, keys, table))
    rows = []
    for k, name in enumerate(QC_CLASSES):
        for lat in range(nLat):
            for pattern in range(1, 1 << len(COLORS)):
                n = bin(pattern).count("1")
                values = {"claimed": k, "lat": lat, "pattern": pattern, "nColors": n}
                for level, keys, table in found:
                    hit = table.get(tuple(values[c] for c in keys))
                    if hit is not None:
                        chi2, count, nEff = hit
                        rows.append(
                            {
                                "class": name,
                                "latitude": lat,
                                "pattern": pattern,
                                "colours": patternColours(pattern, COLORS),
                                "nColors": n,
                                "chi2": round(chi2, 4),
                                "level": level,
                                "n": count,
                                "nEff": round(nEff, 1),
                                "chi2Analytic": round(float(stats.chi2.isf(alpha, max(n - 2, 1))), 4),
                            }
                        )
                        break
    return pd.DataFrame(rows)


def readRun(path):
    """The columns of a fitted Monte Carlo the table needs, from every partition of the run."""
    import pyarrow.parquet as pq

    names = ["ra", "dec", "t_w", *(f"p_{c}" for c in QC_CLASSES), *(f"chi2min{c}" for c in QC_CLASSES)]
    names += [c for c in COLORS] + [c + "Err" for c in COLORS]
    files = sorted(Path(path).glob("dataset/Norder=*/Dir=*/Npix=*.parquet"))
    if not files:
        raise SystemExit(f"{path} holds no partitions")
    return pd.concat([pq.read_table(f, columns=names).to_pandas() for f in files], ignore_index=True)


def table(args):
    """Tabulate the thresholds of a fitted Monte Carlo."""
    t = readRun(args.run)
    errors = t[[c + "Err" for c in COLORS]].to_numpy(float)
    values = t[list(COLORS)].to_numpy(float)
    missing = ~(errors <= MISSING_COLOR_ERR) | ~np.isfinite(values)
    pattern = colourPattern(np.where(missing, np.inf, errors))
    claimed = claimedClass(t)
    chi2 = np.stack([t[f"chi2min{c}"].to_numpy(float) for c in QC_CLASSES], axis=1)
    own = np.where(claimed >= 0, chi2[np.arange(len(t)), np.maximum(claimed, 0)], np.nan)
    frame = pd.DataFrame(
        {
            "claimed": np.where(np.isfinite(own), claimed, -1),
            "lat": latitudeClass(t.ra, t.dec),
            "pattern": pattern,
            "nColors": (~missing).sum(axis=1),
            "chi2": own,
            "w": t.t_w.to_numpy(float),
        }
    )
    frame = frame[frame.pattern > 0]
    out = thresholds(frame, args.alpha, args.min_eff)
    with open(args.out, "w") as f:
        f.write(
            f"# QC table (make_qc_table.py): weighted {100 * (1 - args.alpha):g}th percentile of chi2min of "
            "the "
            f"claimed class, by class, |b| class (edges {list(LATITUDE_EDGES)}) and colour pattern (bit k = "
            f"{','.join(COLORS)}[k] measured); Monte Carlo fit {args.run}; min nEff {args.min_eff:g}\n"
        )
        out.to_csv(f, index=False)
    print(f"{len(frame):,} fitted Monte Carlo stars with a claimed class and a colour")
    summary = out.groupby(["class", "nColors", "level"]).chi2.describe()[["count", "min", "50%", "max"]]
    print(summary.to_string())
    print(f"written {args.out}: {len(out)} thresholds")


def main():
    """Command line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    d = sub.add_parser("draw", help="write the Monte Carlo catalogue")
    d.add_argument(
        "--sim", required=True, help="the TRILEGAL sim's rows (sim_full.parquet): positions, r, A_r"
    )
    d.add_argument("--errmodel", required=True, help="the sim's error model (errmodel.npz)")
    d.add_argument("--locus", required=True)
    d.add_argument("--priors", required=True)
    d.add_argument("--floor", type=float, default=0.03)
    d.add_argument("--ar-max", type=float, default=8.0)
    d.add_argument("--dust-curves", default="")
    d.add_argument("--dust-curve-mode", default="band", choices=("band",))
    d.add_argument("--n", type=int, default=3_000_000, help="stars drawn, before the selection")
    d.add_argument("--seed", type=int, default=SEED)
    d.add_argument("--out", required=True, help="the Monte Carlo catalogue (HATS) to write")
    t = sub.add_parser("table", help="tabulate the thresholds of the fitted Monte Carlo")
    t.add_argument("--run", required=True, help="run_dp2's output for the Monte Carlo catalogue")
    t.add_argument("--out", required=True)
    t.add_argument("--alpha", type=float, default=QC_ALPHA)
    t.add_argument("--min-eff", type=float, default=MIN_EFF)
    h = sub.add_parser("hats", help="write a parquet table of rows as a HATS catalogue (draw's last step)")
    h.add_argument("--rows", required=True)
    h.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.step == "hats":
        hats(args)
    elif args.step == "draw":
        if args.dust_curves == "" and args.dust_curve_mode == "band":
            args.dust_curve_mode = "gauss"  # no curves: the box, which the band would be anyway
        draw(args)
    else:
        table(args)


if __name__ == "__main__":
    main()
