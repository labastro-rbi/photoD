
# photod

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/photod?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/photod/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/labastro-rbi/photod/smoke-test.yml)](https://github.com/labastro-rbi/photod/actions/workflows/smoke-test.yml)
[![Read The Docs](https://img.shields.io/readthedocs/photod)](https://photod.readthedocs.io/)

PhotoD is a package that produces color-based distance estimates for billions of stars using broadband optical photometry (e.g. SDSS and LSST). 

This fully Bayesian procedure also produces estimates of stellar parameters such as metallicity and surface gravity, and interstellar dust extinction along the line of sight to each star. These data products enable studies of the Milky Way ranging from tests of models for its formation and evolution to the search for stellar streams, which are excellent probes of dark matter distribution.

## This branch

The `lovorka` branch (tLoc parametrization of the locus) with these changes:
- reddening is added to each fitted color by name; before, the 3D model reddened the wrong columns for loci with
  tLoc in the first column (and failed without the z-y color)
- the A_r prior is flat between 0 and 1.3 A_r(dust map) + 0.1 when `GlobalParams(ArMapColumn=...)` is given,
  instead of 0-2.5 for every star
- Qr is computed from the true Mr for loci in tLoc
- prior maps are matched to the r bins by magnitude, and are made with a binned KDE on a finer grid
- `fixForStripe82` is off by default (it is only meant for the SDSS locus)
- locus points that only pad an isochrone to the rectangular grid get no prior weight
- the per-star fit uses the fact that chi2 is quadratic in A_r; per core it is about 14 times faster than the
  `lovorka` branch with the dust-map A_r prior (2.5 times with the flat prior). The posterior is scaled to its
  maximum, so bright stars with large chi2 no longer give NaN

Scripts written for the `lovorka` branch run unchanged. The only new option is the dust-map A_r prior: pass the
name of the catalog column with A_r from the dust map as `GlobalParams(..., ArMapColumn="...")`.

### Running on Rubin DP2

`scripts/run_dp2.py` runs the whole thing: point sources from the DP2 object catalog, colours and errors from the
PSF fluxes, the DP2 locus, the TRILEGAL prior maps in HATS, one lsdb `merge_map` over the sky, results written
as a HATS catalog:

```
python scripts/run_dp2.py --catalog <rubin_dp2/object_collection> --priors <prior maps> --out <dir> --name dp2_photod
```

`--cone RA DEC RADIUS` runs a piece of sky, `--workers` sets the dask workers (one JAX device each), `--floor`
the colour-error floor and `--no-dust-map` the flat A_r prior.

Two things differ from a run with `LSSTlocus_10Gyr_fix.txt` and the catalog errors as they are, and both were
measured on DP2 stars with Gaia parallaxes and DESI spectra:

- `data/LSSTlocus_10Gyr_DP2.txt` is the locus to use. Its u-g is corrected by the offset between DP2 and the
  locus at fixed spectroscopic [Fe/H] (+0.03 mag for [Fe/H] < -1.5 to -0.16 mag at solar metallicity, the same
  in two fields), which removes a -0.35 dex bias of the photometric [Fe/H] and halves its scatter. Its main
  sequence is corrected in Mr as a function of g-i, from the mean parallax residual of 331,000 Gaia stars in
  five fields (too faint by 0.05-0.26 mag for G and K dwarfs, too bright by 0.14-0.5 mag at the red end),
  measured over all stars with a parallax and no signal-to-noise cut. `scripts/make_locus.py` builds the file
  and re-measures the Mr table for another catalog.
- `GlobalParams(colorErrFloor=0.03)` adds 0.03 mag in quadrature to every colour error. The locus is not exact,
  and without the floor the 68 % intervals of bright stars contain the Gaia parallax 46 % of the time.
- `--dust-curves` uses a 3D dust map as the A_r prior instead of the flat one. It is what matters at low
  Galactic latitude; see below.

On 127,000 DP2 stars with Gaia parallaxes at l = 14, b = -14, none of which entered the calibration, the median
probability integral transform of the observed parallax under the posterior goes from 0.406 to 0.502 (0.500 is
unbiased), the 68 % intervals contain the parallax 65 % instead of 57 % of the time, the mean parallax residual
of M dwarfs from +0.11 mas to +0.01 mas, and the fraction of bright stars (parallax S/N > 10) whose distance
modulus is off by more than a magnitude from 16.5 % to 9.6 %. Four other fields, two of them outside the
calibration, gain as much or more. Prior maps built from the field's own star counts (the TRILEGAL population
reweighted to the observed r and g-r distribution) help M dwarfs further in some fields; the results above use
the standard maps.

### A 3D dust map as the A_r prior

The A_r prior is flat between 0 and 1.3 A_r(map) + 0.1, which says nothing about where along the line of sight
the dust sits. A 3D map does: locus point i puts the star at mu = r - Mr_i - A_r, the map gives the extinction
A*_i that is consistent with that distance, and the prior becomes a Gaussian around A*_i. A giant hypothesis
then implies more dust in front of the star than a dwarf one, and the map can contradict it. Both the chi2 and
this prior are quadratic in A_r, so they combine into one quadratic and the fit keeps its single pass over the
(locus, A_r) grid: the run takes as long as before.

`scripts/make_dust_curves.py` tabulates the shape of the extinction, A_r(mu) / A_r(total), on a HEALPix grid;
each star scales it by its own A_r from the 2D map, so the band and the calibration of the 3D map cancel and
any of them can be used. Sightlines the 3D map does not reach keep the flat prior.

```
python scripts/make_dust_curves.py --map marshall --bmax 12 --out dust_marshall.npz
python scripts/run_dp2.py --catalog ... --priors ... --out ... --dust-curves dust_marshall.npz
```

The A_r grid has to reach the extinction of the field first. The standard "ArLarge" grid stops at 2.5 mag, so
in a field with A_r = 5.5 two stars in five have their A_r pinned at the edge and their distances go with
them; `run_dp2.py --ar-max` sizes the grid instead. The numbers below use a grid sized to the field.

Held-out DP2 stars with Gaia parallaxes, calibrated locus and a 0.03 mag colour-error floor throughout, in the
two fields at low Galactic latitude, where the extinction is large and most stars sit inside the dust:

| field | A_r | prior | PIT | 68 % coverage | mean parallax residual | K dwarfs PIT | M dwarfs PIT / coverage |
|---|---|---|---|---|---|---|---|
| l = 341, b = 2.8 (15,484 stars) | 5.5 | flat | 0.153 | 0.40 | -2.78 mas | 0.000 | 0.000 / 0.13 |
| | | **3D** | **0.539** | **0.60** | **-0.12 mas** | **0.445** | **0.385 / 0.56** |
| l = 345, b = 3.1 (5,469 stars) | 3.1 | flat | 0.449 | 0.59 | -0.73 mas | 0.305 | 0.037 / 0.31 |
| | | **3D** | **0.538** | 0.60 | **-0.06 mas** | **0.545** | **0.547 / 0.53** |

With a flat A_r prior the fit in the plane does not merely lose precision: a median PIT of 0.000 for both K and
M dwarfs means that every one of them falls outside its own posterior, and the mean parallax residual of the
M dwarfs is -20 mas. Extinction is free to take any value the colours allow, and it trades against absolute
magnitude. The 3D prior removes that freedom: PIT returns to about 0.5, the coverage of M dwarfs goes from 0.13
to 0.56 and their mean residual from -20 mas to -0.09 mas. At A_r = 3 the mean residual improves from -0.73 mas
to -0.06 mas and the bright-star scatter halves. It does not make the plane easy - the coverage is 0.60 against
a nominal 0.68, and the fraction of bright stars whose distance modulus is off by more than a magnitude does
not improve - but it makes it usable.

Above |b| = 10 there is nothing to win, and the flat prior is the better choice: measured in three fields with
A_r of 1.0, 0.5 and 0.2, the 3D prior takes a quarter off the bright-star scatter at A_r = 1 but is neutral to
slightly worse below that, because nearly every star is behind all of the dust and A_r(mu) is flat where it
matters.

Two practical points. The 3D maps disagree with the 2D total column by 24-30 % at high latitude (Bayestar) but
only 3-6 % in the plane (Marshall), which is why the shape is taken from the 3D map and the scale from the 2D
map. And the maps cover different parts of the sky: Marshall (|l| < 100, |b| < 10, any declination, ~10 kpc)
and Chen 2018 (the plane including the anticentre) together cover the Galactic plane, which is the region that
gains; Bayestar reaches three quarters of the sky but stops at dec > -30, and above |b| = 10 it is not needed.
