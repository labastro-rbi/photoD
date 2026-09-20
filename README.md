
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

On 127,000 DP2 stars with Gaia parallaxes at l = 14, b = -14, none of which entered the calibration, the median
probability integral transform of the observed parallax under the posterior goes from 0.406 to 0.502 (0.500 is
unbiased), the 68 % intervals contain the parallax 65 % instead of 57 % of the time, the mean parallax residual
of M dwarfs from +0.11 mas to +0.01 mas, and the fraction of bright stars (parallax S/N > 10) whose distance
modulus is off by more than a magnitude from 16.5 % to 9.6 %. Four other fields, two of them outside the
calibration, gain as much or more. Prior maps built from the field's own star counts (the TRILEGAL population
reweighted to the observed r and g-r distribution) help M dwarfs further in some fields; the results above use
the standard maps.
