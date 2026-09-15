
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
- the per-star fit uses the fact that chi2 is quadratic in A_r and is about 25 times faster per core; the posterior
  is scaled to its maximum, so bright stars with large chi2 no longer give NaN

Scripts written for the `lovorka` branch run unchanged. The only new option is the dust-map A_r prior: pass the
name of the catalog column with A_r from the dust map as `GlobalParams(..., ArMapColumn="...")`.
