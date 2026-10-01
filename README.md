
# photod

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/photod?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/photod/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/labastro-rbi/photod/smoke-test.yml)](https://github.com/labastro-rbi/photod/actions/workflows/smoke-test.yml)
[![Read The Docs](https://img.shields.io/readthedocs/photod)](https://photod.readthedocs.io/)

PhotoD is a package that produces color-based distance estimates for billions of stars using broadband optical photometry (e.g. SDSS and LSST). 

This fully Bayesian procedure also produces estimates of stellar parameters such as metallicity and surface gravity, and interstellar dust extinction along the line of sight to each star. These data products enable studies of the Milky Way ranging from tests of models for its formation and evolution to the search for stellar streams, which are excellent probes of dark matter distribution.

## This branch

The tLoc parametrization of the locus, with these changes:
- reddening is added to each fitted color by name; before, the 3D model reddened the wrong columns for loci with
  tLoc in the first column (and failed without the z-y color)
- the A_r prior is flat between 0 and 1.3 A_r(dust map) + 0.1 when `GlobalParams(ArMapColumn=...)` is given,
  instead of 0-2.5 for every star
- Qr is computed from the true Mr for loci in tLoc
- prior maps are matched to the r bins by magnitude, and are made with a binned KDE on a finer grid
- `fixForStripe82` is off by default (it is only meant for the SDSS locus)
- locus points that only pad an isochrone to the rectangular grid get no prior weight
- the per-star fit uses the fact that chi2 is quadratic in A_r; per core it is about 14 times faster than the
  tLoc parametrization with the dust-map A_r prior (2.6 times with the flat prior: 7.0 and 37 ms per star against
  98, on one core of the same machine). The posterior is scaled to its maximum, so bright stars with large
  chi2 no longer give NaN

Scripts written for the tLoc parametrization run unchanged. The only new option is the dust-map A_r prior: pass the
name of the catalog column with A_r from the dust map as `GlobalParams(..., ArMapColumn="...")`.

`tLoc/` holds the scripts and the notebook the tLoc parametrization was worked out with, kept for reference.
Nothing in the package reads them; `src/photod` is the code that runs.

### Installing

The package needs `lsdb`, `jax`, `astropy` and `scipy`, and imports `numpy`, `pandas`, `nested-pandas`,
`mocpy` and `pyyaml` itself rather than leaving them to those; `pip install -e .` brings in all of them.
Install it from a
clone rather than as a wheel: the locus tables and the prior maps live in `data/` of the
repository rather than inside the package, and that is where the defaults look for them.

### Running on Rubin DP2

`scripts/run_dp2.py` runs the whole thing: point sources from the DP2 object catalog, colours and errors from the
PSF fluxes, the DP2 locus, the TRILEGAL prior maps, a pool of processes over the partition files of the
catalog, results written as a HATS catalog:

```
python scripts/run_dp2.py --catalog <rubin_dp2/object_collection> --out <dir> --name dp2_photod --workers 6
```

`notebooks/run_dp2.ipynb` walks through the same thing: one field first, the checks worth doing on the
answer, then the survey.

The prior maps and the dust curves for the DP2 footprint come with the repository, in `data/`, so a clone has
everything the fit needs and nothing has to be built or fetched first. `--priors` and `--dust-curves` point at
them by default; `--dust-curves ""` turns the 3D dust prior off.

`data/priors_dp2.npz` is 80 MB because the maps are kept on every other point of the grid as a byte of log
each, which moves the median star by under two millimagnitudes. The exact maps, 1366 MB, are attached to the
release for reproducing the published catalog; pass them with `--priors`, which needs no other change, though
they are on the full grid rather than on every other point of it, and the run unpacks the compact form to
scratch first. `scripts/make_priors.py` builds either kind for another footprint, with `--compact` for the
small form; it records the grid it built them on in the file, and the run refuses a file built on another.

Three things about the file that ships here, all from the build that made it: it records no grid, so it is
taken as it comes; three of the order 5 pixels its dust file covers (4821, 4910, 4962, about 10 deg2) have no
maps, and stars of those sightlines come back with bit 64 set and no estimate rather than missing from the
catalog; and its [Fe/H] axis stops at 0.9 instead of 1.0, because the compact writer used to drop the last
point of an axis. The last of those changes nothing, the locus stopping at [Fe/H] = +0.5, and a file built
now carries the whole axis.

`--cone RA DEC RADIUS` runs a piece of sky, `--workers` sets the processes, which share whatever GPUs are
there, `--chunk` how many partitions a process handles before it is replaced, `--floor` the colour-error floor,
`--ar-max` the top of the A_r grid and `--no-dust-map` the flat A_r prior. A partition is a file and fitting
one has nothing to say to the next, so there is nothing to schedule: the pool reads, fits and writes one file
per task, which is what keeps the memory of a survey-wide run flat.

A run that stops part way is finished by repeating the command: a partition whose answers are already there is
left alone, and `--overwrite` starts the result again from nothing. Each file is put in place in one step, so a
killed run leaves whole answers or none, never half of one. A partition that fails takes only itself down, and
the run ends with a non-zero status naming how many are missing; repeating the command fits those.

What the run was configured with is written beside the answers as `photod_run.json`, and a resume that asks
for another fit — another catalog, prior file, error floor, A_r grid, dust prior or cone — is refused, because
the partitions already written are kept whatever this run asks for and nothing in a HATS catalog could later
say which of its stars came from which of the two fits. How the work is divided up is not part of that, so a
run started on a machine with two GPUs is free to finish on a laptop with none.

A batch of stars is also bounded in memory rather than in stars: the posterior of a batch is the locus by the
star's A_r grid, which for the full DP2 locus on a grid to 8 mag is tens of gigabytes at a few hundred stars,
so `--batch-size` is an upper bound and `--batch-bytes`, 2 GiB by default, the limit that applies. It is the
budget of one batch of one worker, so a pool of six asks for six times as much at once.

The prior maps are built once for a footprint by `scripts/make_priors.py`, one map per r bin and HEALPix
pixel, on the tLoc axis of the locus the fit uses.

Three things to know about what a run is set up with:

- `data/LSSTlocus_10Gyr_fix.txt` is the locus a run fits against, named once as
  `photod.locus.RUN_LOCUS_FILE` so that the run, the prior maps and the tests use the same table: the maps are
  indexed by its tLoc, so they have to agree. The two corrections `scripts/make_locus.py` measures on DP2 are
  not applied; that script says why, and it and `data/LSSTlocus_10Gyr_DP2.txt` are kept as the record of those
  measurements.
- `GlobalParams(colorErrFloor=0.03)` adds 0.03 mag in quadrature to every colour error. The locus is not exact,
  and without the floor the 68 % intervals of bright stars contain the Gaia parallax 46 % of the time.
- `--dust-curves` shapes the A_r prior with a 3D dust map. It is off by default: the prior is bounded by the
  extinction column, which is the 2D map, and is flat within that bound. It is what matters at low Galactic
  latitude; see below.

On 127,000 DP2 stars with Gaia parallaxes at l = 14, b = -14, none of which entered the calibration, the median
probability integral transform of the observed parallax under the posterior goes from 0.406 to 0.502 (0.500 is
unbiased), the 68 % intervals contain the parallax 65 % instead of 57 % of the time, the mean parallax residual
of M dwarfs from +0.11 mas to +0.01 mas, and the fraction of bright stars (parallax S/N > 10) whose distance
modulus is off by more than a magnitude from 16.5 % to 9.6 %. Four other fields, two of them outside the
calibration, gain as much or more. Prior maps built from the field's own star counts (the TRILEGAL population
reweighted to the observed r and g-r distribution) help M dwarfs further in some fields; the results above use
the standard maps.

### What the fit writes

Per star, beside `objectId`, `ra`, `dec` and `rmag`: the 16th, 50th and 84th percentiles of the fitted
absolute magnitude, the metallicity `FeH`, the extinction `Ar`, the reddened absolute magnitude
`Qr = Mr + Ar` and the distance modulus `DM`, the entropy the data took out of the prior for the first three,
`chi2min`, and `flags`.

The fit is parametrised by tLoc, the coordinate that runs along the locus, so **`Mr_quantile_*` holds tLoc and
the absolute magnitude is `Mr_true_quantile_*`**. The two agree on the main sequence and part company above
the turn-off, which is where the giants are. `Qr`, `DM` and the two-branch flag are all computed from
`Mr_true`, so distances need no conversion; only a column named `Mr` does.

`DM = rmag - Qr`, so its percentiles come straight from the posterior rather than from combining two
marginals, and the distance is `10 ** (DM / 5 + 1)` parsecs.

`chi2min` is the smallest chi2 over the whole A_r grid, so it says whether the locus passes through the star's
colours at any extinction, not whether it does so at one the prior allows. A star whose extinction is pinned
against its prior bound has bit 8 set and a small `chi2min`.

The three entropy columns are in bits, the entropy of the posterior less that of the prior, so a negative
number is what the colours took out of the prior.

Positions and `rmag` are carried over from the catalog as float64, the fitted columns are float32, the object
id is int64 and `flags` is int32, whether a partition holds stars or none. Answers are written one partition
at a time, indexed and sorted by `_healpix_29`, which is what makes the result a HATS catalog.

`flags` is one bit per thing worth knowing about the answer. Nothing is dropped for being flagged; the row
stays and says so, empty if the fit could not be run at all.

| bit | meaning | on DP2 |
|---|---|---|
| 1 | chi2 above 100, or no answer at all: the locus does not pass through this star's colours | 2.72 % |
| 2 | the Mr_true posterior is lopsided, which is how a giant and a dwarf solution both survive | 3.45 % |
| 4 | [Fe/H] is against the end of the model grid, so it is a limit rather than a measurement | 0.93 % |
| 8 | A_r is against its bound, the dust map's or the grid's, and the distance goes wrong with it | 0.08 % |
| 16 | a colour was effectively unmeasured: its error is above a magnitude and it carries no weight worth the name | 56.2 % |
| 32 | no r magnitude, so neither a prior map nor a distance: the row carries no estimate | 0 |
| 64 | no prior map for this part of the sky, so the fit was never run: no estimate either | 0 |
| 128 | the posterior collapsed into one cell of the grid: precise-looking and not to be trusted | 1.3 % |

The fractions are of the 93,688,843 stars of the whole DP2 run, of which 40.2 % carry no flag at all and
94.2 % are usable. Neither of the last two bits is set by a single star of it: every point source has an r
magnitude, and the three sky pixels the prior file that ships here has no maps for hold none of them.

The first two are the ones that mean the answer is suspect rather than merely uncertain. All of the following
is measured against Gaia DR3 over the whole of the run above: 48.4 million of its stars have a counterpart
within an arcsecond, 1.5 million of those a parallax of signal-to-noise ten or better, and the uncertainty a
star quotes is taken as half its 14th-to-86th percentile range.

The distance scale of an unflagged star is right. Binning on the fit's own parallax and averaging Gaia's,
which is the way round that no parallax cut can bias, the two agree to 0.02 mas at every distance out to five
kiloparsecs, a few per cent, and inside Gaia's own systematic floor. What is not right is the uncertainty:
it is too small by a factor of 1.7, so multiply it by that before using it, and the tails are worse than a
Gaussian's, one unflagged star in six lying beyond three of its own sigma.

A star with bit 1 set is off by seven magnitudes in the median, and its quoted uncertainty says nothing at
all: the median is 0.03 mag on a distance wrong by a factor of thirty. What is wrong with those stars is the
DP2 photometry rather than the fit, their r magnitude differing from Gaia's G by more than a magnitude for
two fifths of them, so the fit is being asked to place a star that is not there. Bit 2 is wrong in both
directions and about half of those stars pick the wrong branch. Bit 4 marks a limit rather than a failure,
and four fifths of the stars carrying it carry bit 1 as well. Bit 16 is about the input and is normal: it is
nearly all the u band, which is the shallowest, and more than half of DP2 has no usable u-g; those stars are
no worse placed than the rest, and marginally better calibrated, their wider intervals being more honest.

So the cut to reach for is `flags & 131 == 0`: bits 1, 2 and 128, the three that mean the answer is wrong
rather than uncertain. Bit 128 is the last of those, and it is there because a narrow posterior is not
always a precise one. Measured against Gaia parallaxes over unflagged stars whose parallax is good to ten per
cent, the half-width of the Qr interval separates the two cleanly: above 0.1 mag, 0.3 to 5 % of stars have
[Fe/H] pinned at an end of its grid and the median distance is off by 0.41 to 0.47 mag; below it, 40 % are
pinned, and between 0.02 and 0.05 mag the median distance is off by 9.5 magnitudes. The fit sets bit 128 on
those, 1.3 % of the stars that would otherwise look like the most precise in the catalog.

### A 3D dust map as the A_r prior

The A_r prior is flat between 0 and 1.3 A_r(map) + 0.1, which says nothing about where along the line of sight
the dust sits. A 3D map does: locus point i puts the star at mu = r - Mr_i - A_r, the map gives the extinction
A*_i that is consistent with that distance, and the prior becomes a Gaussian around A*_i. A giant hypothesis
then implies more dust in front of the star than a dwarf one, and the map can contradict it. Both the chi2 and
this prior are quadratic in A_r, so they combine into one quadratic and the fit keeps its single pass over the
(locus, A_r) grid: the run takes as long as before.

The curves themselves no longer ship here, and the script that built them has been removed: no run uses the
3D prior. `--dust-curves` still takes such a file if one is built elsewhere, and `readCurves` in
`scripts/run_dp2.py` documents the format it expects.

The A_r grid has to reach the extinction of the field first. The `ArLarge` grid of `get3DmodelList` stops at
2.5 mag, so in a field with A_r = 5.5 two stars in five have their A_r pinned at the edge and their distances
go with them. `run_dp2.py` therefore builds the grid from `--ar-max`, 8 mag by default, which covers the
plane; the numbers below use a grid sized to the field. A star pinned against the bound anyway, whether the
grid's or its own from the dust map, carries bit 8.

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
