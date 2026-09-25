Run the fit on a survey
========================================================================================

``scripts/run_dp2.py`` is the entry point for a whole survey: it reads point sources from the object
catalog partition by partition, builds the colours and their errors from the PSF fluxes, fits them against
the locus in ``data/LSSTlocus_10Gyr_DP2.txt`` with the TRILEGAL prior maps and the 3D dust curves that ship
in ``data/``, and writes the answers as a HATS catalog.

.. code-block:: bash

    python scripts/run_dp2.py --catalog <rubin_dp2/object_collection> --out <dir> --name dp2_photod --workers 6

``notebooks/run_dp2.ipynb`` walks through the same run on one field first, with the checks worth making on
the answer, and then over the survey.

Per star the result carries the 16th, 50th and 84th percentiles of the fitted absolute magnitude, the
metallicity, the extinction, the reddened absolute magnitude ``Qr`` and the distance modulus ``DM``, the
entropy the colours took out of the prior, ``chi2min``, and a ``flags`` column with one bit per thing worth
knowing about the answer. The fit is parametrised by tLoc, the coordinate along the locus, so
``Mr_quantile_*`` holds tLoc and the absolute magnitude is ``Mr_true_quantile_*``.

The README of the repository is the reference for the options, the flag bits, the 3D dust prior and what was
measured against Gaia parallaxes and DESI spectra. The pages below predate this pipeline and show the
library being driven directly, which is still how a single field or a handful of stars is best looked at.
