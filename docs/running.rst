Run the fit on a survey
========================================================================================

``scripts/run_dp2.py`` is the entry point for a whole survey: it reads point sources from the object
catalog partition by partition, builds the colours and their errors from the PSF fluxes, fits them against
the locus ``photod.locus.RUN_LOCUS_FILE`` names with the TRILEGAL prior maps ``--priors`` names, and writes
the answers as a HATS catalog. The A_r prior is bounded by the extinction column and is not shaped by
the 3D dust curves unless ``--dust-curves`` names them. A star with no value in that column is fitted over
the whole A_r grid, as in a run with ``--no-dust-map``, and no flag marks it.

.. code-block:: bash

    python scripts/run_dp2.py --catalog <rubin_dp2/object_collection> --priors <priors.npz> --out <dir> \
        --name dp2_photod --workers 6

On a machine with several GPUs the survey can be split between processes, each pinned to its own GPU and
given the same ``--out`` and ``--name``::

    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python scripts/run_dp2.py --catalog <...> --priors <priors.npz> --out <dir> \
            --workers 3 --shard $i/4 &
    done; wait

Each fits every fourth partition, and whichever finishes last writes the catalog metadata for all of them.
``--overwrite`` is refused with ``--shard``: remove the directory once before starting the shards.

Two or three workers per GPU is the whole of it. One process already keeps a GPU at four fifths of its
utilisation and reaches the throughput of a shard fitting the survey, because the fit is the run: reading
the partition, building the colours, interpolating the prior maps and writing the answers together take
one per cent of the time. Twelve per GPU on four H100s fitted 110 million stars at 11,760 stars a second
per GPU, which is the rate a single process reaches on the same partitions, so the other eleven bought
nothing and cost forty-eight processes of memory instead of four.

A batch of each worker is bounded by ``--batch-bytes``, 16 GB by default. Large batches are what make the fit
fast, the stars on a deep A_r grid most of all, and a batch holds well under half of its bound: a DP2 worker
peaked at 6.2 GB of GPU memory, and two of them shared an 80 GB H100 at 18 GB. On a smaller GPU, or one
shared with other work, give it less.

``notebooks/run_dp2.ipynb`` walks through the same run on one field first, with the checks worth making on
the answer, and then over the survey.

Per star the result carries ``objectId``, ``ra``, ``dec`` and ``rmag``; ``chi2min``, the chi-squared of the
best fit; ``fit_flags``, one bit per thing worth knowing about the answer; ``multimodality``; and for the
absolute magnitude ``Mr``, the metallicity ``FeH``, the extinction ``Ar``, the distance modulus ``DM`` and
``tLoc`` the 16th, 50th and 84th percentiles and the standard deviation of the posterior, as
``<name>_quantile_lo``, ``_quantile_median``, ``_quantile_hi`` and ``<name>_std``. tLoc is the coordinate
along the locus that the fit is parametrised by. It equals the absolute magnitude only on the main sequence
fainter than the turn-off (tLoc > 4) and not at the turn-off or brighter, so the absolute magnitude is
``Mr``. The interval and the standard deviation of ``DM`` include the
error on r. ``multimodality`` is how much of the distance modulus posterior lies in a second peak: zero when
the posterior has one, however skewed, and for two a lower bound on the weight of the smaller.

Beside the estimates each row carries the columns of the catalog row it was fitted from: every column of a
prepared catalog, and of the survey object table the ones the fit reads. A catalog column named like an
estimate is written as ``input_<name>``, and ``--no-input-columns`` leaves them all out.

=====  ==================================================================================================
Bit    ``fit_flags``
=====  ==================================================================================================
1      a poor fit: ``chi2min`` above 100, or no answer at all
2      the ``Mr`` posterior is lopsided, which is how a giant and a dwarf solution both survive
4      [Fe/H] is against the end of the model grid, so it is a limit rather than a measurement
8      A_r is against the top of what the grid and the dust map allow, and the distance goes wrong with it
16     at least one colour was not measured and carried no weight in the fit
32     no r magnitude, so neither a prior map nor a distance: the row carries no estimate
64     no prior map for this part of the sky, so the fit was never run: no estimate either
128    the posterior fell into a single grid cell: precise-looking and not to be trusted
=====  ==================================================================================================

Which rows are fitted is ``--selection``. ``recipe``, the default, fits the objects with S/N above 10 in r and
above 3 in g and i (``photod.selection``), and flags those past r 23.5; ``flag`` fits every row and flags the
ones the recipe would drop, so that a prepared catalog keeps all of its rows; ``none`` fits a prepared catalog
as it comes, and of the survey table every point source fainter than r 16.5, with no S/N cut and no r limit. A
colour whose error is above 1 mag, or that is not a number, carries no weight.

White dwarfs and QSOs
---------------------

``data/LSSTlocus_10Gyr_WDmass_QSO.txt`` (``photod.locus.WD_LOCUS_FILE``) is the stellar locus with white dwarf
tracks below it on the tLoc axis, one hydrogen and one helium, whose [Fe/H] rows are masses, and a QSO track
between them and the stars, whose tLoc stands for a redshift. ``scripts/make_qso_locus.py`` builds it from the
files in ``data/``. Its prior maps hold the stars, each white dwarf track and the QSOs as populations, each
held to its share per r bin, and ``scripts/make_priors.py`` builds them with ``--locus``, ``--wd-mass-grid
data/WD_massgrid_QSO.json``, ``--qso-counts`` and ``--dust-map``. The DP2 catalog of the three classes was fitted
with

.. code-block:: bash

    python scripts/make_expo_dust_curves.py --out expo_h125.npz
    python scripts/run_dp2.py --catalog <dp2_inclusive> --priors <white dwarf and QSO priors.npz> \
        --locus data/LSSTlocus_10Gyr_WDmass_QSO.txt --qc-table data/qc_table_dp2_inclusive_wd.csv \
        --floor 0.03 --dust-curves expo_h125.npz --dust-curve-mode band --estimate-class MS \
        --selection flag --ar-column Ar_SFD --ar-scale 0.86005183 --ar-max 8.0 --out <dir> --name dp2_inclusive

``--dust-curve-mode band`` bounds the A_r prior of the white dwarf and QSO rows by the dust in front of each
point's distance, here an exponential layer of 125 pc scale height; ``--qc-table`` sets bit 1024 from the
model's own null distribution of ``chi2min`` (``scripts/make_qc_table.py``); ``--estimate-class MS`` gives every
row the estimates of the stars, whatever class is the most probable.

Every run writes the class columns, which on a locus without the tracks are those of the stars alone:
``p_MS``, ``p_WD`` and ``p_QSO``, the posterior mass of each class, ``p_He`` the mass on the helium white
dwarfs, a part of ``p_WD``, ``lnodds_WD`` and ``lnprior_odds_WD`` the log odds of white dwarf against the rest
after the fit and in the prior alone, ``class`` the most probable of the three (0, 1, 2; -1 with no answer),
``nColors`` the colours measured, ``chi2minMS``, ``chi2minWD`` and ``chi2minQSO`` the best chi-squared within
each class, ``DM_WD_quantile_lo``, ``_median`` and ``_hi`` the white dwarf distance modulus and ``WD_mass``
its mass, both where ``p_WD`` is above 0.05, and ``photoz`` the median redshift along the QSO track. Without
``--estimate-class MS`` the other estimates are of the most probable of the stars and the white dwarfs, and
``DM`` is empty where the QSO class is the most probable.

=====  ==================================================================================================
Bit    ``fit_flags``, continued
=====  ==================================================================================================
256    past r 23.5, the r limit of the selection recipe
512    the QSO class is the most probable
1024   ``chi2min`` of the most probable class is above the 99th percentile of the model's own null for the
       colours measured and the latitude (set only with ``--qc-table``)
2048   fewer than three colours measured
4096   fails the S/N cuts of the selection recipe, fitted because the run keeps every row (``--selection flag``)
=====  ==================================================================================================

``python scripts/run_dp2.py --help`` lists the options. The pages below predate this pipeline and show the
library being driven directly, which is still how a single field or a handful of stars is best looked at.
