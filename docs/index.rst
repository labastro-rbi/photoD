
.. photod documentation main file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PhotoD
========================================================================================

PhotoD is a package that produces color-based distance estimates for billions of stars using broadband optical photometry (e.g. SDSS and LSST).

This fully Bayesian procedure also produces estimates of stellar parameters such as metallicity and surface gravity, and interstellar dust extinction
along the line of sight to each star. These data products enable studies of the Milky Way ranging from tests of models for its formation and evolution
to the search for stellar streams, which are excellent probes of dark matter distribution.

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Generate data inputs <data_generation>
   Compute Bayes estimates <pre_executed/run_with_lsdb_gpu.ipynb>
   Plot posteriors <pre_executed/plotting/plot_stars.ipynb>
   Validate estimates <validation>
   Dev guide <dev_guide>
 