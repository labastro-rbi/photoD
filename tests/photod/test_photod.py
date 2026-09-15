import jax
import lsdb
import nested_pandas as npd
import numpy as np

import photod.locus as lt
from photod.bayes import getEstimatesMeta, makeBayesEstimates3d
from photod.parameters import GlobalParams
from photod.priors import initializePriorGrid


def merging_function(partition, map_partition, partition_pixel, map_pixel, globalParams, **kwargs):
    """Bayes estimates for one catalog partition and the prior maps of its sky pixel."""
    # the test catalog predates the objectId column
    partition = partition.assign(objectId=np.arange(len(partition)))
    priorGrid = jax.numpy.array(list(initializePriorGrid(map_partition, globalParams).values()))
    estimatesDf, _ = makeBayesEstimates3d(partition, priorGrid, globalParams, batchSize=10)
    return npd.NestedFrame(estimatesDf)


def test_make_bayes_estimates_3d(s82_0_5_dir, s82_priors_dir, locus_file_path):
    """End to end test of the Make Bayes Estimates 3D for S82 HP(5,0)"""
    LSSTlocus = lt.LSSTsimsLocus(datafile=locus_file_path)
    OKlocus = LSSTlocus[(LSSTlocus["gi"] > 0.2) & (LSSTlocus["gi"] < 3.55)]
    locusData = lt.subsampleLocusData(OKlocus, kMr=10, kFeH=2)

    fitColors = ("ug", "gr", "ri", "iz")
    ArGridList, locus3DList = lt.get3DmodelList(locusData, fitColors)
    globalParams = GlobalParams(fitColors, locusData, ArGridList, locus3DList)

    s82_stripe_catalog = lsdb.read_hats(s82_0_5_dir)
    prior_map_catalog = lsdb.read_hats(s82_priors_dir)
    merge_lazy = s82_stripe_catalog.merge_map(
        prior_map_catalog, merging_function, globalParams=globalParams, meta=getEstimatesMeta()
    )
    result = merge_lazy.compute()

    assert len(result) == len(s82_stripe_catalog.compute())
    finite = np.isfinite(result["Mr_quantile_median"])
    assert finite.mean() > 0.95
    assert np.all(result["Mr_quantile_lo"][finite] <= result["Mr_quantile_median"][finite])
    assert np.all(result["Mr_quantile_median"][finite] <= result["Mr_quantile_hi"][finite])
    assert np.all(
        (result["FeH_quantile_median"][finite] >= -2.5) & (result["FeH_quantile_median"][finite] <= 0.5)
    )
