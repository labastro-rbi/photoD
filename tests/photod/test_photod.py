import jax
import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
from dask import delayed
from numpy.testing import assert_allclose
from scipy.interpolate import griddata

import photod.locus as lt
from photod.bayes import makeBayesEstimates3d
from photod.parameters import GlobalParams


def merging_function(partition, map_partition, partition_pixel, map_pixel, globalParams, *kwargs):
    priorGrid = {}
    for rind, r in enumerate(np.sort(map_partition["rmag"].to_numpy())):
        # interpolate prior map onto locus Mr-FeH grid
        Z = map_partition[map_partition["rmag"] == r]
        Zval = np.frombuffer(Z.iloc[0]["kde"], dtype=np.float64).reshape((96, 36))
        X = np.frombuffer(Z.iloc[0]["xGrid"], dtype=np.float64).reshape((96, 36))
        Y = np.frombuffer(Z.iloc[0]["yGrid"], dtype=np.float64).reshape((96, 36))
        points = np.array((X.flatten(), Y.flatten())).T
        values = Zval.flatten()
        # actual (linear) interpolation
        priorGrid[rind] = griddata(
            points,
            values,
            (globalParams.locusData["FeH"], globalParams.locusData[globalParams.MrColumn]),
            method="linear",
            fill_value=0,
        )
    priorGrid = jax.numpy.array(list(priorGrid.values()))
    estimatesDf, _ = makeBayesEstimates3d(partition, priorGrid, globalParams, batchSize=10)
    return npd.NestedFrame(estimatesDf)


def test_make_bayes_estimates_3d(tmp_path, s82_0_5_dir, s82_priors_dir, locus_file_path):
    """End to end test of the Make Bayes Estimates 3D for S82 HP(5,0)"""
    LSSTlocus = lt.LSSTsimsLocus(fixForStripe82=False, datafile=locus_file_path)
    OKlocus = LSSTlocus[(LSSTlocus["gi"] > 0.2) & (LSSTlocus["gi"] < 3.55)]
    locusData = lt.subsampleLocusData(OKlocus, kMr=10, kFeH=2)

    fitColors = ("ug", "gr", "ri", "iz")

    fitColors = ("ug", "gr", "ri", "iz")
    ArGridList, locus3DList = lt.get3DmodelList(locusData, fitColors)
    globalParams = GlobalParams(fitColors, locusData, ArGridList, locus3DList)

    col_names = [
        "glon",
        "glat",
        "FeHEst",
        "FeHUnc",
        "MrEst",
        "MrUnc",
        "chi2min",
        "MrdS",
        "FeHdS",
        "ArEst",
        "ArUnc",
        "ArdS",
        "D",
        "DUnc"
    ]
    meta = npd.NestedFrame.from_dict({col: pd.Series([], dtype=float) for col in col_names})

    s82_stripe_catalog = lsdb.read_hats(s82_0_5_dir)
    prior_map_catalog = lsdb.read_hats(s82_priors_dir)

    delayed_global_params = delayed(globalParams)

    merge_lazy = s82_stripe_catalog.merge_map(
        prior_map_catalog, merging_function, globalParams=delayed_global_params, meta=meta
    )
    result = merge_lazy.compute()

    expected_metallicity = [-0.627993, -0.379619, -0.608126, -2.232135]
    assert_allclose(result["FeHEst"][0:4], expected_metallicity, rtol=1e-6)
