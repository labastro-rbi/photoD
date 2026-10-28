"""The QC table of scripts/make_qc_table.py: the thresholds of chi2min under the model's own null."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from wd_loci import script


def test_the_qc_thresholds_are_weighted_percentiles_pooled_where_a_cell_is_thin():
    """The weighted 99th percentile per cell, and a cell no level has enough stars for is left empty."""
    qc = script("make_qc_table")
    rng = np.random.default_rng(5)
    n = 40000
    frame = pd.DataFrame(
        {
            "claimed": rng.choice([0, 1], n, p=[0.7, 0.3]),
            "lat": rng.choice([0, 1, 2], n),
            "pattern": 31,
            "nColors": 5,
            "chi2": rng.chisquare(3, n),
            "w": rng.choice([0.5, 2.0], n),
        }
    )
    thin = pd.DataFrame({"claimed": 1, "lat": 0, "pattern": 7, "nColors": 3, "chi2": [50.0] * 20, "w": 1.0})
    out = qc.thresholds(pd.concat([frame, thin], ignore_index=True), alpha=0.01, minEff=1000)
    cell = out[(out["class"] == "MS") & (out.latitude == 1) & (out.pattern == 31)].iloc[0]
    g = frame[(frame.claimed == 0) & (frame.lat == 1)]
    assert cell.level == "cell"
    assert_allclose(cell.chi2, qc.weightedQuantile(g.chi2.to_numpy(), g.w.to_numpy(), 0.99), rtol=1e-4)
    assert abs(cell.chi2 - 11.34) < 1.0  # chi2(3) 99th percentile
    assert out[(out["class"] == "WD") & (out.pattern == 7)].empty  # 20 stars, nothing to pool them with
    x = np.arange(1000.0)
    assert qc.weightedQuantile(x, np.ones(1000), 0.99) == 989.0
