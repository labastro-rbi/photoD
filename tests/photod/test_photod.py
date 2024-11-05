import photod.bayes as bt
import photod.locus as lt


def test_make_bayes_estimates_3d(tmp_path, s82_0_5_df):
    """End to end test of the Make Bayes Estimates 3D for S82 HP(5,0)"""
    LSSTlocus = lt.LSSTsimsLocus(
        fixForStripe82=False, datafile="/home/scampos/photoD/data/MSandRGBcolors_v1.3.txt"
    )
    OKlocus = LSSTlocus[(LSSTlocus["gi"] > 0.2) & (LSSTlocus["gi"] < 3.55)]
    locusData = lt.subsampleLocusData(OKlocus, kMr=10, kFeH=2)

    fitColors = ("ug", "gr", "ri", "iz")
    ArGridList, locus3DList = lt.get3DmodelList(locusData, fitColors)
    priorsRootName = "/mnt/beegfs/scratch/scampos/photod/priors/TRILEGAL/S82/5/0"
    outfile = tmp_path / "results.txt"

    bt.makeBayesEstimates3D(
        s82_0_5_df.reset_index(drop=True),
        fitColors,
        locusData,
        locus3DList,
        ArGridList,
        priorsRootName,
        outfile,
        iStart=0,
        iEnd=len(s82_0_5_df),
    )
