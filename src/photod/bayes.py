import nested_pandas as nd
import numpy as np
from astropy.table import Table

import photod.locus as lt
from photod.plotting import plot_star
from photod.priors import getBayesConstants, getPriorMapIndex, make3Dprior, readPriors
from photod.stats import Entropy, getMargDistr3D, getStats


def makeBayesEstimates3D(
    catalog: nd.NestedFrame,
    fitColors: tuple,
    locusData: Table,
    locus3DList: dict[str, np.ndarray],
    ArGridList: dict[int, np.ndarray],
    priorsRootName: str,
    outfile: str,
    iStart: int = 0,
    iEnd: int = -1,
    myStars: list = [],
    verbose: bool = False,
    MrColumn: str = "Mr",
):
    (
        ArCoeff,
        ArGridMediumMax,
        ArGridSmallMax,
        FeH1d,
        Mr1d,
        dFeH,
        dMr,
        iEnd,
        iStart,
        mdLocus,
        priorGrid,
        priorind,
        reddCoeffs,
        xLabel,
        yLabel,
    ) = _setup_bayes3d(ArGridList, MrColumn, catalog, iEnd, iStart, locusData, priorsRootName)

    #######################################################################
    loop_bayes_3d(
        ArCoeff,
        ArGridList,
        ArGridMediumMax,
        ArGridSmallMax,
        FeH1d,
        Mr1d,
        catalog,
        dFeH,
        dMr,
        fitColors,
        iEnd,
        iStart,
        locus3DList,
        mdLocus,
        myStars,
        priorGrid,
        priorind,
        reddCoeffs,
        verbose,
        xLabel,
        yLabel,
    )

    # store results
    # writeBayesEstimates(catalog, outfile, iStart, iEnd, do3D=True)


def loop_bayes_3d(
    ArCoeff,
    ArGridList,
    ArGridMediumMax,
    ArGridSmallMax,
    FeH1d,
    Mr1d,
    catalog,
    dFeH,
    dMr,
    fitColors,
    iEnd,
    iStart,
    locus3DList,
    mdLocus,
    myStars,
    priorGrid,
    priorind,
    reddCoeffs,
    verbose,
    xLabel,
    yLabel,
):
    selections = step_1(ArCoeff, ArGridMediumMax, ArGridSmallMax, catalog["Ar"])

    ## loop over all stars (can be trivially parallelized)
    for i in range(iStart, iEnd):
        Ar1d, locus3Dok = ArGridList[selections[i]], locus3DList[selections[i]]

        ##### the main Bayes block
        ## compute chi2 map using provided 3D model locus (locus3Dok) ##
        # return i, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, FeH1d, Mr1d, Ar1d
        chi2map = lt.getPhotoDchi2map3D(
            i, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, masterLocus=True
        )
        dAr, likeCube, priorCube = like_and_prior(Ar1d, FeH1d, Mr1d, catalog, chi2map, i, priorGrid, priorind)
        ## posterior data cube
        postCube = post(likeCube, priorCube)

        margpostAr, margpostFeH, margpostMr = postprocess(
            Ar1d, FeH1d, Mr1d, catalog, dAr, dFeH, dMr, i, likeCube, postCube, priorCube
        )

        if i in myStars:
            plot_star(
                i,
                catalog,
                margpostAr,
                likeCube,
                priorCube,
                postCube,
                mdLocus,
                xLabel,
                yLabel,
                Mr1d,
                margpostMr,
                FeH1d,
                margpostFeH,
                Ar1d,
            )


def postprocess(Ar1d, FeH1d, Mr1d, catalog, dAr, dFeH, dMr, i, likeCube, postCube, priorCube):
    ## process to get expectation values and uncertainties
    # marginalize and get stats
    margpostMr = {}
    margpostFeH = {}
    margpostAr = {}
    margpostMr[0], margpostFeH[0], margpostAr[0] = getMargDistr3D(priorCube, dMr, dFeH, dAr)
    margpostMr[1], margpostFeH[1], margpostAr[1] = getMargDistr3D(likeCube, dMr, dFeH, dAr)
    margpostMr[2], margpostFeH[2], margpostAr[2] = getMargDistr3D(postCube, dMr, dFeH, dAr)
    # stats
    catalog["MrEst"][i], catalog["MrEstUnc"][i] = getStats(Mr1d, margpostMr[2])
    catalog["FeHEst"][i], catalog["FeHEstUnc"][i] = getStats(FeH1d, margpostFeH[2])
    catalog["ArEst"][i], catalog["ArEstUnc"][i] = getStats(Ar1d, margpostAr[2])
    catalog["MrdS"][i] = Entropy(margpostMr[2]) - Entropy(margpostMr[0])
    catalog["FeHdS"][i] = Entropy(margpostFeH[2]) - Entropy(margpostFeH[0])
    catalog["ArdS"][i] = Entropy(margpostAr[2]) - Entropy(margpostAr[0])
    return margpostAr, margpostFeH, margpostMr


def post(likeCube, priorCube):
    postCube = priorCube * likeCube
    return postCube


def like_and_prior(Ar1d, FeH1d, Mr1d, catalog, chi2map, i, priorGrid, priorind):
    ## likelihood map
    likeGrid = np.exp(-0.5 * chi2map)
    # print('likeGrid:', likeGrid.shape)
    likeCube = likeGrid.reshape(np.size(FeH1d), np.size(Mr1d), np.size(Ar1d))
    # print('likeCube:', likeCube.shape)
    #############################################################################################
    if Ar1d.size > 1:
        dAr = Ar1d[1] - Ar1d[0]
    else:
        dAr = 0.01
    catalog["chi2min"][i] = np.min(chi2map)
    ## generate 3D (Mr, FeH, Ar) prior from 2D (Mr, FeH) prior using uniform prior for Ar
    prior2d = priorGrid[priorind[i]].reshape(np.size(FeH1d), np.size(Mr1d))
    priorCube = make3Dprior(prior2d, np.size(Ar1d))
    return dAr, likeCube, priorCube


def step_1(ArCoeff, ArGridMediumMax, ArGridSmallMax, catalog_Ar):
    # Compute ArMax and ArMin for each star
    ArMax = ArCoeff[0] * catalog_Ar + ArCoeff[1]
    # Select the appropriate ArGrid and locus3D based on ArMax values
    small_mask = ArMax < ArGridSmallMax
    medium_mask = (ArMax >= ArGridSmallMax) & (ArMax < ArGridMediumMax)
    grid_selections = np.select([small_mask, medium_mask], ["ArSmall", "ArMedium"], default="ArLarge")
    return grid_selections


def _setup_bayes3d(ArGridList, MrColumn, catalog, iEnd, iStart, locusData, priorsRootName):
    if iEnd < iStart:
        iStart = 0
        iEnd = np.size(catalog)
    # read maps with priors (and interpolate on the Mr-FeH grid given by locusData, which is same for all stars)
    priorGrid = readPriors(rootname=priorsRootName, locusData=locusData, MrColumn=MrColumn)
    # get prior map indices using observed r band mags
    priorind = getPriorMapIndex(catalog["rmag"])
    # properties of Ar grid for prior and likelihood
    bc = getBayesConstants()
    ArCoeff = {}
    ArCoeff[0] = bc["ArCoeff0"]
    ArCoeff[1] = bc["ArCoeff1"]
    ArCoeff[2] = bc["ArCoeff2"]
    ArCoeff[3] = bc["ArCoeff3"]
    ArCoeff[4] = bc["ArCoeff4"]
    # color corrections due to dust reddening (for each Ar in the grid for this particular star)
    # for finding extinction, too
    C = lt.extcoeff()
    reddCoeffs = {}
    reddCoeffs["ug"] = C["u"] - C["g"]
    reddCoeffs["gr"] = C["g"] - C["r"]
    reddCoeffs["ri"] = C["r"] - C["i"]
    reddCoeffs["iz"] = C["i"] - C["z"]
    # Mr and FeH 1-D grid properties extracted from locus data (same for all stars)
    xLabel = "FeH"
    yLabel = "Mr"
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    # grid step
    dFeH = FeH1d[1] - FeH1d[0]
    dMr = Mr1d[1] - Mr1d[0]
    # metadata for plotting likelihood maps below
    FeHmin = np.min(FeH1d)
    FeHmax = np.max(FeH1d)
    FeHNpts = FeH1d.size
    MrFaint = np.max(Mr1d)
    MrBright = np.min(Mr1d)
    MrNpts = Mr1d.size
    mdLocus = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
    print("Mr1d=", np.min(Mr1d), np.max(Mr1d), len(Mr1d))
    print("MrBright, MrFaint=", MrBright, MrFaint)
    # setup arrays for holding results
    catalog["MrEst"] = 0.0 * catalog["Mr"] - 99
    catalog["MrEstUnc"] = 0.0 * catalog["Mr"] - 1
    catalog["FeHEst"] = 0.0 * catalog["Mr"] - 99
    catalog["FeHEstUnc"] = 0.0 * catalog["Mr"] - 1
    catalog["ArEst"] = 0.0 * catalog["Mr"] - 99
    catalog["ArEstUnc"] = 0.0 * catalog["Mr"] - 1
    catalog["QrEst"] = 0.0 * catalog["Mr"] - 99
    catalog["QrEstUnc"] = 0.0 * catalog["Mr"] - 1
    catalog["chi2min"] = 0.0 * catalog["Mr"] - 99
    catalog["MrdS"] = 0.0 * catalog["Mr"] - 1
    catalog["FeHdS"] = 0.0 * catalog["Mr"] - 1
    catalog["ArdS"] = 0.0 * catalog["Mr"] - 1
    ### maximum grid values for Ar from master locus
    ArGridSmallMax = np.max(ArGridList["ArSmall"])
    ArGridMediumMax = np.max(ArGridList["ArMedium"])
    return (
        ArCoeff,
        ArGridMediumMax,
        ArGridSmallMax,
        FeH1d,
        Mr1d,
        dFeH,
        dMr,
        iEnd,
        iStart,
        mdLocus,
        priorGrid,
        priorind,
        reddCoeffs,
        xLabel,
        yLabel,
    )


def writeBayesEstimates(catalog, outfile, iStart, iEnd, do3D=False):
    fout = open(outfile, "w")
    if do3D:
        fout.write(
            "      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  ArEst  ArUnc  chi2min    MrdS     FeHdS      ArdS \n"
        )
    else:
        fout.write("      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  chi2min    MrdS     FeHdS \n")
    for i in range(iStart, iEnd):
        r1 = catalog["glon"].iloc[i]
        r2 = catalog["glat"].iloc[i]
        r3 = catalog["FeHEst"].iloc[i]
        r4 = catalog["FeHEstUnc"].iloc[i]
        r5 = catalog["MrEst"].iloc[i]
        r6 = catalog["MrEstUnc"].iloc[i]
        r7 = catalog["chi2min"].iloc[i]
        r8 = catalog["MrdS"].iloc[i]
        r9 = catalog["FeHdS"].iloc[i]
        s = str("%12.8f " % r1) + str("%12.8f  " % r2) + str("%6.2f  " % r3) + str("%5.2f  " % r4)
        s = s + str("%6.2f  " % r5) + str("%5.2f  " % r6)
        if do3D:
            r15 = catalog["ArEst"].iloc[i]
            r16 = catalog["ArEstUnc"].iloc[i]
            r19 = catalog["ArdS"].iloc[i]
            s = s + str("%6.2f  " % r15) + str("%5.2f  " % r16) + str("%5.2f  " % r7) + str("%8.1f  " % r8)
            s = s + str("%8.1f  " % r9) + str("%8.1f  " % r19) + str("%8.0f" % i) + "\n"
        else:
            s = s + str("%5.2f  " % r7) + str("%8.1f  " % r8) + str("%8.1f  " % r9) + "\n"
        fout.write(s)
    fout.close()
