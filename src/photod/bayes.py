import numpy as np

import photod.locus as lt
from photod.plotting import show3Flat2Dmaps, showCornerPlot3, showMargPosteriors3D, showQrCornerPlot
from photod.priors import getBayesConstants, getPriorMapIndex, make3Dprior, readPriors
from photod.stats import Entropy, getMargDistr3D, getStats


def makeBayesEstimates3D(
    catalog,
    fitColors,
    locusData,
    locus3DList,
    ArGridList,
    priorsRootName,
    outfile,
    iStart=0,
    iEnd=-1,
    myStars=[],
    verbose=False,
    MrColumn="Mr",
):

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

    #######################################################################
    ## loop over all stars (can be trivially parallelized)
    for i in range(iStart, iEnd):
        if verbose:
            if int(i / 10000) * 10000 == i:
                print("working on star", i)

        ######## NEED TO SORT ALL INPUT AND WRITE & CALL makeLikelihoodCube()
        # chi2min, likeCube = makeLikelihoodCube(...inputs...)
        # input: i, catalog, ArCoeff,

        ### produce likelihood array for this star, using provided locus color model

        ### THIS BLOCK USES A LIST OF THREE 3D MODEL LOCII, with 3 different resolutions for Ar
        # note that here true Ar is used (catalog['Ar'][i]); for real stars, need to use SFD or another extinction map
        # ArMax = ArCoeff[0]*catalog['Ar'][i] + ArCoeff[1]
        # ArMin = ArCoeff[3]*catalog['Ar'][i] + ArCoeff[4]

        ArMax = ArCoeff[0] * catalog["Ar"].iloc[i] + ArCoeff[1]
        ArMin = ArCoeff[3] * catalog["Ar"].iloc[i] + ArCoeff[4]

        # depending on ArMax, pick the adequate Ar resolution of locus3D
        if ArMax < ArGridSmallMax:
            ArGrid = ArGridList["ArSmall"]
            locus3D = locus3DList["ArSmall"]
            # print('selected ArSmall locus3D with', len(locus3D), ' elements')
        else:
            if ArMax < ArGridMediumMax:
                ArGrid = ArGridList["ArMedium"]
                locus3D = locus3DList["ArMedium"]
                # print('selected ArMedium locus3D with', len(locus3D), ' elements')
            else:
                ArGrid = ArGridList["ArLarge"]
                locus3D = locus3DList["ArLarge"]
                # print('selected ArLarge locus3D with', len(locus3D), ' elements')
        # subselect from chosen 3D locus (simply to have fewer Ar points and thus faster execution)
        Ar1d = ArGrid[(ArGrid >= ArMin) & (ArGrid <= ArMax)]

        # simply limit by maximum plausible extinction Ar
        # print('trying to select from locus3D with', len(locus3D), ' elements, ArMax=', ArMax)
        locus3Dok = locus3D[(locus3D["Ar"] >= ArMin) & (locus3D["Ar"] <= ArMax)]
        # print('selected locus3Dok with', len(locus3Dok), ' elements, from locus3D with', len(locus3D))

        ##### the main Bayes block
        ## compute chi2 map using provided 3D model locus (locus3Dok) ##
        # return i, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, FeH1d, Mr1d, Ar1d
        chi2map = lt.getPhotoDchi2map3D(
            i, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, masterLocus=True
        )
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
        ## posterior data cube
        postCube = priorCube * likeCube

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

        # for testing and illustration
        if i in myStars:
            # plot
            FeHStar = catalog["FeH"][i]
            MrStar = catalog["Mr"][i]
            ArStar = catalog["Ar"][i]
            indA = np.argmax(margpostAr[2])
            show3Flat2Dmaps(
                priorCube[:, :, indA],
                likeCube[:, :, indA],
                postCube[:, :, indA],
                mdLocus,
                xLabel,
                yLabel,
                logScale=True,
                x0=FeHStar,
                y0=MrStar,
            )
            showMargPosteriors3D(
                Mr1d,
                margpostMr,
                "Mr",
                "p(Mr)",
                FeH1d,
                margpostFeH,
                "FeH",
                "p(FeH)",
                Ar1d,
                margpostAr,
                "Ar",
                "p(Ar)",
                MrStar,
                FeHStar,
                ArStar,
            )
            # these show marginal 2D and 1D distributions (aka "corner plot")
            showCornerPlot3(
                postCube,
                Mr1d,
                FeH1d,
                Ar1d,
                mdLocus,
                xLabel,
                yLabel,
                logScale=True,
                x0=FeHStar,
                y0=MrStar,
                z0=ArStar,
            )
            # Qr vs. FeH posterior and marginal 1D distributions for Qr and FeH
            Qr1d, margpostQr = showQrCornerPlot(
                postCube, Mr1d, FeH1d, Ar1d, x0=FeHStar, y0=MrStar, z0=ArStar, logScale=True
            )
            catalog["QrEst"][i], catalog["QrEstUnc"][i] = getStats(Qr1d, margpostQr)
            # basic info
            print(" *** 3D Bayes results for star i=", i)
            print("r mag:", catalog["rmag"][i], "g-r:", catalog["gr"][i], "chi2min:", catalog["chi2min"][i])
            print("Mr: true=", MrStar, "estimate=", catalog["MrEst"][i], " +- ", catalog["MrEstUnc"][i])
            print("FeH: true=", FeHStar, "estimate=", catalog["FeHEst"][i], " +- ", catalog["FeHEstUnc"][i])
            print("Ar: true=", ArStar, "estimate=", catalog["ArEst"][i], " +- ", catalog["ArEstUnc"][i])
            print(
                "Qr: true=", MrStar + ArStar, "estimate=", catalog["QrEst"][i], " +- ", catalog["QrEstUnc"][i]
            )
            print("Mr drop in entropy:", catalog["MrdS"][i])
            print("FeH drop in entropy:", catalog["FeHdS"][i])
            print("Ar drop in entropy:", catalog["ArdS"][i])

    # store results
    writeBayesEstimates(catalog, outfile, iStart, iEnd, do3D=True)


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
