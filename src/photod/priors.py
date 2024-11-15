import numpy as np
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

import photod.plotting as pt

from photod.bayes_constants import BayesConstants


## given 2D numpy array, make a 3D numpy array by replicating it N3rd times
## e.g. for prior.shape = (51, 1502) and N3rd=20, returns (51, 1502, 20)
def make3Dprior(prior, N3rd):
    return np.repeat(prior[:, :, np.newaxis], N3rd, axis=2)


def readPriors(rootname, locusData, MrColumn="Mr", bc = None):
    # TRILEGAL-based maps were pre-computed for this range...
    bc = bc if bc is not None else BayesConstants()
    rmagMin = bc.rmagMin
    rmagMin = bc.rmagMin
    rmagMax = bc.rmagMax
    rmagNsteps = bc.rmagNsteps
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    priors, rmagBinWidth = readPrior(rmagMin, rmagMax, rmagNsteps, rootname)
    if bc.rmagBinWidth != rmagBinWidth:
        raise ValueError(
            f"inconsistency with rmagBinWidth in readPriors (see src/BayesTools.py) {bc['rmagBinWidth']} != {rmagBinWidth}"
        )
    priorGrid = {}
    for rind, r in enumerate(rGrid):
        # interpolate prior map onto locus Mr-FeH grid
        Z = priors[rind]
        Zval = Z["kde"]
        X = Z["xGrid"]
        Y = Z["yGrid"]
        points = np.array((X.flatten(), Y.flatten())).T
        values = Zval.flatten()
        # actual (linear) interpolation
        priorGrid[rind] = griddata(
            points, values, (locusData["FeH"], locusData[MrColumn]), method="linear", fill_value=0
        )

    return priorGrid


def readPrior(rmagMin, rmagMax, rmagNsteps, rootname):
    # read all maps, index by rmag index
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    priors = {}
    for rind, r in enumerate(rGrid):
        extfile = "-%02d" % (rind)
        infile = rootname + extfile + ".npz"
        priors[rind] = np.load(infile)
    rmagBinWidth = priors[0]["metadata"][13]  # volatile
    return priors, rmagBinWidth


# get prior map indices for provided array of observed r band mags
def getPriorMapIndex(rObs, bc = None):
    bc = bc if bc is not None else BayesConstants()
    rmagMin = bc.rmagMin
    rmagMax = bc.rmagMax
    rmagNsteps = bc.rmagNsteps
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    rind = np.arange(rmagNsteps)
    zint = np.interp(rObs, rGrid, rind) + bc.rmagBinWidth
    return zint.astype(int)


def dumpPriorMaps_testing(sample, fileRootname, pix, show2Dmap=False, verbose=True, NrowMax=200000, bayes_constants = None):

    ## data frame called "sample" here must have the following columns: 'FeH', 'Mr', 'rmag'
    labels = ["FeH", "Mr", "rmag"]
    print("sample", type(sample))
    ## numerical model specifications and constants for Bayesian PhotoD method
    bc = bayes_constants if bayes_constants is not None else BayesConstants()
    FeHmin = bc.FeHmin
    FeHmax = bc.FeHmax
    FeHNpts = bc.FeHNpts
    MrFaint = bc.MrFaint
    MrBright = bc.MrBright
    MrNpts = bc.MrNpts
    rmagBinWidth = bc.rmagBinWidth
    rmagMin = bc.rmagMin
    rmagMax = bc.rmagMax
    rmagNsteps = bc.rmagNsteps

    # -------
    metadata = np.array(
        [
            FeHmin,
            FeHmax,
            FeHNpts,
            MrFaint,
            MrBright,
            MrNpts,
            np.mean(sample["glon"]),
            np.mean(sample["glat"]),
            pix.order,
            pix.pixel,
        ]
    )
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    # summary file
    outsumfile = fileRootname + "-SummaryStats.txt"
    foutsum = open(outsumfile, "w")
    foutsum.write(
        " rMin    rMax   Ntotal     Amin   Amed    Amax     pMS      ppMS     pAGB      pWD        pA1       pA2      pA3      pA4       ptnD     ptkD       pH        pB        pMC \n"
    )
    print("Healpix: ", pix, "\n---------------------------------------")
    for rind, r in enumerate(rGrid):
        # r magnitude limits for this subsample
        rMin = r - rmagBinWidth
        rMax = r + rmagBinWidth
        # select subsample

        # tS = sample[(sample[labels[2]]>rMin)&(sample[labels[2]]<rMax)]

        # alternative version
        rfilter = (sample.loc[:, "rmag"] > rMin) & (sample.loc[:, "rmag"] < rMax)
        tS = sample[rfilter]
        # print("tS", type(tS))

        if verbose:
            print("r=", rMin, "to", rMax, "N=", len(sample), "Ns=", len(tS))
        if len(tS) > NrowMax:
            tSmap = tS.sample(n=NrowMax)
            if verbose:
                print("subsampled for 2D map to n=", NrowMax)
        else:
            tSmap = tS

        # this is work horse, where data are binned and map arrays produced
        # it takes about 2 mins on Mac OS Apple M1 Pro
        # so about 70 hours for 2,000 healpix samples
        # maps per healpix are about 2 MB, total 4 GB
        if len(tSmap) < 3:  # Changed to 3 in an attempt to avoid LinAlgErr
            # print("tSmap", tSmap)
            print("ERROR: no data to make map (see dumpPriorMaps)")
            # xGrid, yGrid, Z = 0
            # xGrid = yGrid = Z = np.zeros(0) # <-- ispravio, bolje bi bilo staviti pass
            # pass  # <-- dodao pass. još bolje bi bilo dodati ga iznad.
        else:
            xGrid, yGrid, Z = get2Dmap(tSmap, labels, metadata)
            # display for sanity tests, it can be turned off
            if show2Dmap:
                pt.show2Dmap(xGrid, yGrid, Z, metadata, labels[0], labels[1], logScale=True)
            # store this particular map (given healpix and r band magnitude slice)
            extfile = "-%02d" % (rind)  # re.findall(r'\d+', input_string)
            # it is assumed that fileRootname knows which healpix is being processed,
            # as well as the magnitude range specified by rmagMin and rmagMax
            outfile = fileRootname + extfile + ".npz"
            Zrshp = Z.reshape(xGrid.shape)
            tSsize = np.size(tS)  # Total size of array: rows*cols
            mdExt = np.concatenate(
                (metadata, np.array([rmagMin, rmagMax, rmagNsteps, rmagBinWidth, tSsize, r]))
            )
            np.savez(outfile, xGrid=xGrid, yGrid=yGrid, kde=Zrshp, metadata=mdExt, labels=labels)
            ## summary info
            s1 = str("%5.1f  " % rMin) + str("%5.1f  " % rMax) + str("%10.0f  " % len(tS))  # Ntotal!
            # dust extinction information
            A1 = np.min(tS["Av"])
            A2 = np.median(tS["Av"])
            A3 = np.max(tS["Av"])
            s2 = str("%6.2f " % A1) + str("%6.2f " % A2) + str("%6.2f " % A3)
            # evolutionary stats
            df = tS
            dfTms = df[df["label"] == 1]
            dfTpms = df[
                (df["label"] > 1) & (df["label"] < 7)
            ]  # according to TRILEGAL labels column these would be SGB, RGB, CHeB (4,5,6). Nomenclature is confusing because "pms" usually corresponds to pre-main sequence. Given the cuts, this would rather apply to post main sequence.
            dfTagb = df[(df["label"] > 6) & (df["label"] < 9)]  # EAGBs & TPAGBs
            dfTwd = df[df["label"] == 9]  # PAGBs + WDs
            Tms = len(dfTms)
            Tpms = len(dfTpms)
            Tagb = len(dfTagb)
            Twd = len(dfTwd)
            # probabilities
            Ttotal = Tms + Tpms + Tagb + Twd
            if Ttotal > 0:
                pms = Tms / Ttotal
                ppms = Tpms / Ttotal
                pagb = Tagb / Ttotal
                pwd = Twd / Ttotal
            else:
                pms = ppms = pagb = pwd = -1
            s3 = str("%8.3e " % pms) + str("%8.3e " % ppms) + str("%8.3e " % pagb) + str("%8.3e " % pwd)
            # age distribution
            dfA1 = df[df["logage"] < 7]
            dfA2 = df[df["logage"] < 8]
            dfA3 = df[df["logage"] < 9]
            dfA4 = df[df["logage"] < 10]
            Atotal = len(df)
            pA1 = len(dfA1) / Atotal
            pA2 = len(dfA2) / Atotal
            pA3 = len(dfA3) / Atotal
            pA4 = len(dfA4) / Atotal
            s4 = str("%8.3e " % pA1) + str("%8.3e " % pA2) + str("%8.3e " % pA3) + str("%8.3e " % pA4)
            # galactic components
            dfC1 = df[df["comp"] == 1]
            dfC2 = df[df["comp"] == 2]
            dfC3 = df[df["comp"] == 3]
            dfC4 = df[df["comp"] == 4]
            dfC5 = df[df["comp"] == 5]
            Ctotal = len(df)
            pC1 = len(dfC1) / Ctotal
            pC2 = len(dfC2) / Ctotal
            pC3 = len(dfC3) / Ctotal
            pC4 = len(dfC4) / Ctotal
            pC5 = len(dfC5) / Ctotal
            s5 = (
                str("%8.3e " % pC1)
                + str("%8.3e " % pC2)
                + str("%8.3e " % pC3)
                + str("%8.3e " % pC4)
                + str("%8.3e " % pC5)
            )
            s = s1 + s2 + s3 + s4 + s5 + "\n"
            foutsum.write(s)
    foutsum.close()


def get2Dmap(sample, labels, metadata):
    data = np.vstack([sample[labels[0]], sample[labels[1]]])
    try:
        kde = gaussian_kde(data)
        # evaluate on a regular grid
        xMin = metadata[0]  # FeHmin
        xMax = metadata[1]  # FeHMax
        nXbin = int(metadata[2])  # FeHNpts ??? why
        yMin = metadata[3]
        yMax = metadata[4]
        nYbin = int(metadata[5])
        xgrid = np.linspace(xMin, xMax, nXbin)
        ygrid = np.linspace(yMin, yMax, nYbin)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        return (Xgrid, Ygrid, Z)
    except:
        # print("LinAlgError", metadata)
        print("Error", metadata)
        print(metadata)
        pass
