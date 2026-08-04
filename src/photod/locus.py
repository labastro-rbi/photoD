import numpy as np
from astropy.table import Table
from scipy.spatial import KDTree
from scipy.interpolate import interpn


def LSSTsimsLocus(fixForStripe82=True, datafile="", colnames = ["Mr", "FeH", "ug", "gr", "ri", "iz", "zy"]):
    ## Mr, as function of [Fe/H], along the SDSS/LSST stellar
    ## for more details see the file header

    if datafile == "":
        datafile = "../../data/MSandRGBcolors_v1.3.txt"
    LSSTlocus = Table.read(datafile, format="ascii", names=colnames)
    LSSTlocus["gi"] = LSSTlocus["gr"] + LSSTlocus["ri"]
    if fixForStripe82:
        print("Fixing input Mr-FeH-colors grid to agree with the SDSS v4.2 catalog")
        # for SDSS v4.2 catalog, see: http://faculty.washington.edu/ivezic/sdss/catalogs/stripe82.html
        # implement empirical corrections for u-g and i-z colors to make it better agree with the SDSS v4.2 catalog
        # fix u-g: slightly redder for progressively redder stars and fixed for gi>giMax
        ugFix = LSSTlocus["ug"] + 0.02 * (2 + LSSTlocus["FeH"]) * LSSTlocus["gi"]
        giMax = 1.8
        ugMax = 2.53 + 0.13 * (1 + LSSTlocus["FeH"])
        LSSTlocus["ug"] = np.where(LSSTlocus["gi"] > giMax, ugMax, ugFix)
        # fix i-z color: small offsets as functions of r-i and [Fe/H]
        off0 = 0.08
        off2 = -0.09
        off5 = 0.008
        offZ = 0.01
        Z0 = 2.5
        LSSTlocus["iz"] += off0 * LSSTlocus["ri"] + off2 * LSSTlocus["ri"] ** 2 + off5 * LSSTlocus["ri"] ** 5
        LSSTlocus["iz"] += offZ * (Z0 + LSSTlocus["FeH"])
    return LSSTlocus


## subsample locusData along Mr and FeH grids by factors kMr and kFeH (if both are 1, no subsampling)
### this is the original function that assumes a rectangular grid
def subsampleLocusData(locusData, kMr, kFeH, xLabel = "FeH", yLabel = "Mr", verbose=True):
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    # original grid sizes
    nFeH = FeH1d.size
    nMr = Mr1d.size
    # subsampled grid sizes
    nFeHs = int(nFeH / kFeH)
    nMrs = int(nMr / kMr)
    if verbose:
        print("subsampled locus 2D grid in FeH and Mr from", nFeH, nMr, "to:", nFeHs, nMrs)
        print(nFeHs * kMr + nMrs * kFeH * nMr)
        print(nFeH * nMr, len(locusData))
    subsampled = locusData[:0].copy()
    # now add subsampled rows from the input table
    for j in range(0, nFeHs):
        for i in range(0, nMrs):
            k = i * kMr + j * kFeH * nMr
            subsampled.add_row(locusData[k])
    return subsampled

# This is a new function that can handle any grid shape, but it is a bit slower (made with LLM)
### Not used.
def subsampleLocusData_new(locusData, kMr, kFeH, verbose=True):
    xLabel = "FeH"
    yLabel = "Mr"
    FeHGrid = locusData[xLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    nFeH = FeH1d.size
    nFeHs = int(nFeH / kFeH)

    subsampled = locusData[:0].copy()

    for j in range(0, nFeHs):
        feh = FeH1d[j * kFeH]
        block = locusData[FeHGrid == feh]
        # sort this block by Mr to make subsampling well-defined
        block = block[np.argsort(block[yLabel])]
        nMrBlock = len(block)
        nMrs = int(nMrBlock / kMr)
        for i in range(0, nMrs):
            k = i * kMr
            subsampled.add_row(block[k])

    if verbose:
        print("subsampled locus 2D grid in FeH from", nFeH, "to:", nFeHs)
        print("total subsampled rows:", len(subsampled))

    return subsampled

def get3DmodelList(locusData, fitColors, agressive=False, DSED=False, xLabel = "FeH", yLabel = "Mr", ArFixed=0.2):

    if agressive:
        ## AGRESSIVE
        # for small 3D locus:
        ArGridSmall = np.linspace(0, 0.5, 101)  # step 0.005 mag
        # for medium 3D locus:
        ArGridMedium = np.linspace(0, 2.0, 201)  # step 0.01 mag
        # for large 3D locus:
        ArGridLarge = np.linspace(0, 5.0, 251)  # step 0.02 mag
    else:
        ## LESS AGRESSIVE
        ArGridSmall = np.linspace(0, 0.3, 31)  # step 0.01 mag
        ArGridMedium = np.linspace(0, 0.8, 81)  # step 0.01 mag
        ArGridLarge = np.linspace(0, 2.5, 126)  # step 0.02 mag
    
    ArGridFixed=np.array([ArFixed])
    
    AGList = []
    AGList.append(ArGridSmall)
    AGList.append(ArGridMedium)
    AGList.append(ArGridLarge)
    AGList.append(ArGridFixed)

    ### call the workhorse
    L3Dlist = make3DlocusList(locusData, fitColors, AGList, DSED=DSED, xLabel = xLabel, yLabel = yLabel)

    # repack
    ArGridList = {}
    locus3DList = {}
    locus3DList["ArSmall"] = L3Dlist[0]
    ArGridList["ArSmall"] = ArGridSmall
    locus3DList["ArMedium"] = L3Dlist[1]
    ArGridList["ArMedium"] = ArGridMedium
    locus3DList["ArLarge"] = L3Dlist[2]
    ArGridList["ArLarge"] = ArGridLarge
    locus3DList["ArFixed"] = L3Dlist[3]
    ArGridList["ArFixed"] = ArGridFixed
    return ArGridList, locus3DList


def make3DlocusList(locusData, fitColors, ArGridList, DSED=False, xLabel = "FeH", yLabel = "Mr"):

    # color corrections due to dust reddening
    # for finding extinction, too
    C = extcoeff()
    reddCoeffs = {}
    reddCoeffs["ug"] = C["u"] - C["g"]
    reddCoeffs["gr"] = C["g"] - C["r"]
    reddCoeffs["ri"] = C["r"] - C["i"]
    reddCoeffs["iz"] = C["i"] - C["z"]

    # intrinsic table sizes
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))

    # turn astropy table into numpy array
    locusData["Ar"] = 0 * locusData[yLabel]
    LocusNP = np.array(locusData)
    # the repeating block
    locus3D0 = LocusNP.reshape(np.size(FeH1d), np.size(Mr1d))

    locus3DList = []
    for ArGrid in ArGridList:
        colCorr = {}
        for color in fitColors:
            colCorr[color] = ArGrid * reddCoeffs[color]
        if DSED:
            locus3D = make3DlocusFastDSED(locus3D0, ArGrid, fitColors, colCorr, FeH1d, Mr1d)
        else:
            locus3D = make3DlocusFast(locus3D0, ArGrid, fitColors, colCorr, FeH1d, Mr1d)
        locus3DList.append(locus3D)
    return locus3DList


def make3DlocusFastDSED(locus3D0, ArGrid, colors, colorCorrection, FeH1d, Mr1d):

    N3rd = np.size(ArGrid)
    locus3D = np.repeat(locus3D0[:, :, np.newaxis], N3rd, axis=2)
    for i in range(0, np.size(FeH1d)):
        for j in range(0, np.size(Mr1d)):
            for k in range(0, np.size(ArGrid)):
                locus3D[i, j, k][3] = locus3D[i, j, k][3] + colorCorrection["ug"][k]
                locus3D[i, j, k][4] = locus3D[i, j, k][4] + colorCorrection["gr"][k]
                locus3D[i, j, k][5] = locus3D[i, j, k][5] + colorCorrection["ri"][k]
                locus3D[i, j, k][6] = locus3D[i, j, k][6] + colorCorrection["iz"][k]
                locus3D[i, j, k][9] = ArGrid[k]
    return locus3D


## VOLATILE: assumes order of colors in locus3D0 (that must be consistent with colCorr
##      NB IT WILL BREAK WHEN ANOTHER COLOR IS ADDED!  (e.g. z-y for LSST data)
## given 2D numpy array, make a 3D numpy array by replicating it for each element
## in ArGrid and apply reddening corrections
## n.b. colors is not used (place holder to fix VOLATILE problem...)
def make3DlocusFast(locus3D0, ArGrid, colors, colorCorrection, FeH1d, Mr1d):

    N3rd = np.size(ArGrid)
    locus3D = np.repeat(locus3D0[:, :, np.newaxis], N3rd, axis=2)
    for i in range(0, np.size(FeH1d)):
        for j in range(0, np.size(Mr1d)):
            for k in range(0, np.size(ArGrid)):
                locus3D[i, j, k][2] = locus3D[i, j, k][2] + colorCorrection["ug"][k]
                locus3D[i, j, k][3] = locus3D[i, j, k][3] + colorCorrection["gr"][k]
                locus3D[i, j, k][4] = locus3D[i, j, k][4] + colorCorrection["ri"][k]
                locus3D[i, j, k][5] = locus3D[i, j, k][5] + colorCorrection["iz"][k]
                locus3D[i, j, k][8] = ArGrid[k]
    return locus3D


def extcoeff():
    ## coefficients to correct for ISM dust (for S82 from Berry+2012, Table 1)
    ## extcoeff(band) = A_band / A_r
    extcoeff = {}
    extcoeff["u"] = 1.810
    extcoeff["g"] = 1.400
    extcoeff["r"] = 1.000  # by definition
    extcoeff["i"] = 0.759
    extcoeff["z"] = 0.561
    return extcoeff


def readTRILEGALLSDB(trilegal):
    ### NOTE THAT THIS IS NO LONGER NEEDED AS TRILEGAL IS IMPORTED INTO HIPSCAT WITH COLUMN NAMES FIXED, AND THE REQUIRED COLUMNS ADDED!!!!
    colnames = [
        "glon",
        "glat",
        "comp",
        "logage",
        "FeH",
        "DM",
        "Av",
        "logg",
        "gmag",
        "rmag",
        "imag",
        "umag",
        "zmag",
        "label",
    ]
    # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
    # logage with age in years
    # DM = m-M is called true distance modulus in DalTio+(2022), so presumably extinction is not included
    # and thus Mr = rmag - Ar - DM
    ## read TRILEGAL simulation (per healpix, as extracted by Dani, ~1-2M stars)
    # trilegal = Table.read(infile, format='ascii', names=colnames) <<-- replaced with pd.read_csv
    trilegal = trilegal[colnames].copy()
    # dust extinction: Berry+ give Ar = 2.75E(B-V) and DalTio+ used Av=3.10E(B-V)
    trilegal.loc[:, "Ar"] = 2.75 * trilegal.loc[:, "Av"] / 3.10
    C = extcoeff()
    # correcting colors for extinction effects
    trilegal.loc[:, "ug"] = (
        trilegal.loc[:, "umag"] - trilegal.loc[:, "gmag"] - (C["u"] - C["g"]) * trilegal.loc[:, "Ar"]
    )
    trilegal.loc[:, "gr"] = (
        trilegal.loc[:, "gmag"] - trilegal.loc[:, "rmag"] - (C["g"] - C["r"]) * trilegal.loc[:, "Ar"]
    )
    trilegal.loc[:, "ri"] = (
        trilegal.loc[:, "rmag"] - trilegal.loc[:, "imag"] - (C["r"] - C["i"]) * trilegal.loc[:, "Ar"]
    )
    trilegal.loc[:, "iz"] = (
        trilegal.loc[:, "imag"] - trilegal.loc[:, "zmag"] - (C["i"] - C["z"]) * trilegal.loc[:, "Ar"]
    )
    trilegal.loc[:, "gi"] = trilegal.loc[:, "gr"] + trilegal.loc[:, "ri"]
    return trilegal


def getPhotoDchi2map3D(i, colors, colorReddCoeffs, data2fit, locus, ArCoeff, masterLocus=True):

    # first adopt, or generate, 3D model locus
    if masterLocus:
        locus3D = locus
    else:
        # extend 2D Mr-FeH grid in zero-reddening locus (astropy Table), to a 3D color grid by
        # adding reddening grid to each entry in locus (which corresponds to ArGrid[0] = 0)
        ArMax = ArCoeff[0] * data2fit["Ar"][i] + ArCoeff[1]
        nArGrid = int(ArMax / ArCoeff[2]) + 1
        if nArGrid > 1000:
            print("resetting nArGrid to 1000 in getPhotoDchi2map3D, from:", nArGrid)
            nArGrid = 1000
        if 1:
            ArGrid = np.linspace(0, ArMax, nArGrid)
        else:
            # this is for testing performance when Ar prior is delta function centered on true value
            ArGrid = np.linspace(data2fit["Ar"][i], data2fit["Ar"][i], 1)

        # color corrections due to dust reddening (for each Ar in the grid for this particular star)
        colorCorrection = {}
        for color in colors:
            colorCorrection[color] = ArGrid * colorReddCoeffs[color]
        locus3D = make3Dlocus(locus, ArGrid, colors, colorCorrection)

    # set up colors for fitting (for this star specified by input "i")
    ObsColor = {}
    ObsColorErr = {}
    for color in colors:
        # print('    color=', color)
        # ObsColor[color] = data2fit[color][i]
        ObsColor[color] = data2fit[color].iloc[i]
        errname = color + "Err"
        # ObsColorErr[color] = data2fit[errname][i]
        ObsColorErr[color] = data2fit[errname].iloc[i]

    ## return chi2map (data cube) for each grid point in locus3D
    if masterLocus:
        return getLocusChi2colors(colors, locus3D, ObsColor, ObsColorErr)
    else:
        return ArGrid, getLocusChi2colors(colors, locus3D, ObsColor, ObsColorErr)


# given a grid of model colors, Mcolors, compute chi2
# for a given set of observed colors Ocolors, with errors Oerrors
# colors to be used in chi2 computation are listed in colorNames
# Mcolors is astropy Table
def getLocusChi2colors(colorNames, Mcolors, Ocolors, Oerrors):
    chi2 = 0 * Mcolors[colorNames[0]]
    for color in colorNames:
        chi2 += ((Ocolors[color] - Mcolors[color]) / Oerrors[color]) ** 2
    return chi2


### WHY IS THIS CODE SCALING WITH THE SQUARE OF ArGrid LENGTH???
# replace each row in locus (astropy Table) with np.size(ArGrid) rows where colors in colors
# are reddened using the values in colCorr and return the resulting astropy Table
def make3Dlocus(locus, ArGrid, colors, colCorr):

    # initialize the first block of 3D table that corresponds to Ar=0 and the input table
    locus3D = Table((locus["Mr"], locus["FeH"]), copy=True)
    for color in colors:
        locus3D.add_column(locus[color])
    # the first point in Ar grid is usually, but NOT necessarily, equal to 0
    locus3D["Ar"] = 0 * locus3D["Mr"] + ArGrid[0]
    for color in colors:
        locus3D[color] = locus3D[color] + colCorr[color][0]

    # loop over all >0 reddening values
    for k in range(1, np.size(ArGrid)):
        # new block, start with a copy of the input table
        locusAr = Table((locus["Mr"], locus["FeH"]), copy=True)
        # add a column with the corresponding value of Ar
        locusAr["Ar"] = 0 * locusAr["Mr"] + ArGrid[k]
        # and now redden zero-reddening colors with provided reddening corrections
        cRed = {}
        for color in colors:
            cRed[color] = locus[color] + colCorr[color][k]
            locusAr.add_column(cRed[color])
        # now vstack the segment for this Ar value to locus3D table:
        locus3D = np.vstack([locus3D, locusAr])

    return locus3D


##Below are old functions used to simulate a catalog from trilegal data
    
def with_kdtree(x_model: np.ndarray, x_data, y_model, y_data):
    tree = KDTree(np.stack([x_model, y_model], axis=-1))
    return tree.query(np.stack([x_data, y_data], axis=-1))

def getColorsFromMrFeHDSED(L, Lvalues, colors='', Mr_label='Mr'):
    # L is an astropy Table, Lvalues a Pandas DataFrame
    # Prebaciti sve u numpy pa probati vrtiti kao loop
    # taj kod zapravo nije ni bitan za LSST jer sada se koristi samo zato da se poprave boje koje nisu dobre u TRILEGALu
    # Teoretski se to može i ignorirati i uzeti smao TRILEGAL boje
    SDSScolors = ['ug', 'gr', 'ri', 'iz']
    if not colors:
        colors = SDSScolors
    # Calculate squared distances using vectorized operations
    ## distSq_Mr = ((L['Mr'][:, np.newaxis] - Lvalues['Mr'].values) ** 2) / 0.01 ** 2
    ## distSq_FeH = ((L['FeH'][:, np.newaxis] - Lvalues['FeH'].values) ** 2) / 0.1 ** 2
    ## distSq_total = distSq_Mr + distSq_FeH

    # Find indices of minimum distances for each row
    ## min_indices = np.argmin(distSq_total, axis=0)

    min_indices = with_kdtree(L[Mr_label], Lvalues[Mr_label], L['FeH'], Lvalues['FeH'])[1]
    
    # Assign values to Lvalues based on minimum distances
    Lvalues['{}Assigned'.format(Mr_label)] = L[Mr_label][min_indices].data
    for c in colors:
        Lvalues[c] = L[c][min_indices].data

    return Lvalues

def getLSSTm5(data, depth='coadd', magVersion=False, suffix=''):
    # temporary: only use SDSS colors
    bandpasses = ['u', 'g', 'r', 'i', 'z']
    # from https://iopscience.iop.org/article/10.3847/1538-4365/ac3e72
    coaddm5 = {}
    coaddm5['u'] = 25.73
    coaddm5['g'] = 26.86
    coaddm5['r'] = 26.88
    coaddm5['i'] = 26.34
    coaddm5['z'] = 25.63
    coaddm5['y'] = 24.87
    singlem5 = {}
    singlem5['u'] = 23.50
    singlem5['g'] = 24.44 
    singlem5['r'] = 23.98 
    singlem5['i'] = 23.41
    singlem5['z'] = 22.77
    singlem5['y'] = 22.01
    gg = {}
    gg['u'] = 0.038 
    gg['g'] = 0.039 
    gg['r'] = 0.039 
    gg['i'] = 0.039 
    gg['z'] = 0.039 
    gg['y'] = 0.039   
    m5 = {}
    for b in bandpasses:
        if (depth=='coadd'):
            m5[b] = coaddm5[b] 
        else:
            m5[b] = singlem5[b] 
    mags = {}
    for b in bandpasses:
        if (magVersion):
            mags[b] = data[b+'mag'+suffix]
        else:
            mags[b] = data[b]
    errors = {}
    for b in bandpasses:
        x = 10**(0.4*(mags[b]-m5[b]))
        errors[b] = np.sqrt(0.005**2 + (0.04-gg[b])*x + gg[b]*x**2)
    return errors

### this inverts the error(mag) relation from getLSSTm5 and returns errors for provided magnitudes
### N.B. getLSSTm5 also assumes SDSS bandpasses (that is, no y band) 
def getLSSTm5err(mags, depth='coadd'):
    # temporary: only use SDSS colors (no y band)
    bandpasses = ['u', 'g', 'r', 'i', 'z']
    # arrays for interpolation
    magGrid = np.linspace(10, 30, 2001)  # 0.01 mag steps
    magData = {}
    for b in bandpasses:
        magData[b] = magGrid
    errGrid = getLSSTm5(magData, depth)
    # now interpolate to get errors 
    errors = {}
    for b in bandpasses:
        errors[b] = np.interp(mags[b], magGrid, errGrid[b]) 
    return errors

## given Bayes estimates FeHEst and MrEst, where MrEst is really tLoc, variable
## along the locus, use locus info about Mr = func(tLoc, FeH), to get the true
## meaningful MrEst 
def getMrFromFeHtLoc_old(Locus, Catalog):
    Catalog['MrTrueEst'] = 0*Catalog['tLoc'] + 89.99
    for j in range(0,len(Catalog)):
        distSq = (Locus['tLoc']-Catalog['tLoc'][j])**2/0.0001 + (Locus['FeH']-Catalog['FeHEst'][j])**2/0.01
        Catalog['MrTrueEst'][j] = Locus['MrTrue'][np.argmin(distSq)] 
    return

## given Bayes posterior medians FeH_quantile_median and 'tLoc_quantile_median',
## where 'tLoc_quantile_median' is the direct output from running photoD with tLoc variable,
## renamed back to be correct from the 'Mr_quantile_median' in output
## use locus info about Mr = func(tLoc, FeH), to get the true meaningful 'Mr_quantile_median'
## might not work if there are nans
def getMrFromFeHtLoc(df, locus):
    FeH1D=np.unique(locus['FeH'])
    tLoc1D=np.unique(locus['tLoc'])
    df['Mr_quantile_median']=interpn((FeH1D, tLoc1D),
                                      locus['Mr'].reshape(len(FeH1D),len(tLoc1D)),
                                      (df['FeH_quantile_median'],df['tLoc_quantile_median']),
                                    bounds_error=False,)
    return df


def splitMonotonicSegments(tLocVals, MrTrueVals, minSegmentLen=4):
    """
    This is used in assignTLocFromLabel function which transforms Mr to tLoc
    taking trilegal 'label' into account where it is ambiguous.
    Split a (tLoc, Mr_true) curve (for one fixed FeH row, tLoc ascending)
    into monotonic runs. Tiny spurious segments (numerical noise, e.g. from
    rounding in MrTrueTable) shorter than minSegmentLen points get merged
    into their neighbor.

    Returns a list of (startIdx, endIdx) index pairs into tLocVals, ordered
    by tLoc ascending (i.e. NOT yet ordered by evolutionary sequence --
    that ordering/labeling is done by the caller).
    """
    diffs = np.diff(MrTrueVals)
    signs = np.sign(diffs)
    signs[signs == 0] = signs[signs != 0][0] if np.any(signs != 0) else 1

    breakpoints = [0]
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            breakpoints.append(i)
    breakpoints.append(len(tLocVals) - 1)

    segments = [(breakpoints[i], breakpoints[i + 1]) for i in range(len(breakpoints) - 1)]

    # merge segments shorter than minSegmentLen into the previous one
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < minSegmentLen:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(list(seg))
    return [tuple(s) for s in merged]


def assignTLocFromLabel(
    trilegalCatalog,
    globalParams,
    segmentLabelMap={np.float64(-2.5): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-2.4): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-2.3): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-2.2): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-2.1): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-2.0): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.9): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.8): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.7): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.6): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.5): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.4): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.3): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.2): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.1): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-1.0): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-0.9): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-0.8): {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
         np.float64(-0.7): {2: {0, 1, 2}, 1: {3,4,5,6,7}, 0: {8}},
         np.float64(-0.6): {2: {0, 1, 2}, 1: {3,4,5,6,7}, 0: {8}},      
         np.float64(-0.5): {2: {0, 1, 2}, 1: {3,4,5,6,7}, 0: {8}},
         np.float64(-0.4): {3: {0, 1}, 2: {2}, 1: {3,4,5,6,7}, 0: {8}},
         np.float64(-0.3): {3: {0, 1}, 2: {2}, 1: {3,4,5,6,7}, 0: {8}},
         np.float64(-0.2): {4: {0, 1}, 3: {2}, 2: {3,4,5,6,7}, 1: {8}, 0: {}},
         np.float64(-0.1): {3: {0, 1,2}, 2: {3,4,5,6,7}, 1: {8}, 0: {}},
         np.float64(0.0): {4: {0, 1}, 3: {2}, 2: {3,4,5,6,7}, 1: {8}, 0: {}},     
         np.float64(0.1): {4: {0, 1}, 3: {2}, 2: {3,4,5,6,7}, 1: {8}, 0: {}},
         np.float64(0.2): {3: {0, 1}, 2: {2,3,4,5,6,7}, 1: {8}, 0: {}},
         np.float64(0.3): {5: {0, 1}, 4: {2}, 3: {3,4,5,6,7}, 2: {8}, 1: {}, 0: {}},
         np.float64(0.4): {5: {0, 1}, 4: {2}, 3: {3,4,5,6,7}, 2: {8}, 1: {}, 0: {}},
         np.float64(0.5): {5: {0, 1}, 4: {2}, 3: {3,4,5,6,7}, 2: {8}, 1: {}, 0: {}}},
    turnoffTLoc=4.0,
    starFeHCol="FeH",
    starMrCol="Mr",
    starLabelCol="label",
    newCol="tLoc",
):    
    """
    Function which transforms Mr to tLoc taking trilegal 'label' into account
    where it is ambiguous.

    segmentLabelMap defaults to hardcoded version for LSSTlocus_10Gyr_fix.txt
    """
    FeH1d = globalParams.FeH1d
    tLoc1d = globalParams.Mr1d
    degenMask = tLoc1d <= turnoffTLoc
    tLocDegen = tLoc1d[degenMask]

    isPerFeH = any(isinstance(v, dict) for v in segmentLabelMap.values())

    starFeH = trilegalCatalog[starFeHCol].to_numpy()
    starMr = trilegalCatalog[starMrCol].to_numpy()
    starLabel = trilegalCatalog[starLabelCol].to_numpy()

    tLocOut = np.full(len(trilegalCatalog), np.nan)

    unambigMask = starMr > turnoffTLoc
    tLocOut[unambigMask] = starMr[unambigMask]

    idx = np.clip(np.searchsorted(FeH1d, starFeH), 1, len(FeH1d) - 1)
    left, right = FeH1d[idx - 1], FeH1d[idx]
    feHIdx = np.where(np.abs(starFeH - left) <= np.abs(starFeH - right), idx - 1, idx)

    remaining = ~unambigMask
    ambiguousFallbackCount = 0
    outOfRangeCount = 0

    for i in np.unique(feHIdx[remaining]):
        rowMask = remaining & (feHIdx == i)
        if not np.any(rowMask):
            continue

        MrTrueDegenRow = globalParams.MrTrueTable[i][degenMask]
        segments = splitMonotonicSegments(tLocDegen, MrTrueDegenRow)

        if isPerFeH:
            fehVal = FeH1d[i]
            localMap = segmentLabelMap.get(fehVal, segmentLabelMap.get("default", {}))
        else:
            localMap = segmentLabelMap

        # === NEW: precompute each segment's true (non-clamped) Mr range,
        # and check, per star, which segments it actually falls inside ===
        segRanges = []
        segInterpData = []
        for s, e in segments:
            segTLoc = tLocDegen[s:e + 1]
            segMrTrue = MrTrueDegenRow[s:e + 1]
            order = np.argsort(segMrTrue)
            segMrTrueSorted = segMrTrue[order]
            segTLocSorted = segTLoc[order]
            segRanges.append((segMrTrueSorted[0], segMrTrueSorted[-1]))
            segInterpData.append((segMrTrueSorted, segTLocSorted))

        starIdxThisFeH = np.where(rowMask)[0]
        for starIdx in starIdxThisFeH:
            mrVal = starMr[starIdx]
            inRangeSegs = [
                segIdx for segIdx, (lo, hi) in enumerate(segRanges)
                if lo <= mrVal <= hi
            ]

            if len(inRangeSegs) == 1:
                # unique physical match -- use it regardless of label
                segIdx = inRangeSegs[0]
            elif len(inRangeSegs) > 1:
                # genuinely ambiguous -- fall back to label to disambiguate
                # among only the segments the star could plausibly be on
                labMatches = [
                    segIdx for segIdx in inRangeSegs
                    if starLabel[starIdx] in localMap.get(segIdx, set())
                ]
                if len(labMatches) == 1:
                    segIdx = labMatches[0]
                elif len(labMatches) > 1:
                    segIdx = labMatches[0]  # ties: arbitrary, but flagged below
                    ambiguousFallbackCount += 1
                else:
                    outOfRangeCount += 1
                    continue  # label doesn't match any candidate segment; leave NaN
            else:
                # mrVal outside ALL segments' true ranges at this FeH
                outOfRangeCount += 1
                continue

            segMrTrueSorted, segTLocSorted = segInterpData[segIdx]
            tLocOut[starIdx] = np.interp(mrVal, segMrTrueSorted, segTLocSorted)

    if ambiguousFallbackCount:
        print(f"NOTE: {ambiguousFallbackCount} stars had Mr consistent with multiple "
              "segments AND label matched more than one -- resolved arbitrarily "
              "(first match). Consider refining segmentLabelMap for these cases.")
    if outOfRangeCount:
        print(f"WARNING: {outOfRangeCount} stars' Mr fell outside all segments' true "
              "ranges at their FeH, or label didn't match any in-range candidate -- left unassigned.")

    nUnassigned = np.sum(np.isnan(tLocOut))
    if nUnassigned > 0:
        print(f"WARNING: {nUnassigned} stars got no tLoc match.")

    trilegalCatalog[newCol] = tLocOut
    return trilegalCatalog