import numpy as np
from astropy.table import Table


def LSSTsimsLocus(fixForStripe82=True, datafile=""): 
    ## Mr, as function of [Fe/H], along the SDSS/LSST stellar 
    ## for more details see the file header
    colnames = ['Mr', 'FeH', 'ug', 'gr', 'ri', 'iz', 'zy']
    if (datafile==""): datafile = '../../data/MSandRGBcolors_v1.3.txt'
    LSSTlocus = Table.read(datafile, format='ascii', names=colnames)
    LSSTlocus['gi'] = LSSTlocus['gr'] + LSSTlocus['ri']
    if (fixForStripe82):
        print('Fixing input Mr-FeH-colors grid to agree with the SDSS v4.2 catalog')
        # for SDSS v4.2 catalog, see: http://faculty.washington.edu/ivezic/sdss/catalogs/stripe82.html
        # implement empirical corrections for u-g and i-z colors to make it better agree with the SDSS v4.2 catalog
        # fix u-g: slightly redder for progressively redder stars and fixed for gi>giMax
        ugFix = LSSTlocus['ug']+0.02*(2+LSSTlocus['FeH'])*LSSTlocus['gi']
        giMax = 1.8
        ugMax = 2.53 + 0.13*(1+LSSTlocus['FeH'])
        LSSTlocus['ug'] = np.where(LSSTlocus['gi']>giMax, ugMax , ugFix)
        # fix i-z color: small offsets as functions of r-i and [Fe/H]
        off0 = 0.08
        off2 = -0.09
        off5 = 0.008
        offZ = 0.01
        Z0 = 2.5
        LSSTlocus['iz'] += off0*LSSTlocus['ri']+off2*LSSTlocus['ri']**2+off5*LSSTlocus['ri']**5
        LSSTlocus['iz'] += offZ*(Z0+LSSTlocus['FeH'])
    return LSSTlocus

## subsample locusData along Mr and FeH grids by factors kMr and kFeH (if both are 1, no subsampling)
def subsampleLocusData(locusData, kMr, kFeH, verbose=True):
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    # original grid sizes
    nFeH = FeH1d.size
    nMr = Mr1d.size
    # subsampled grid sizes
    nFeHs = int(nFeH/kFeH)
    nMrs = int(nMr/kMr)
    if verbose: print('subsampled locus 2D grid in FeH and Mr from', nFeH, nMr, 'to:', nFeHs, nMrs)
    subsampled = locusData[:0].copy()
    # now add subsampled rows from the input table
    for j in range(0,nFeHs):
        for i in range(0,nMrs):
            k = i*kMr + j*kFeH*nMr
            subsampled.add_row(locusData[k]) 
    return subsampled

def get3DmodelList(locusData, fitColors, agressive=False, DSED=False):

    if agressive:
        ## AGRESSIVE 
        # for small 3D locus: 
        ArGridSmall = np.linspace(0,0.5,101)   # step 0.005 mag
        # for medium 3D locus: 
        ArGridMedium = np.linspace(0,2.0,201)  # step 0.01 mag
        # for large 3D locus: 
        ArGridLarge = np.linspace(0,5.0,251)   # step 0.02 mag
    else:
        ## LESS AGRESSIVE
        ArGridSmall = np.linspace(0,0.3,31)    # step 0.01 mag
        ArGridMedium = np.linspace(0,0.8,81)   # step 0.01 mag
        ArGridLarge = np.linspace(0,2.5,126)   # step 0.02 mag

    AGList = []
    AGList.append(ArGridSmall)
    AGList.append(ArGridMedium)
    AGList.append(ArGridLarge)

    ### call the workhorse 
    L3Dlist = make3DlocusList(locusData, fitColors, AGList, DSED=DSED)
    
    # repack
    ArGridList = {}
    locus3DList = {}
    locus3DList['ArSmall'] = L3Dlist[0]
    ArGridList['ArSmall'] = ArGridSmall
    locus3DList['ArMedium'] = L3Dlist[1] 
    ArGridList['ArMedium'] = ArGridMedium
    locus3DList['ArLarge'] = L3Dlist[2] 
    ArGridList['ArLarge'] = ArGridLarge
    return ArGridList, locus3DList

def make3DlocusList(locusData, fitColors, ArGridList, DSED=False):

    # color corrections due to dust reddening 
    # for finding extinction, too
    C = extcoeff()
    reddCoeffs = {}
    reddCoeffs['ug'] = C['u']-C['g']
    reddCoeffs['gr'] = C['g']-C['r']
    reddCoeffs['ri'] = C['r']-C['i']
    reddCoeffs['iz'] = C['i']-C['z']

    # intrinsic table sizes
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))
    
    # turn astropy table into numpy array
    locusData['Ar'] = 0*locusData['Mr']
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
    for i in range(0,np.size(FeH1d)):
        for j in range(0,np.size(Mr1d)):
            for k in range(0,np.size(ArGrid)):
                locus3D[i,j,k][3] = locus3D[i,j,k][3] + colorCorrection['ug'][k] 
                locus3D[i,j,k][4] = locus3D[i,j,k][4] + colorCorrection['gr'][k] 
                locus3D[i,j,k][5] = locus3D[i,j,k][5] + colorCorrection['ri'][k] 
                locus3D[i,j,k][6] = locus3D[i,j,k][6] + colorCorrection['iz'][k] 
                locus3D[i,j,k][9] = ArGrid[k]   
    return locus3D 

## VOLATILE: assumes order of colors in locus3D0 (that must be consistent with colCorr
##      NB IT WILL BREAK WHEN ANOTHER COLOR IS ADDED!  (e.g. z-y for LSST data) 
## given 2D numpy array, make a 3D numpy array by replicating it for each element 
## in ArGrid and apply reddening corrections
## n.b. colors is not used (place holder to fix VOLATILE problem...)
def make3DlocusFast(locus3D0, ArGrid, colors, colorCorrection, FeH1d, Mr1d):

    N3rd = np.size(ArGrid)
    locus3D = np.repeat(locus3D0[:, :, np.newaxis], N3rd, axis=2)
    for i in range(0,np.size(FeH1d)):
        for j in range(0,np.size(Mr1d)):
            for k in range(0,np.size(ArGrid)):
                locus3D[i,j,k][2] = locus3D[i,j,k][2] + colorCorrection['ug'][k] 
                locus3D[i,j,k][3] = locus3D[i,j,k][3] + colorCorrection['gr'][k] 
                locus3D[i,j,k][4] = locus3D[i,j,k][4] + colorCorrection['ri'][k] 
                locus3D[i,j,k][5] = locus3D[i,j,k][5] + colorCorrection['iz'][k] 
                locus3D[i,j,k][8] = ArGrid[k]   
    return locus3D 

def extcoeff():
        ## coefficients to correct for ISM dust (for S82 from Berry+2012, Table 1)
        ## extcoeff(band) = A_band / A_r 
        extcoeff = {}
        extcoeff['u'] = 1.810
        extcoeff['g'] = 1.400
        extcoeff['r'] = 1.000  # by definition
        extcoeff['i'] = 0.759 
        extcoeff['z'] = 0.561 
        return extcoeff 

def readTRILEGALLSDB(trilegal):
    ### NOTE THAT THIS IS NO LONGER NEEDED AS TRILEGAL IS IMPORTED INTO HIPSCAT WITH COLUMN NAMES FIXED, AND THE REQUIRED COLUMNS ADDED!!!!
        colnames = ['glon', 'glat', 'comp', 'logage', 'FeH', 'DM', 'Av', 'logg', 'gmag', 'rmag', 'imag', 'umag', 'zmag', 'label']
        # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
        # logage with age in years
        # DM = m-M is called true distance modulus in DalTio+(2022), so presumably extinction is not included
        # and thus Mr = rmag - Ar - DM 
        ## read TRILEGAL simulation (per healpix, as extracted by Dani, ~1-2M stars)
        # trilegal = Table.read(infile, format='ascii', names=colnames) <<-- replaced with pd.read_csv
        trilegal = trilegal[colnames].copy()
        # dust extinction: Berry+ give Ar = 2.75E(B-V) and DalTio+ used Av=3.10E(B-V)
        trilegal.loc[:, 'Ar'] = 2.75 * trilegal.loc[:, 'Av'] / 3.10   
        C = extcoeff()
        # correcting colors for extinction effects 
        trilegal.loc[:, 'ug'] = trilegal.loc[:, 'umag'] - trilegal.loc[:, 'gmag'] - (C['u'] - C['g'])*trilegal.loc[:, 'Ar']  
        trilegal.loc[:, 'gr'] = trilegal.loc[:, 'gmag'] - trilegal.loc[:, 'rmag'] - (C['g'] - C['r'])*trilegal.loc[:, 'Ar']   
        trilegal.loc[:, 'ri'] = trilegal.loc[:, 'rmag'] - trilegal.loc[:, 'imag'] - (C['r'] - C['i'])*trilegal.loc[:, 'Ar']   
        trilegal.loc[:, 'iz'] = trilegal.loc[:, 'imag'] - trilegal.loc[:, 'zmag'] - (C['i'] - C['z'])*trilegal.loc[:, 'Ar']   
        trilegal.loc[:, 'gi'] = trilegal.loc[:, 'gr'] + trilegal.loc[:, 'ri']
        return trilegal 

def getPhotoDchi2map3D(i, colors, colorReddCoeffs, data2fit, locus, ArCoeff, masterLocus=True):

        # first adopt, or generate, 3D model locus
        if masterLocus:
            locus3D = locus 
        else:
            # extend 2D Mr-FeH grid in zero-reddening locus (astropy Table), to a 3D color grid by
            # adding reddening grid to each entry in locus (which corresponds to ArGrid[0] = 0)
            ArMax = ArCoeff[0]*data2fit['Ar'][i] + ArCoeff[1]
            nArGrid = int(ArMax/ArCoeff[2]) + 1
            if (nArGrid>1000):
                print('resetting nArGrid to 1000 in getPhotoDchi2map3D, from:', nArGrid)
                nArGrid = 1000
            if (1):
                ArGrid = np.linspace(0, ArMax, nArGrid)
            else:
                # this is for testing performance when Ar prior is delta function centered on true value
                ArGrid = np.linspace(data2fit['Ar'][i], data2fit['Ar'][i], 1)

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
            errname = color + 'Err'
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
    chi2 = 0*Mcolors[colorNames[0]]
    for color in colorNames:   
        chi2 += ((Ocolors[color]-Mcolors[color])/Oerrors[color])**2 
    return chi2

### WHY IS THIS CODE SCALING WITH THE SQUARE OF ArGrid LENGTH???    
# replace each row in locus (astropy Table) with np.size(ArGrid) rows where colors in colors
# are reddened using the values in colCorr and return the resulting astropy Table
def make3Dlocus(locus, ArGrid, colors, colCorr):
    
    # initialize the first block of 3D table that corresponds to Ar=0 and the input table 
    locus3D = Table((locus['Mr'], locus['FeH']), copy=True)
    for color in colors: locus3D.add_column(locus[color])
    # the first point in Ar grid is usually, but NOT necessarily, equal to 0
    locus3D['Ar'] = 0*locus3D['Mr'] + ArGrid[0]
    for color in colors:
        locus3D[color] = locus3D[color] + colCorr[color][0]
    
    # loop over all >0 reddening values 
    for k in range(1, np.size(ArGrid)):
        # new block, start with a copy of the input table
        locusAr = Table((locus['Mr'], locus['FeH']), copy=True)
        # add a column with the corresponding value of Ar
        locusAr['Ar'] = 0*locusAr['Mr'] + ArGrid[k]
        # and now redden zero-reddening colors with provided reddening corrections 
        cRed = {}
        for color in colors:
            cRed[color] = locus[color] + colCorr[color][k]
            locusAr.add_column(cRed[color])
        # now vstack the segment for this Ar value to locus3D table:  
        locus3D = np.vstack([locus3D, locusAr])
 
    return locus3D 