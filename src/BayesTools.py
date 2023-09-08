import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import LocusTools as lt 
import PlotTools as pt


### numerical model specifications and constants for Bayesian PhotoD method
def getBayesConstants():
    BayesConst = {}
    BayesConst['FeHmin'] = -2.5       # the range of considered [Fe/H], in priors and elsewhere
    BayesConst['FeHmax'] = 1.0
    BayesConst['FeHNpts'] = 35        # defines the grid step for [Fe/H]
    BayesConst['MrFaint'] = 15.0      # the range of considered Mr, in priors and elsewhere
    BayesConst['MrBright'] = -2.0 
    BayesConst['MrNpts'] = 85         # defines the grid step for [Fe/H]
    BayesConst['rmagBinWidth'] = 0.5  # could be dependent on rmag and larger for bright mags...
    # parameters for pre-computed TRILEGAL-based prior maps
    BayesConst['rmagMin'] = 14
    BayesConst['rmagMax'] = 27
    BayesConst['rmagNsteps'] = 27
    return BayesConst   


### IMPLEMENTATION OF MAP-BASED PRIORS
def getMetadataPriors(priorMap=""):
    if (priorMap==""):
        bc = bt.getBayesConstants()
        FeHmin = bc['FeHmin']
        FeHmin = bc['FeHmin']
        FeHNpts = bc['FeHNpts']
        MrFaint = bc['MrFaint']
        MrBright = bc['MrBright'] 
        MrNpts = bc['MrNpts']
    else:
        FeHmin = priorMap['metadata'][0]
        FeHmax = priorMap['metadata'][1]
        FeHNpts = priorMap['metadata'][2]
        MrFaint = priorMap['metadata'][3]
        MrBright = priorMap['metadata'][4]
        MrNpts = priorMap['metadata'][5]
    return np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])


def get2Dmap(sample, labels, metadata):
    data = np.vstack([sample[labels[0]], sample[labels[1]]])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = int(metadata[2])
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = int(metadata[5])
    xgrid = np.linspace(xMin, xMax, nXbin)
    ygrid = np.linspace(yMin, yMax, nYbin)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    return (Xgrid, Ygrid, Z)

def dumpPriorMaps(sample, rmagMin, rmagMax, rmagNsteps, fileRootname):
    ## numerical model specifications and constants for Bayesian PhotoD method
    bc = bt.getBayesConstants()
    FeHmin = bc['FeHmin']
    FeHmin = bc['FeHmin']
    FeHNpts = bc['FeHNpts']
    MrFaint = bc['MrFaint']
    MrBright = bc['MrBright'] 
    MrNpts = bc['MrNpts']
    rmagBinWidth = bc['rmagBinWidth']
    rmagMin = bc['rmagMin']
    rmagMax = bc['rmagMax'] 
    rmagNsteps = bc['rmagNsteps']  

    # -------
    labels = ['FeH', 'Mr']
    metadata = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    for rind, r in enumerate(rGrid):
        # r magnitude limits for this subsample
        rMin = r - rmagBinWidth
        rMax = r + rmagBinWidth
        # select subsample
        tS = sample[(sample['rmag']>rMin)&(sample['rmag']<rMax)]
        tSsize = np.size(tS)
        print('r=', rMin, 'to', rMax, 'N=', np.size(sample), 'Ns=', np.size(tS))
        # this is work horse, where data are binned and map arrays produced
        # it takes about 2 mins on Mac OS Apple M1 Pro
        # so about 70 hours for 2,000 healpix samples 
        # maps per healpix are about 2 MB, total 4 GB
        xGrid, yGrid, Z = get2Dmap(tS, labels, metadata)
        # display for sanity tests, it can be turned off 
        pt.show2Dmap(xGrid, yGrid, Z, metadata, labels[0], labels[1], logScale=True)
        # store this particular map (given healpix and r band magnitude slice) 
        extfile = "-%02d" % (rind)
        # it is assumed that fileRootname knows which healpix is being processed,
        # as well as the magnitude range specified by rmagMin and rmagMax
        outfile = fileRootname + extfile + '.npz' 
        Zrshp = Z.reshape(xGrid.shape)
        mdExt = np.concatenate((metadata, np.array([rmagMin, rmagMax, rmagNsteps, rmagBinWidth, tSsize, r])))
        np.savez(outfile, xGrid=xGrid, yGrid=yGrid, kde=Zrshp, metadata=mdExt, labels=labels)


def readPriors(rootname, locusData):
    # TRILEGAL-based maps were pre-computed for this range...
    bc = getBayesConstants()
    rmagMin = bc['rmagMin']
    rmagMax = bc['rmagMax'] 
    rmagNsteps = bc['rmagNsteps']  
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    priors, rmagBinWidth = readPrior(rmagMin, rmagMax, rmagNsteps,rootname)
    if (bc['rmagBinWidth'] != rmagBinWidth):
        error('inconsistency with rmagBinWidth in readPriors (see src/BayesTools.py)')
    priorGrid = {}
    for rind, r in enumerate(rGrid):  
        # interpolate prior map onto locus Mr-FeH grid 
        Z = priors[rind]
        Zval = Z['kde']
        X = Z['xGrid']
        Y = Z['yGrid']
        points = np.array((X.flatten(), Y.flatten())).T
        values = Zval.flatten()
        # actual (linear) interpolation
        priorGrid[rind] = griddata(points, values, (locusData['FeH'], locusData['Mr']), method='linear')
    return priorGrid
 
        
def readPrior(rmagMin, rmagMax, rmagNsteps,rootname):
    # read all maps, index by rmag index 
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    priors = {}
    for rind, r in enumerate(rGrid):
        extfile = "-%02d" % (rind)
        infile = rootname + extfile + '.npz' 
        priors[rind] = np.load(infile)
    rmagBinWidth = priors[0]['metadata'][9] # volatile
    return priors, rmagBinWidth


# get prior map indices for provided array of observed r band mags
def getPriorMapIndex(rObs):
    bc = getBayesConstants()
    rmagMin = bc['rmagMin']
    rmagMax = bc['rmagMax'] 
    rmagNsteps = bc['rmagNsteps']  
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    rind = np.arange(rmagNsteps)
    zint = np.interp(rObs, rGrid, rind) + bc['rmagBinWidth']
    return zint.astype(int) 
    
def getMetadataLikelihood(locusData=""):
    if (locusData==""):
        raise Exception("you must specify locus as it is not uniquely determined!")
    else:
        FeHmin = np.min(locusData['FeH'])
        FeHmax = np.max(locusData['FeH'])
        FeHNpts = np.size(np.unique(locusData['FeH']))  
        MrFaint = np.max(locusData['Mr'])
        MrBright = np.min(locusData['Mr'])
        MrNpts = np.size(np.unique(locusData['Mr']))
    return np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])


def pnorm(pdf, dx):
   return pdf/np.sum(pdf)/dx
    
def getStats(x,pdf):
    mean = np.sum(x*pdf)/np.sum(pdf)
    V = np.sum((x-mean)**2*pdf)/np.sum(pdf)
    return mean, np.sqrt(V)

def sigGzi(x):
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))

def getMedianSigG(basicStatsLine):
    med = "%.3f" % basicStatsLine[3]
    sigG = "%.2f" % basicStatsLine[4]
    return [med, sigG, basicStatsLine[0]]

def basicStats(df, colName):
    yAux = df[colName]
    # robust estimate of standard deviation: 0.741*(q75-q25)
    sigmaG = sigGzi(yAux)
    median = np.median(yAux)
    return [np.size(yAux), np.min(yAux), np.max(yAux), median, sigmaG]

def getMargDistr(arr2d, dX, dY):
    margX = np.sum(arr2d, axis=0)
    margY = np.sum(arr2d, axis=1)
    return pnorm(margX, dX), pnorm(margY, dY)

# calculate the kl divergence
def KLdivergence(p, q):
    return np.sum(p*np.log2(p/q))

# calculate the js divergence
def JSdivergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * KLdivergence(p, m) + 0.5 * KLdivergence(q, m)

def Entropy(p):
    pOK = p[p>0]
    return -np.sum(pOK*np.log2(pOK))
    
######  plotting tools to support Bayes method ###### 

def showMargPosteriors(x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, trueX1, trueX2): 
 
    fig, axs = plt.subplots(1,2,figsize=(9,4))
    # plot 
    axs[0].plot(x1d1, margp1[2], 'r', lw=3)
    axs[0].plot(x1d1, margp1[1], 'g')
    axs[0].plot(x1d1, margp1[0], 'b')
   
    axs[0].set(xlabel=xLab1, ylabel=yLab1)
    axs[0].plot([trueX1, trueX1], [0, 1.05*np.max([margp1[0], margp1[2]])], 'k', lw=1)
    meanX1, sigX1 = getStats(x1d1, margp1[2])
    axs[0].plot([meanX1, meanX1], [0, 1.05*np.max([margp1[0], margp1[2]])], '--r')

    
    axs[1].plot(x1d2, margp2[2], 'r', lw=3)
    axs[1].plot(x1d2, margp2[1], 'g')
    axs[1].plot(x1d2, margp2[0], 'b')

    axs[1].set(xlabel=xLab2, ylabel=yLab2)
    axs[1].plot([trueX2, trueX2], [0, 1.05*np.max([margp2[0], margp2[2]])], 'k', lw=1)
    meanX2, sigX2 = getStats(x1d2, margp2[2])
    axs[1].plot([meanX2, meanX2], [0, 1.05*np.max([margp2[0], margp2[2]])], '--r')

    plt.savefig('../plots/margPosteriors.png')
    plt.show() 
 
    
def showPosterior(iS): 
    # grid (same for all stars)
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusRG[xLabel]
    MrGrid = locusRG[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid)) 
    
    ## for selecting prior
    rmagStar = rmagTrue[iS]

    ## for comparison
    MrStar = MrTrue[iS]
    FeHStar = FeHTrue[iS]
    print('rmagStar =', rmagStar, 'true Mr=', MrStar, 'true FeH=',FeHStar)

    ## likelihood info     
    # chi2/likelihood info for this (iS) star
    chi2Grid = chi2RG[iS]
    likeGrid = np.exp(-0.5*chi2Grid)

    # axis limits for likelihood maps
    mdLocusRG = getMetadataLikelihood(locusRG)
    
    # get map indices for observed r band mags
    rObs = np.array([rmagStar, rmagStar]) 
    rind = np.arange(rmagNsteps)
    zint = np.interp(rObs, rGrid, rind) + rmagBinWidth 
    priorind = zint.astype(int)
    
    # interpolate a prior map on locus Mr-FeH grid and show the values as points color-coded by prior
    Z = priors[priorind[0]]
    Zval = Z['kde']
    X = Z['xGrid']
    Y = Z['yGrid']
    points = np.array((X.flatten(), Y.flatten())).T
    values = Zval.flatten()
    # actual (linear) interpolation
    Z0 = griddata(points, values, (locusRG['FeH'], locusRG['Mr']), method='linear')
    
    ## posterior
    posterior = likeGrid * Z0 
    post2d = posterior.reshape(np.size(FeH1d), np.size(Mr1d))
    
    # marginalize and get stats 
    margpostMr = np.sum(post2d, axis=0)
    margpostFeH = np.sum(post2d, axis=1)
    pnorm(margpostMr)
    pnorm(margpostFeH)
    # stats
    meanFeH, stdFeH = getStats(FeH1d, margpostFeH)
    meanMr, stdMr = getStats(Mr1d, margpostMr)
    print('Mr=', meanMr,'+-', stdMr)
    print('FeH=', meanFeH,'+-', stdFeH)
    
    # show plots
    pt.show3Flat2Dmaps(Z0, likeGrid, posterior, mdLocusRG, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar)
    pt.showMargPosteriors(Mr1d, margpostMr, 'Mr', 'p(Mr)', FeH1d, margpostFeH, 'FeH', 'p(FeH)', MrStar, FeHStar)
    
    return  

 
def makeBayesEstimates(catalog, fitColors, locusData, priorsRootName, outfile, iStart=0, iEnd=-1, myStars=[], verbose=False):

    if (iEnd < iStart):
        iStart = 0
        iEnd = np.size(catalog)
     
    # read maps with priors (and interpolate on the locus Mr-FeH grid which is same for all stars)
    priorGrid = readPriors(rootname=priorsRootName, locusData=locusData)
    # get prior map indices using observed r band mags
    priorind = getPriorMapIndex(catalog['rmag'])

    # Mr and FeH 1-D grid properties extracted from locus data (same for all stars)
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusData[xLabel]
    MrGrid = locusData[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid))  
    # grid step
    dFeH = FeH1d[1]-FeH1d[0]
    dMr = Mr1d[1]-Mr1d[0]
    # Mr and DeH axis limits and # of points used with likelihood maps
    mdLocus = getMetadataLikelihood(locusData)

    # setup arrays for holding results
    catalog['MrEst'] = 0*catalog['Mr'] - 99 
    catalog['MrEstUnc'] = 0*catalog['Mr'] -1 
    catalog['FeHEst'] = 0*catalog['Mr'] - 99
    catalog['FeHEstUnc'] = 0*catalog['Mr'] - 1 
    catalog['chi2min'] = 0*catalog['Mr'] + 999
    catalog['MrdS'] = 0*catalog['Mr'] - 1 
    catalog['FeHdS'] = 0*catalog['Mr'] - 1 

    # loop over all stars (could be parallelized) 
    for i in range(iStart, iEnd):
        if (verbose):
            print(' ----------- ')
            print('    i =', i, 'working on star', i)

        ########################################################################
        ### call the main workhorse! 
        ### chi2 map for this star, for provided isochrones: the slowest step! 
        chi2map = lt.getPhotoDchi2map(i, fitColors, catalog, locusData)
        ########################################################################
        catalog['chi2min'][i] = np.min(chi2map)
        
        # likelihood map 
        likeGrid = np.exp(-0.5*chi2map)
        # interpolated prior map onto the same Mr-FeH grid as likelihood map
        prior = priorGrid[priorind[i]] 
        # posterior pdf 
        posterior = likeGrid * prior 
        # reshape
        prior2d = prior.reshape(np.size(FeH1d), np.size(Mr1d))
        like2d = likeGrid.reshape(np.size(FeH1d), np.size(Mr1d))
        post2d = posterior.reshape(np.size(FeH1d), np.size(Mr1d))

        ## process to get expectation values and uncertainties
        # marginalize and get stats 
        margpostMr = {}
        margpostFeH = {}
        margpostMr[0], margpostFeH[0] = getMargDistr(prior2d, dMr, dFeH)
        margpostMr[1], margpostFeH[1] = getMargDistr(like2d, dMr, dFeH) 
        margpostMr[2], margpostFeH[2] = getMargDistr(post2d, dMr, dFeH) 

        # stats
        catalog['MrEst'][i], catalog['MrEstUnc'][i] = getStats(Mr1d, margpostMr[2])
        catalog['FeHEst'][i], catalog['FeHEstUnc'][i] = getStats(FeH1d, margpostFeH[2])
        catalog['MrdS'][i] = Entropy(margpostMr[2]) - Entropy(margpostMr[0])
        catalog['FeHdS'][i] = Entropy(margpostFeH[2]) - Entropy(margpostFeH[0])

        if (i in myStars):
            # plot 
            print(' showing detailed results for star i=', i)
            print('Mr:', catalog['MrEst'][i], ' +- ', catalog['MrEstUnc'][i])
            print('FeH:', catalog['FeHEst'][i], ' +- ', catalog['FeHEstUnc'][i])
            print('Mr drop in entropy:', Entropy(margpostMr[2]) - Entropy(margpostMr[0]))
            print('FeH drop in entropy:', Entropy(margpostFeH[2]) - Entropy(margpostFeH[0]))
            FeHStar = catalog['FeH'][i]
            MrStar = catalog['Mr'][i]
            pt.show3Flat2Dmaps(prior, likeGrid, posterior, mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar)
            showMargPosteriors(Mr1d, margpostMr, 'Mr', 'p(Mr)', FeH1d, margpostFeH, 'FeH', 'p(FeH)', MrStar, FeHStar)

    # store results 
    writeBayesEstimates(catalog, outfile, iStart, iEnd)
    return
 

def writeBayesEstimates(catalog, outfile, iStart, iEnd):
    fout = open(outfile, "w")
    fout.write("      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  chi2min    MrdS     FeHdS \n")
    for i in range(iStart, iEnd):
        r1 = catalog['glon'][i]
        r2 = catalog['glat'][i]
        r3 = catalog['FeHEst'][i]
        r4 = catalog['FeHEstUnc'][i]
        r5 = catalog['MrEst'][i]
        r6 = catalog['MrEstUnc'][i]
        r7 = catalog['chi2min'][i]
        r8 = catalog['MrdS'][i]
        r9 = catalog['FeHdS'][i]
        s = str("%12.8f " % r1) + str("%12.8f  " % r2) + str("%6.2f  " % r3) + str("%5.2f  " % r4)
        s = s + str("%6.2f  " % r5) + str("%5.2f  " % r6) + str("%5.2f  " % r7) 
        s = s + str("%8.1f  " % r8) + str("%8.1f  " % r9) + "\n"
        fout.write(s)             
    fout.close()
    return 



## test Bayes estimates
def checkBayes(infile1, infile2, chi2max=10, umagMax=99.9, chiTest=False, cmd=False, fitQ=False):
    
    ## input simulation
    if (umagMax < 99):
        simsAll = lt.readTRILEGALLSST(inTLfile=infile1, chiTest=chiTest)
        sims = simsAll[simsAll['umag']<umagMax]
        print('from', np.size(simsAll),' selected', np.size(sims))
        ## file with Bayes estimates
        simsBayesAll = lt.readTRILEGALLSSTestimates(infile=infile2)
        # volatile: assumes the same order 
        simsBayes = simsBayesAll[simsAll['umag']<umagMax]
    else:
        sims = lt.readTRILEGALLSST(inTLfile=infile1, chiTest=chiTest)
        ## file with Bayes estimates
        simsBayes = lt.readTRILEGALLSSTestimates(infile=infile2)

    # these are dust-reddened colors and thus dust-extincted magnitudes
    sims['gmag'] = sims['rmag'] + sims['gr']  
    sims['umag'] = sims['gmag'] + sims['ug']
    sims['imag'] = sims['rmag'] - sims['ri']
    sims['zmag'] = sims['imag'] - sims['iz'] 

    ## define quantities for testing
    sims['dMr'] = sims['Mr'] - simsBayes['MrEst']
    sims['dMrNorm'] = sims['dMr'] / simsBayes['MrUnc'] 
    sims['dFeH'] = sims['FeH'] - simsBayes['FeHEst']
    sims['dFeHNorm'] = sims['dFeH'] / simsBayes['FeHUnc'] 
    sims['MrML'] = simsBayes['MrEst']
    sims['FeHML'] = simsBayes['FeHEst']
    sims['chi2min'] = simsBayes['chi2min']
    
    ## dummies - Bayes cannot do Ar (yet)
    sims['dAr'] = 0*sims['Ar'] + 0.1
    sims['dArNorm'] = 0*sims['dAr'] + 0.1
    sims['ArML'] = sims['Ar']
    sims['test_set'] = 1 + 0*sims['Ar']

    # entropy change
    sims['MrdS'] = simsBayes['MrdS'] 
    sims['FeHdS'] = simsBayes['FeHdS'] 

    ## extract testing sample
    simsTest = sims[simsBayes['chi2min']<chi2max]   
    print('after chi2min<', chi2max, 'selection:', np.size(sims), np.size(simsTest))  
    
    plotAll(simsTest, cmd=cmd, fitQ=fitQ)
    return sims, simsTest


def plotAll(dfName, cmd=False, fitQ=False):
    if (0):
        g1 = pt.plot2Dmap(dfName['dMr'], dfName['dFeH'], -1.0, 1.0, 50, -1.0, 1.0, 50, 'dMrML', 'dFeHML')
        g2 = pt.plot2Dmap(dfName['Mr'], dfName['dMr'], 13.0, -1.0, 56, -1.0, 1.0, 40, 'Mr', 'dMr', logScale=False)
        g3 = pt.plot2Dmap(dfName['umag'], dfName['dMr'], 16.0, 29.0, 65, -1.0, 1.0, 40, 'u mag', 'dMr')
        g4 = pt.plot2Dmap(dfName['umag'], dfName['dFeH'], 16.0, 29.0, 65, -1.0, 1.0, 40, 'u mag', 'dMr')
        g5 = pt.plot2Dmap(dfName['FeH'], dfName['dFeH'], -3, 0.5, 70, -1.0, 1.0, 40, '[Fe/H]', 'd[Fe/H]')
        g6 = pt.plot2Dmap(dfName['rmag'], dfName['dFeH'], 16, 27, 70, -1.0, 1.0, 40, 'dAr', 'd[Fe/H]')
        g7 = pt.plot2Dmap(dfName['rmag'], dfName['dMr'], -3, 0.5, 70, -1.0, 1.0, 40, '[Fe/H]', 'd[Fe/H]2')
    
    # 
    if fitQ:
        print('calling qpBcmd')
        pt.qpBcmd(dfName, color='gi', mag='umag', scatter=False)
    else:
        print('calling qpB Mr')
        pt.qpB(dfName, 'dMr', Dname='Mr', cmd=cmd)
        print('calling qpB FeH')
        pt.qpB(dfName, 'dFeH', Dname='FeH', cmd=cmd)
        # pt.qpB(dfName, 'dAr', Dname='Ar', cmd=cmd)

    
## quick comparison of Karlo's NN estimates for Mr, FeH, Ar
def cK(infile1, infile2, sim3=False, simtype='a'):
    ## input simulation
    sims = lt.readTRILEGALLSST(inTLfile=infile1)
    # these are dust-reddened colors and thus dust-extincted magnitudes
    sims['gmag'] = sims['rmag'] + sims['gr']  
    sims['umag'] = sims['gmag'] + sims['ug']
    sims['imag'] = sims['rmag'] - sims['ri']
    sims['zmag'] = sims['imag'] - sims['iz'] 
    ## Karlo's file
    if sim3 == False:
        simsML = lt.readKarloMLestimates(multiMethod=False, inKfile=infile2)  
    else: 
        simsML = lt.readKarloMLestimates3(inKfile=infile2, simtype=simtype) 
    # define quantities for testing
    sims['dMr'] = sims['Mr'] - simsML['simple_single_Mr']
    sims['dMrNorm'] = sims['dMr'] / simsML['simple_single_MrErr'] 
    sims['dFeH'] = sims['FeH'] - simsML['simple_single_FeH']
    sims['dFeHNorm'] = sims['dFeH'] / simsML['simple_single_FeHErr'] 
    sims['dAr'] = sims['Ar'] - simsML['simple_single_Ar']
    sims['dArNorm'] = sims['dAr'] / simsML['simple_single_ArErr'] 
    sims['MrML'] = simsML['simple_single_Mr']
    sims['FeHML'] = simsML['simple_single_FeH']
    sims['ArML'] = simsML['simple_single_Ar']
    sims['test_set'] = simsML['test_set']                           
    ## extract testing sample        
    simsTest = sims[sims['test_set']==1]   
    print(np.size(simsTest))  
    
    plotAll(simsTest)
    return sims, simsTest 
    

