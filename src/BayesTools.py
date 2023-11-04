import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import LocusTools as lt 
import PlotTools as pt


### numerical model specifications and constants for Bayesian PhotoD method
def getBayesConstants():
    
    BayesConst = {}
    # parameters for stellar locus (main sequence and red giants) table, as well as priors 
    BayesConst['FeHmin'] = -2.5       # the range of considered [Fe/H], in priors and elsewhere
    BayesConst['FeHmax'] = 1.0
    BayesConst['FeHNpts'] = 36        # defines the grid step for [Fe/H]
    BayesConst['MrFaint'] = 17.0      # the range of considered Mr, in priors and elsewhere
    BayesConst['MrBright'] = -2.0 
    BayesConst['MrNpts'] = 96         # defines the grid step for Mr
    BayesConst['rmagBinWidth'] = 0.5  # could be dependent on rmag and larger for bright mags...
    # parameters for pre-computed TRILEGAL-based prior maps
    BayesConst['rmagMin'] = 14
    BayesConst['rmagMax'] = 27
    BayesConst['rmagNsteps'] = 27
    # parameters for dust extinction grid when fitting 3D (Mr, FeH, Ar)
    # prior for Ar is flat from 0 to (1.3*Ar+0.1) with a step of 0.01 mag ***
    # where 1.3 <= ArCoef0, 0.1 <= ArCoef1 and 0.01 mag <= ArCoef2 are set here (or user supplied)
    BayesConst['ArCoeff0'] = 1.3
    BayesConst['ArCoeff1'] = 0.1
    BayesConst['ArCoeff2'] = 0.01 
    
    return BayesConst   


### IMPLEMENTATION OF MAP-BASED PRIORS
def getMetadataPriors(priorMap=""):
    if (priorMap==""):
        bc = getBayesConstants()
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

def dumpPriorMaps(sample, fileRootname):

    ## data frame called "sample" here must have the following columns: 'FeH', 'Mr', 'rmag'
    labels = ['FeH', 'Mr', 'rmag']

    ## numerical model specifications and constants for Bayesian PhotoD method
    bc = getBayesConstants()
    FeHmin = bc['FeHmin']
    FeHmax = bc['FeHmax']
    FeHNpts = bc['FeHNpts']
    MrFaint = bc['MrFaint']
    MrBright = bc['MrBright'] 
    MrNpts = bc['MrNpts']
    rmagBinWidth = bc['rmagBinWidth']
    rmagMin = bc['rmagMin']
    rmagMax = bc['rmagMax'] 
    rmagNsteps = bc['rmagNsteps']  

    # -------
    metadata = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    for rind, r in enumerate(rGrid):
        # r magnitude limits for this subsample
        rMin = r - rmagBinWidth
        rMax = r + rmagBinWidth
        # select subsample
        tS = sample[(sample[labels[2]]>rMin)&(sample[labels[2]]<rMax)]
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
    priors, rmagBinWidth = readPrior(rmagMin, rmagMax, rmagNsteps, rootname)
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


# probably obsolete
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

def getMargDistr3D(arr3d, dX, dY, dZ):
    margX = np.sum(arr3d, axis=(0,2))
    margY = np.sum(arr3d, axis=(1,2))
    margZ = np.sum(arr3d, axis=(0,1))
    return pnorm(margX, dX), pnorm(margY, dY), pnorm(margZ, dZ)


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

def getQmap(cube, FeH1d, Mr1d, Ar1d):
    Smax = -1
    # interpolate 3D cube(FeH, Mr, Ar) onto Qr=Mr+Ar vs. FeH 2D grid 
    Qmap = 0*cube[:,:,0]
    # Q grid, same size as Mr1d array 
    Qr1d = np.linspace(np.min(Mr1d), (np.max(Ar1d)+np.max(Mr1d)), np.size(Mr1d))
    for i in range(0,np.size(FeH1d)):
        for j in range(0,np.size(Qr1d)):
            # summation
            Ssum = 0.0
            for k in range(0,np.size(Ar1d)):
                Mr = Qr1d[j] - Ar1d[k]
                # now need to get the value of index for this Mr
                jk = np.int((Mr-Mr1d[0])/(Mr1d[1]-Mr1d[0])) 
                if ((jk>=0)&(jk<np.size(Mr1d))):
                    Ssum += cube[i,jk,k]
            Qmap[i,j] = Ssum
    return Qmap, Qr1d


# bt.showQrCornerPlot(postCube, Mr1d, FeH1d, Ar1d, x0=FeHStar, y0=MrStar, z0=ArStar, logScale=True)
def showQrCornerPlot(postCube, Mr1d, FeH1d, Ar1d, x0=-99, y0=-99, z0=-99, logScale=False, cmap='Blues'):

    def oneImage(ax, image, extent, title, showTrue, x0, y0, origin, logScale=True, cmap='Blues'):
        im = image/image.max()
        if (logScale):
            cmap = ax.imshow(im.T,
               origin=origin, aspect='auto', extent=extent,
               cmap=cmap, norm=LogNorm(im.max()/100, vmax=im.max()))
        else:
            cmap = ax.imshow(im.T, origin='upper', aspect='auto', extent=extent, cmap=cmap)
        ax.set_title(title)
        if (showTrue):
            ax.scatter(x0, y0, s=150, c='red', alpha=0.3) 
            ax.scatter(x0, y0, s=40, c='yellow', alpha=0.3) 
        return cmap
    
    # 2-D distribution in the Qr vs. FeH plane
    Qmap, Qr1d = getQmap(postCube, FeH1d, Mr1d, Ar1d)

    # 1-D marginal distribution for Qr
    dFeH = FeH1d[1]-FeH1d[0]
    dQr = Qr1d[1]-Qr1d[0]
    margQr, margFeH = getMargDistr(Qmap, dFeH, dQr)

    # map plotting limits
    xMin = np.min(FeH1d)  
    xMax = np.max(FeH1d)  
    yMin = np.min(Qr1d) 
    yMax = np.max(Qr1d)   

    showTrue = False
    if ((x0>-99)&(y0>-99)):
        showTrue = True
        
    ### plot  
    fig, axs = plt.subplots(1,3,figsize=(10,3))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    myExtent=[xMin, xMax, yMax, yMin]
    cmap = oneImage(axs[0], Qmap, myExtent, '', showTrue, x0, y0+z0, origin='upper', logScale=logScale)
    axs[0].set(xlabel='FeH', ylabel='Qr = Mr + Ar')
    axs[1].plot(Qr1d, margQr, 'r', lw=3)
    axs[1].plot([y0+z0, y0+z0], [0, 1.1*np.max(margQr)], '--k', lw=1)
    axs[1].set(xlabel='Qr', ylabel='p(Qr)')
    axs[2].plot(FeH1d, margFeH, 'r', lw=3)
    axs[2].plot([x0, x0], [0, 1.1*np.max(margFeH)], '--k', lw=1)
    axs[2].set(xlabel='FeH', ylabel='p(FeH)')
    
    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    #cb = fig.colorbar(cmap, ax=cax)
    #if (logScale):
        #cb.set_label("density on log scale")
    #else:
        #cb.set_label("density on linear scale")

    #for ax in axs.flat:
        # ax.set(xlabel=xLab, ylabel=yLab)
        # print('pero')
        
    plt.savefig('../plots/QrCornerPlot.png')
    plt.show()
    return Qr1d, margQr
     



# bt.showCornerPlot3(postCube, Mr1d, FeH1d, Ar1d, mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar, z0=ArStar)
def showCornerPlot3(postCube, Mr1d, FeH1d, Ar1d, md, xLab, yLab, x0=-99, y0=-99, z0=-99, logScale=False, cmap='Blues'):

    def oneImage(ax, image, extent, title, showTrue, x0, y0, origin, logScale=True, cmap='Blues'):
        im = image/image.max()
        if (logScale):
            cmap = ax.imshow(im.T,
               origin=origin, aspect='auto', extent=extent,
               cmap=cmap, norm=LogNorm(im.max()/100, vmax=im.max()))
        else:
            cmap = ax.imshow(im.T, origin='upper', aspect='auto', extent=extent, cmap=cmap)
        ax.set_title(title)
        if (showTrue):
            ax.scatter(x0, y0, s=150, c='red', alpha=0.3) 
            ax.scatter(x0, y0, s=40, c='yellow', alpha=0.3) 
        return cmap

 
    # unpack metadata
    xMin = md[0]  # FeH
    xMax = md[1]
    yMin = md[3]  # Mr
    yMax = md[4]
    zMin = 0      # Ar
    zMax = Ar1d[-1]
       
    #### make 3 marginal (summed) 2-D distributions and 3 1-D marginal distributions 
    # grid steps
    dFeH = FeH1d[1]-FeH1d[0]
    dMr = Mr1d[1]-Mr1d[0]
    dAr = Ar1d[1]-Ar1d[0]
    
    # 1-D marginal distributions
    margMr, margFeH, margAr = getMargDistr3D(postCube, dMr, dFeH, dAr) 

    # 2-D marginal distributions
    # Mr vs. FeH
    im1 = np.sum(postCube, axis=(2))
    # Ar vs. FeH
    im2 = np.sum(postCube, axis=(1))
    # Ar vs. Mr
    im3 = np.sum(postCube, axis=(0))
        
    showTrue = False
    if ((x0>-99)&(y0>-99)):
        showTrue = True
        
    ### plot  
    fig, axs = plt.subplots(3,3,figsize=(12,12))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

    
    # row 1: marginal FeH
    myExtent=[xMin, xMax, yMin, yMax]
    axs[0,0].plot(FeH1d, margFeH, 'r', lw=3)
    axs[0,0].plot([x0, x0], [0, 1.1*np.max(margFeH)], '--k', lw=1)
    axs[0,0].set(xlabel='FeH', ylabel='p(FeH)')
    axs[0,1].set_axis_off()
    axs[0,2].set_axis_off()
  
    # row 2: im1 and marginal Mr
    myExtent=[xMin, xMax, yMin, yMax]
    cmap = oneImage(axs[1,0], im1, myExtent, '', showTrue, x0, y0, origin='upper', logScale=logScale)
    axs[1,0].set(xlabel='FeH', ylabel='Mr')
    axs[1,1].plot(Mr1d, margMr, 'r', lw=3)
    axs[1,1].plot([y0, y0], [0, 1.1*np.max(margMr)], '--k', lw=1)
    axs[1,1].set(xlabel='Mr', ylabel='p(Mr)')
    axs[1,2].set_axis_off()

    # row 3: im2, im3, and marginal Ar
    myExtent=[xMin, xMax, zMin, zMax]
    cmap = oneImage(axs[2,0], im2, myExtent, '', showTrue, x0, z0, origin='lower', logScale=logScale)
    axs[2,0].set(xlabel='FeH', ylabel='Ar')
    myExtent=[yMax, yMin, zMin, zMax]
    cmap = oneImage(axs[2,1], im3, myExtent, '', showTrue, y0, z0, origin='lower', logScale=logScale)
    axs[2,1].set(xlabel='Mr', ylabel='Ar')
    axs[2,2].plot(Ar1d, margAr, 'r', lw=3)
    axs[2,2].plot([z0, z0], [0, 1.1*np.max(margAr)], '--k', lw=1)
    axs[2,2].set(xlabel='Ar', ylabel='p(Ar)')

    
    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    #cb = fig.colorbar(cmap, ax=cax)
    #if (logScale):
        #cb.set_label("density on log scale")
    #else:
        #cb.set_label("density on linear scale")

    #for ax in axs.flat:
        # ax.set(xlabel=xLab, ylabel=yLab)
        # print('pero')
        
    plt.savefig('../plots/cornerPlot3.png')
    plt.show()
     


    
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
 

def showMargPosteriors3D(x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, x1d3, margp3, xLab3, yLab3, trueX1, trueX2, trueX3): 
 
    fig, axs = plt.subplots(1,3,figsize=(12.7,4))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

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

    axs[2].plot(x1d3, margp3[2], 'r', lw=3)
    axs[2].plot(x1d3, margp3[1], 'g')
    axs[2].plot(x1d3, margp3[0], 'b')

    axs[2].set(xlabel=xLab3, ylabel=yLab3)
    axs[2].plot([trueX3, trueX3], [0, 1.05*np.max([margp3[0], margp3[2]])], 'k', lw=1)
    meanX3, sigX3 = getStats(x1d3, margp3[2])
    axs[2].plot([meanX3, meanX3], [0, 1.05*np.max([margp3[0], margp3[2]])], '--r')

    plt.savefig('../plots/margPosteriors3D.png')
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

 
def makeBayesEstimates2D(catalog, fitColors, locusData, priorsRootName, outfile, iStart=0, iEnd=-1, myStars=[], verbose=False):

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
 


## given 2D numpy array, make a 3D numpy array by replicating it N3rd times
## e.g. for prior.shape = (51, 1502) and N3rd=20, returns (51, 1502, 20)
def make3Dprior(prior, N3rd):
    return np.repeat(prior[:, :, np.newaxis], N3rd, axis=2)

## reorder axes to correspond to prior array
def make3Dlikelihood(likeGrid):
    return np.moveaxis(likeGrid, [0, 1, 2], [2, 0, 1]) 

 
def makeBayesEstimates3D(catalog, fitColors, locusData, locus3DList, ArGridList, priorsRootName, outfile, iStart=0, iEnd=-1, myStars=[], verbose=False):

    # this is for prototype testing, set to False for production
    # if set to True, it works only for the star i = 100592  
    restrictLocus = False 
    
    if (iEnd < iStart):
        iStart = 0
        iEnd = np.size(catalog)
     
    # read maps with priors (and interpolate on the Mr-FeH grid given by locusData, which is same for all stars)
    priorGrid = readPriors(rootname=priorsRootName, locusData=locusData)
    # get prior map indices using observed r band mags
    priorind = getPriorMapIndex(catalog['rmag'])
              
    # properties of Ar grid for prior and likelihood
    bc = getBayesConstants()
    ArCoeff = {}
    ArCoeff[0] = bc['ArCoeff0']
    ArCoeff[1] = bc['ArCoeff1'] 
    ArCoeff[2] = bc['ArCoeff2']

    # color corrections due to dust reddening (for each Ar in the grid for this particular star) 
    # for finding extinction, too
    C = lt.extcoeff()
    reddCoeffs = {}
    reddCoeffs['ug'] = C['u']-C['g']
    reddCoeffs['gr'] = C['g']-C['r']
    reddCoeffs['ri'] = C['r']-C['i']
    reddCoeffs['iz'] = C['i']-C['z']

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
    # metadata for plotting likelihood maps below
    FeHmin = np.min(FeH1d)
    FeHmax = np.max(FeH1d)
    FeHNpts = FeH1d.size  
    MrFaint = np.max(Mr1d)
    MrBright = np.min(Mr1d)
    MrNpts = Mr1d.size 
    mdLocus = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])

    # setup arrays for holding results
    catalog['MrEst'] = 0*catalog['Mr'] - 99 
    catalog['MrEstUnc'] = 0*catalog['Mr'] -1 
    catalog['FeHEst'] = 0*catalog['Mr'] - 99
    catalog['FeHEstUnc'] = 0*catalog['Mr'] - 1 
    catalog['ArEst'] = 0*catalog['Mr'] - 99 
    catalog['ArEstUnc'] = 0*catalog['Mr'] -1 
    catalog['QrEst'] = 0*catalog['Mr'] - 99 
    catalog['QrEstUnc'] = 0*catalog['Mr'] -1 
    catalog['chi2min'] = 0*catalog['Mr'] + 999
    catalog['MrdS'] = 0*catalog['Mr'] - 1 
    catalog['FeHdS'] = 0*catalog['Mr'] - 1 
    catalog['ArdS'] = 0*catalog['Mr'] - 1 

    ### maximum grid values for Ar from master locus 
    ArGridSmallMax = np.max(ArGridList['ArSmall'])
    ArGridMediumMax = np.max(ArGridList['ArMedium'])

    #######################################################################
    ## loop over all stars (can be trivially parallelized) 
    for i in range(iStart, iEnd):
        if (verbose):
            if (int(i/10000)*10000 == i): 
                print('working on star', i)

        ######## NEED TO SORT ALL INPUT AND WRITE & CALL makeLikelihoodCube()         
        # chi2min, likeCube = makeLikelihoodCube(...inputs...)    
        # input: i, catalog, ArCoeff, 

                
        ### produce likelihood array for this star, using provided locus color model 
        if (0):
            ### THIS BLOCK GENERATES A 3D MODEL LOCUS FOR EACH STAR (obsolete)
            Ar1d, chi2map = lt.getPhotoDchi2map3D(i, fitColors, reddCoeffs, catalog, locusData, ArCoeff)
            # likelihood map 
            likeGrid = np.exp(-0.5*chi2map)
            L3d = likeGrid.reshape(np.size(Ar1d), np.size(FeH1d), np.size(Mr1d))
            likeCube = make3Dlikelihood(L3d)
        else:
            ### THIS BLOCK USES A LIST OF THREE 3D MODEL LOCII, with 3 different resolutions for Ar
            # note that here true Ar is used (catalog['Ar'][i]); for real stars, need to use SFD or another extinction map
            ArMax = ArCoeff[0]*catalog['Ar'][i] + ArCoeff[1]
            # depending on ArMax, pick the adequate Ar resolution of locus3D
            if (ArMax < ArGridSmallMax):
                ArGrid = ArGridList['ArSmall']
                locus3D = locus3DList['ArSmall']
            else:
                if (ArMax < ArGridMediumMax):
                    ArGrid = ArGridList['ArMedium']
                    locus3D = locus3DList['ArMedium']
                else:
                    ArGrid = ArGridList['ArLarge']
                    locus3D = locus3DList['ArLarge']
            # subselect from chosen 3D locus (simply to have fewer Ar points and thus faster execution) 
            Ar1d = ArGrid[ArGrid<=ArMax]
 
            if restrictLocus: 
                ######### TEST: can we subsample in 3D? ################
                # *** based on 3D Bayes results for star i= 100592: 
                minMr = 4.521381434007792  -  3*0.30204335173302443
                maxMr = 4.521381434007792  +  3*0.30204335173302443
                minFeH = -0.9864917239302899  -  3*0.10189896142866414
                maxFeH = -0.9864917239302899  +  3*0.10189896142866414
                minAr = 0.38892432982611685  -  3*0.06565803528930408
                maxAr = 0.38892432982611685  +  3*0.06565803528930408

                L = locus3D
                locus3Dok = L[(L['Ar']<=ArMax)&(L['Mr']>minMr)&(L['Mr']<maxMr)&(L['FeH']>minFeH)&(L['FeH']<maxFeH)]
                print('Locus from', np.size(L), 'to', np.size(locus3Dok))

                # also need to reset grids 
                FeH1d = np.sort(np.unique(locus3Dok['FeH']))
                Mr1d = np.sort(np.unique(locus3Dok['Mr']))  
                print('grid sizes:', np.size(FeH1d), np.size(Mr1d), np.size(Ar1d)) 
                mdLocus = np.array([minFeH, maxFeH, np.size(FeH1d), maxMr, minMr, np.size(Mr1d)])
                ################### this works, but 2-pass implemention TBW 
            else:
                # simply limit by maximum plausible extinction Ar
                locus3Dok = locus3D[locus3D['Ar']<=ArMax]

                
            ### compute chi2 map using provided 3D model locus (locus3Dok)
            chi2map = lt.getPhotoDchi2map3D(i, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, masterLocus=True)
            ## likelihood map
            likeGrid = np.exp(-0.5*chi2map)
            likeCube = likeGrid.reshape(np.size(FeH1d), np.size(Mr1d), np.size(Ar1d)) 
        #############################################################################################
        
        if (Ar1d.size>1):
            dAr = Ar1d[1] - Ar1d[0]
        else:
            dAr = 0.01
        catalog['chi2min'][i] = np.min(chi2map)
        
        # 2D Mr-FeH prior map replicated to get the same Mr-FeH-Ar grid as likelihood map
        if (restrictLocus):
            #### NEED TO RESAMPLE PRIOR
            # read maps with priors (and interpolate on the Mr-FeH grid given by locusData, which is same for all stars)
            L = locusData 
            smallLocus = L[(L['Ar']<=ArMax)&(L['Mr']>minMr)&(L['Mr']<maxMr)&(L['FeH']>minFeH)&(L['FeH']<maxFeH)]
            priorGrid = readPriors(rootname=priorsRootName, locusData=smallLocus)
            # get prior map indices using observed r band mags
            priorind = getPriorMapIndex(catalog['rmag'])
        # and proceed...
        prior2d = priorGrid[priorind[i]].reshape(np.size(FeH1d), np.size(Mr1d))
        priorCube = make3Dprior(prior2d, np.size(Ar1d))

        # posterior data cube
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
        catalog['MrEst'][i], catalog['MrEstUnc'][i] = getStats(Mr1d, margpostMr[2])
        catalog['FeHEst'][i], catalog['FeHEstUnc'][i] = getStats(FeH1d, margpostFeH[2])
        catalog['ArEst'][i], catalog['ArEstUnc'][i] = getStats(Ar1d, margpostAr[2])
        catalog['MrdS'][i] = Entropy(margpostMr[2]) - Entropy(margpostMr[0])
        catalog['FeHdS'][i] = Entropy(margpostFeH[2]) - Entropy(margpostFeH[0])
        catalog['ArdS'][i] = Entropy(margpostAr[2]) - Entropy(margpostAr[0])

        if (i in myStars):
            # plot 
            FeHStar = catalog['FeH'][i]
            MrStar = catalog['Mr'][i]
            ArStar = catalog['Ar'][i]
            indA = np.argmax(margpostAr[2])  
            pt.show3Flat2Dmaps(priorCube[:,:,indA], likeCube[:,:,indA], postCube[:,:,indA], mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar)
            showMargPosteriors3D(Mr1d, margpostMr, 'Mr', 'p(Mr)', FeH1d, margpostFeH, 'FeH', 'p(FeH)', Ar1d, margpostAr, 'Ar', 'p(Ar)', MrStar, FeHStar, ArStar)
            # these show marginal 2D and 1D distributions (aka "corner plot")
            showCornerPlot3(postCube, Mr1d, FeH1d, Ar1d, mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar, z0=ArStar)
            # Qr vs. FeH posterior and marginal 1D distributions for Qr and FeH
            Qr1d, margpostQr = showQrCornerPlot(postCube, Mr1d, FeH1d, Ar1d, x0=FeHStar, y0=MrStar, z0=ArStar, logScale=True)
            catalog['QrEst'][i], catalog['QrEstUnc'][i] = getStats(Qr1d, margpostQr)
            # basic info 
            print(' *** 3D Bayes results for star i=', i)
            print('r mag:', catalog['rmag'][i], 'g-r:', catalog['gr'][i], 'chi2min:', catalog['chi2min'][i])
            print('Mr: true=', MrStar, 'estimate=', catalog['MrEst'][i], ' +- ', catalog['MrEstUnc'][i])
            print('FeH: true=', FeHStar, 'estimate=', catalog['FeHEst'][i], ' +- ', catalog['FeHEstUnc'][i])
            print('Ar: true=', ArStar, 'estimate=', catalog['ArEst'][i], ' +- ', catalog['ArEstUnc'][i])
            print('Qr: true=', MrStar+ArStar, 'estimate=', catalog['QrEst'][i], ' +- ', catalog['QrEstUnc'][i])
            print('Mr drop in entropy:', catalog['MrdS'][i])
            print('FeH drop in entropy:', catalog['FeHdS'][i])
            print('Ar drop in entropy:', catalog['ArdS'][i])

    # store results 
    writeBayesEstimates(catalog, outfile, iStart, iEnd, do3D=True)
    return postCube, Mr1d, FeH1d, Ar1d


def writeBayesEstimates(catalog, outfile, iStart, iEnd, do3D=False):
    fout = open(outfile, "w")
    if do3D:
        fout.write("      glon       glat        FeHEst FeHUnc  MrEst  MrUnc  ArEst  ArUnc  chi2min    MrdS     FeHdS      ArdS \n")
    else:
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
        s = s + str("%6.2f  " % r5) + str("%5.2f  " % r6) 
        if do3D:
            r15 = catalog['ArEst'][i]
            r16 = catalog['ArEstUnc'][i]
            r19 = catalog['ArdS'][i]
            s = s + str("%6.2f  " % r15) + str("%5.2f  " % r16) + str("%5.2f  " % r7) + str("%8.1f  " % r8)  
            s = s + str("%8.1f  " % r9) + str("%8.1f  " % r19) + str("%8.0f" % i) + "\n"
        else:
            s = s + str("%5.2f  " % r7) + str("%8.1f  " % r8) + str("%8.1f  " % r9) + "\n"
        fout.write(s)             
    fout.close()
    return 


####################################################################################################
### toosl for testing Bayes results

## test Bayes estimates
# chiTest: if True, read noise-free stellar locus colors in lt.readTRILEGALLSST
# cmd: if True, in qpB, called by plotAll, plot umag vs. g-i instead of FeH vs. Mr diagram  
# fitQ: if true, call qpBcmd in plotAll and plot mean values of Mr, FeH and Ar in umag vs. g-i diagrams
# b3D: 3D Bayes version with Ar results
def checkBayes(infile1, infile2, chi2max=10, umagMax=99.9, chiTest=False, cmd=False, fitQ=False, b3D=False):
    
    ## input simulation
    if (umagMax < 99):
        simsAll = lt.readTRILEGALLSST(inTLfile=infile1, chiTest=chiTest)
        sims = simsAll[simsAll['umag']<umagMax]
        print('from', np.size(simsAll),' selected with u <', umagMax, np.size(sims))
        ## file with Bayes estimates
        simsBayesAll = lt.readTRILEGALLSSTestimates(infile=infile2, b3D=b3D)
        # volatile: assumes the same order 
        simsBayes = simsBayesAll[simsAll['umag']<umagMax]
    else:
        sims = lt.readTRILEGALLSST(inTLfile=infile1, chiTest=chiTest, b3D=b3D)
        ## file with Bayes estimates
        simsBayes = lt.readTRILEGALLSSTestimates(infile=infile2, b3D=b3D)

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
    sims['dAr'] = sims['Ar'] - simsBayes['ArEst']
    sims['dArNorm'] = sims['dAr'] / simsBayes['ArUnc'] 
    sims['MrML'] = simsBayes['MrEst']
    sims['FeHML'] = simsBayes['FeHEst']
    sims['ArML'] = simsBayes['ArEst']
    sims['chi2min'] = simsBayes['chi2min']
    sims['MrEst'] = simsBayes['MrEst']
    sims['FeHEst'] = simsBayes['FeHEst']
    
    ## dummy 
    sims['test_set'] = 1 + 0*sims['Ar']

    # entropy change
    sims['MrdS'] = simsBayes['MrdS'] 
    sims['FeHdS'] = simsBayes['FeHdS'] 
    sims['ArdS'] = simsBayes['ArdS'] 

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
    if (fitQ): 
        print('calling qpBcmd')
        pt.qpBcmd(dfName, color='gi', mag='umag', scatter=False)
    else:
        print('calling qpB Mr')
        pt.qpB(dfName, 'dMr', Dname='Mr', cmd=cmd)
        pt.qpB(dfName, 'dMr', Dname='Mr', cmd=cmd, estQ=True)
        print('calling qpB FeH')
        pt.qpB(dfName, 'dFeH', Dname='FeH', cmd=cmd)
        pt.qpB(dfName, 'dFeH', Dname='FeH', cmd=cmd, estQ=True)
        print('calling qpB Ar')
        pt.qpB(dfName, 'dAr', Dname='Ar', cmd=cmd)
        pt.qpB(dfName, 'dAr', Dname='Ar', cmd=cmd, estQ=True)

        
    
## quick comparison of Karlo's NN estimates for Mr, FeH, Ar
def cK(infile1, infile2, sim3=False, simtype='a', chiTest=True):
    ## input simulation
    # sims = lt.readTRILEGALLSST(inTLfile=infile1)
    sims = lt.readTRILEGALLSST(inTLfile=infile1, chiTest=chiTest)
    
    # these are dust-reddened colors and thus dust-extincted magnitudes
    sims['gmag'] = sims['rmag'] + sims['gr']  
    sims['umag'] = sims['gmag'] + sims['ug']
    sims['imag'] = sims['rmag'] - sims['ri']
    sims['zmag'] = sims['imag'] - sims['iz'] 
    ## Karlo's file
    if sim3 == False:
        # old version
        # simsML = lt.readKarloMLestimates(multiMethod=False, inKfile=infile2)
        # final version
        simsML = lt.readKarloMLestimates3D(inKfile=infile2)  
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
    print('from:', np.size(sims), 'selected in testset:', np.size(simsTest))  
    
    plotAll(simsTest)
    return sims, simsTest 
    

