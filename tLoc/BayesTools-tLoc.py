import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import LocusTools as lt 
import BayesTools as bt 
import PlotTools as pt
from astropy.table import Table


def prepareBayesEstimate(catalog, fitColors, locusData, locus3DList, ArGridList, priorsRootName):

    MrColumn='Mr'

    ## priors
    # read maps with priors (and interpolate on the Mr-FeH grid given by locusData, which is same for all stars)
    priorGrid = bt.readPriors(rootname=priorsRootName, locusData=locusData, MrColumn=MrColumn)
    # get prior map indices using observed r band mags
    priorind = bt.getPriorMapIndex(catalog['rmag'])

    ## properties of Ar grid for prior and likelihood
    bc = bt.getBayesConstants()
    ArCoeffs = {}
    ArCoeffs[0] = bc['ArCoeff0']
    ArCoeffs[1] = bc['ArCoeff1'] 
    ArCoeffs[2] = bc['ArCoeff2']
    ArCoeffs[3] = bc['ArCoeff3']
    ArCoeffs[4] = bc['ArCoeff4']
    
    ## color corrections due to dust reddening 
    C = lt.extcoeff()
    reddCoeffs = {}
    reddCoeffs['ug'] = C['u']-C['g']
    reddCoeffs['gr'] = C['g']-C['r']
    reddCoeffs['ri'] = C['r']-C['i']
    reddCoeffs['iz'] = C['i']-C['z']

    return priorGrid, priorind, ArCoeffs, reddCoeffs                    
                            

# NB obsColors must include all fitColors
# priorMap = priorGrid[priorind[i]]  where i comes from catalog[i] and catalog[rmag]
# 
def singleBayesEstimate(iStar, catalog, ArFit, MrStar, FeHStar, ArStar, priorMap, ArCoeff, reddCoeffs, fitColors, locusData, locus3DList, ArGridList):

    ### some quick prep work
    ## Mr and FeH 1-D grid properties extracted from locusData 
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
    
    ## maximum grid values for Ar from master locus 
    ArGridSmallMax = np.max(ArGridList['ArSmall'])
    ArGridMediumMax = np.max(ArGridList['ArMedium'])
    
    # the range of Ar to fit
    if ArFit<0:
        ArMin = 0.0
        ArMax = ArCoeff[2]
    else:
        ArMin = ArCoeff[3]*ArFit + ArCoeff[4]
        if ArMin<0: ArMin=0 
        ArMax = ArCoeff[0]*ArFit + ArCoeff[1]
        if ArMax<ArMin+ArCoeff[2]: ArMax=ArMin+ArCoeff[2]
    print('adopted Ar in the range:', ArMin, ArMax) 
    
    ## depending on ArMax, pick the adequate Ar resolution of locus3D
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
    locus3Dok = locus3D[(locus3D['Ar']>=ArMin)&(locus3D['Ar']<=ArMax)]
    Ar1d = ArGrid[(ArGrid>=ArMin)&(ArGrid<=ArMax)]
    if (Ar1d.size>1):
        dAr = Ar1d[1] - Ar1d[0]
    else:
        dAr = 0.01

   
    #########  the main Bayes block  ###############################     
    ## compute chi2 map using provided 3D model locus (locus3Dok) ##
    chi2map = lt.getPhotoDchi2map3D(iStar, fitColors, reddCoeffs, catalog, locus3Dok, ArCoeff, masterLocus=True)
    chi2min = np.min(chi2map)

    # likelihood map
    likeGrid = np.exp(-0.5*chi2map)
    likeCube = likeGrid.reshape(np.size(FeH1d), np.size(Mr1d), np.size(Ar1d))

    # generate 3D prior (Mr, FeH, Ar) from 2D (Mr, FeH) prior using uniform prior for Ar    
    prior2d = priorMap.reshape(np.size(FeH1d), np.size(Mr1d))
    priorCube = bt.make3Dprior(prior2d, np.size(Ar1d))        

    # posterior data cube
    postCube = priorCube * likeCube
    ################################################################

    
    ## get expectation values and uncertainties
    # first marginalize 
    margpostMr, margpostFeH, margpostAr = getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr)
    # and get stats
    MrEst, MrEstUnc = bt.getStats(Mr1d, margpostMr[2])
    FeHEst, FeHEstUnc = bt.getStats(FeH1d, margpostFeH[2])
    ArEst, ArEstUnc = bt.getStats(Ar1d, margpostAr[2])

    
    ## plots 
    indA = np.argmax(margpostAr[2])
    print('selected a slice at Ar=', Ar1d[indA], 'while from getStats Ar=', ArEst, '+-', ArEstUnc)
    pt.show3Flat2Dmaps(priorCube[:,:,indA], likeCube[:,:,indA], postCube[:,:,indA], mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar)
    bt.showMargPosteriors3D(Mr1d, margpostMr, 'Mr', 'p(Mr)', FeH1d, margpostFeH, 'FeH', 'p(FeH)', Ar1d, margpostAr, 'Ar', 'p(Ar)', MrStar, FeHStar, ArStar)
    # these show marginal 2D and 1D distributions (aka "corner plot")
    bt.showCornerPlot3(postCube, Mr1d, FeH1d, Ar1d, mdLocus, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar, z0=ArStar)

    # print basic info 
    print(' *** results,   chi2min:', chi2min)
    print('Mr: true=', MrStar, 'estimate=', MrEst, ' +- ', MrEstUnc) 
    print('FeH: true=', FeHStar, 'estimate=', FeHEst, ' +- ', FeHEstUnc) 
    print('Ar: true=', ArStar, 'estimate=', ArEst, ' +- ', ArEstUnc) 
    return margpostAr[2]



def getMargPosteriors(priorCube, likeCube, postCube, dMr, dFeH, dAr):
    margpostMr = {}
    margpostFeH = {}
    margpostAr = {}
    margpostMr[0], margpostFeH[0], margpostAr[0] = bt.getMargDistr3D(priorCube, dMr, dFeH, dAr)
    margpostMr[1], margpostFeH[1], margpostAr[1] = bt.getMargDistr3D(likeCube, dMr, dFeH, dAr) 
    margpostMr[2], margpostFeH[2], margpostAr[2] = bt.getMargDistr3D(postCube, dMr, dFeH, dAr) 
    return margpostMr, margpostFeH, margpostAr 






def compareColors4bestFits(locus, star1, star2, errors, alpha=0.8, title="", plotname='compareColors.png'):

    fig,ax = plt.subplots(6,2,figsize=(12,12))

    def plotPanel(xP, yP, df, xName, yName, cName, xMin, xMax, yMin, yMax, xLabel, yLabel, alpha):
        ax[xP,yP].scatter(df[xName], df[yName], s=0.3, c=df[cName], cmap=plt.cm.jet, alpha=alpha)
        ax[xP,yP].set_xlim(xMin, xMax)
        ax[xP,yP].set_ylim(yMin, yMax)
        ax[xP,yP].set_xlabel(xLabel)
        ax[xP,yP].set_ylabel(yLabel)


    df = locus
    star = star1
    ugT = star['ug']
    grT = star['gr'] 
    riT = star['ri'] 
    izT = star['iz']
    sum = ((df['ug']-ugT)/errors['ug'])**2 + ((df['gr']-grT)/errors['gr'])**2 + ((df['ri']-riT)/errors['ri'])**2
    df['colorDist1'] = np.sqrt(sum + ((df['iz']-izT)/errors['iz'])**2)
    star = star2
    ugT = star['ug']
    grT = star['gr'] 
    riT = star['ri'] 
    izT = star['iz']
    sum = ((df['ug']-ugT)/errors['ug'])**2 + ((df['gr']-grT)/errors['gr'])**2 + ((df['ri']-riT)/errors['ri'])**2
    df['colorDist2'] = np.sqrt(sum + ((df['iz']-izT)/errors['iz'])**2)
    print('For supplied colors and errors, without reddening correction (Ar=0):') 
    print('  Minimum chi2:', np.min(df['colorDist1']), 'VS', np.min(df['colorDist2']))
    print('        at Mr=:', df['MrTrue'][np.argmin(df['colorDist1'])], 'VS', df['MrTrue'][np.argmin(df['colorDist2'])])
    print('       at FeH=:', df['FeH'][np.argmin(df['colorDist1'])], 'VS', df['FeH'][np.argmin(df['colorDist2'])])

    tLocMin = 0
    tLocMax = 17
    ax[0,0].set_title(title)

    chi2max = 50
    plotPanel(0, 0, df, 'Mr', 'colorDist1', 'FeH', -1, 10, 0, chi2max, 'Mr', 'chi2', alpha) 
    plotPanel(0, 1, df, 'Mr', 'colorDist2', 'FeH', -1, 10, 0, chi2max, 'Mr', 'chi2', alpha) 

    plotPanel(1, 0, df, 'tLoc', 'colorDist1', 'FeH', tLocMin, tLocMax, 0, chi2max, 'tLocus', 'chi2', alpha) 
    plotPanel(1, 1, df, 'tLoc', 'colorDist2', 'FeH', tLocMin, tLocMax, 0, chi2max, 'tLocus', 'chi2', alpha) 

    plotPanel(2, 0, df, 'gi', 'colorDist1', 'FeH', -0.7, 4.4, 0, chi2max, 'g-i', 'chi2', alpha) 
    plotPanel(2, 1, df, 'gi', 'colorDist2', 'FeH', -0.7, 4.4, 0, chi2max, 'g-i', 'chi2', alpha) 

    plotPanel(3, 0, df, 'ug', 'gr', 'FeH', 0.5, 3.6, -0.2, 2.2, 'u-g', 'g-r', alpha) 
    plotPanel(3, 1, df, 'ug', 'gr', 'FeH', 0.5, 3.6, -0.2, 2.2, 'u-g', 'g-r', alpha) 
    ax[3,0].scatter(star1['ug'], star1['gr'], s=25.0, c='black')
    ax[3,0].scatter(star1['ug'], star1['gr'], s=12.0, c='white')
    ax[3,1].scatter(star2['ug'], star2['gr'], s=25.0, c='black')
    ax[3,1].scatter(star2['ug'], star2['gr'], s=12.0, c='white')

    plotPanel(4, 0, df, 'gr', 'ri', 'FeH', -0.2, 1.6, -0.2, 2.2, 'g-r', 'r-i', alpha) 
    plotPanel(4, 1, df, 'gr', 'ri', 'FeH', -0.2, 1.6, -0.2, 2.2, 'g-r', 'r-i', alpha) 
    ax[4,0].scatter(star1['gr'], star1['ri'], s=25.0, c='black')
    ax[4,0].scatter(star1['gr'], star1['ri'], s=12.0, c='white')
    ax[4,1].scatter(star2['gr'], star2['ri'], s=25.0, c='black')
    ax[4,1].scatter(star2['gr'], star2['ri'], s=12.0, c='white')

    plotPanel(5, 0, df, 'ri', 'iz', 'FeH', -0.2, 2.2, -0.2, 2.2, 'r-i', 'i-z', alpha) 
    plotPanel(5, 1, df, 'ri', 'iz', 'FeH', -0.2, 2.2, -0.2, 2.2, 'r-i', 'i-z', alpha) 
    ax[5,0].scatter(star1['ri'], star1['iz'], s=25.0, c='black')
    ax[5,0].scatter(star1['ri'], star1['iz'], s=12.0, c='white')
    ax[5,1].scatter(star2['ri'], star2['iz'], s=25.0, c='black')
    ax[5,1].scatter(star2['ri'], star2['iz'], s=12.0, c='white')
  
    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return
