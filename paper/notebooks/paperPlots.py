
import numpy as np
import pylab as plt
import scipy.stats as stats
from scipy.stats import norm
from astroML.stats import binned_statistic_2d
import LocusTools as lt 
import BayesTools as bt 
import PlotTools as pt


def plot3diagsBobAbel(df1, df2, df3, df4, L0, L1, L2, WD, WDMD, BHB, zoom=False):

    fig=plt.figure(2,figsize=(12,4))
    fig.subplots_adjust(wspace=0.2,hspace=0.2,top=0.97,bottom=0.1,left=0.09,right=0.95)
    plt.rcParams['font.size'] = 12

    def getMr(gi, FeH):
        MrFit = -5.06 + 14.32*gi -12.97*gi**2 + 6.127*gi**3 -1.267*gi**4 + 0.0967*gi**5
        ## offset for metallicity, valid for -2.5 < FeH < 0.2
        FeHoffset = 4.50 -1.11*FeH -0.18*FeH**2
        return MrFit + FeHoffset

    def plotPanel(ax, df1, df2, df3, df4, Xname, Yname, xLabel, yLabel, xMin, xMax, yMin, yMax):
        # ax.scatter(df1[Xname], df1[Yname],s=0.01, c='red')
        ax.scatter(df2[Xname], df2[Yname],s=0.012, c='blue')
        ax.scatter(df3[Xname], df3[Yname],s=0.014, c='cyan')
        ax.scatter(df4[Xname], df4[Yname],s=0.016, c='green')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        # plt.title(title, x=0.30, y=1.008)
        
    cLocus = 'yellow'
  
    ### subset with Gaia parallaxes (SNR>20)
    ax=fig.add_subplot(131)
    ax.scatter(df4['gi'], df4['Mr'], s=3.5, c='green') 
    if zoom:
        #ax.text(-1.8, 14, 'DA: yellow')
        #ax.text(-1.8, 14.8, 'DB: red')
        #ax.text(-1.8, 15.6, 'DC: cyan')
        ax.text(-1.8, 14, 'H: yellow')
        ax.text(-1.8, 14.8, 'He: cyan') 
    if (1):
        giGrid = np.linspace(0.2, 4.0, 300)
        Mr0 = getMr(giGrid, 0.0)
        Mr1 = getMr(giGrid, -1.0)
        Mr2 = getMr(giGrid, -2.0)
        ax.plot(giGrid, Mr0, ls='dotted', c='yellow')
        ax.plot(giGrid, Mr1, ls='dotted', c='yellow')
        ax.plot(giGrid, Mr2, ls='dotted', c='yellow')
    ax.set_xlim(-1.8,4.1)
    ax.set_ylim(16.1, 1.0)
    if zoom:
        ax.set_xlim(-2.0, 1.0)
        ax.set_ylim(16.1, 8.0)
       
    ax.set_xlabel('g-i')
    ax.set_ylabel('Mr')
    if (1):
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='--', c='yellow')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-.', c='red')
        ax.plot(WD['DCgr']+WD['DCri'], WD['Mr'], ls='--', c='cyan')
    if (zoom==False):
        ax.scatter(WDMD['DAd_gr']+WDMD['DAd_ri'], WDMD['DAd_Mr'], s=0.01, c='black')
        # ax.scatter(BHB['gr']+BHB['ri'], BHB['Mr'], s=0.01, c='red')
  
    
    ax=fig.add_subplot(132)
    if zoom:
        plotPanel(ax, df1, df2, df3, df4, 'ug', 'gr', 'u-g', 'g-r', -1.0, 1.5, -0.7, 0.4)
    else:
        plotPanel(ax, df1, df2, df3, df4, 'ug', 'gr', 'u-g', 'g-r', -1.0, 4.0, -0.7, 1.9)
    ax.plot(L0['ug'], L0['gr'], ls='--', c=cLocus)
    ax.plot(L1['ug'], L1['gr'], ls='-', c=cLocus)
    ax.plot(L2['ug'], L2['gr'], ls='-.', c=cLocus)
    if (1):
        ax.plot(WD['DAug'], WD['DAgr'], ls='-', c='black')
        ax.plot(WD['DAug'], WD['DAgr'], ls='--', c=cLocus)
        ax.plot(WD['DBug'], WD['DBgr'], ls='-', c='black')
        ax.plot(WD['DBug'], WD['DBgr'], ls='--', c='red')
        ax.plot(WD['DCug'], WD['DCgr'], ls='--', c='cyan')
        # ax.scatter(BHB['ug'], BHB['gr'], s=0.01, c='red')

    if (zoom==False):
        ax.scatter(WDMD['DAd_ug'], WDMD['DAd_gr'], s=0.01, c='black')
 
        
    ax=fig.add_subplot(133)
    if zoom:
        plotPanel(ax, df1, df2, df3, df4, 'gr', 'ri', 'g-r', 'r-i', -0.7, 0.4, -0.5, 0.4)
    else:
        plotPanel(ax, df1, df2, df3, df4, 'gr', 'ri', 'g-r', 'r-i', -0.7, 1.9, -0.5, 2.3)
        
    ax.plot(L0['gr'], L0['ri'], ls='--', c=cLocus)
    ax.plot(L1['gr'], L1['ri'], ls='-', c=cLocus)
    ax.plot(L2['gr'], L2['ri'], ls='-.', c=cLocus)
    if (1):
        ax.plot(WD['DAgr'], WD['DAri'], ls='-', c='black')
        ax.plot(WD['DAgr'], WD['DAri'], ls='--', c=cLocus)
        ax.plot(WD['DBgr'], WD['DBri'], ls='-', c='black')
        ax.plot(WD['DBgr'], WD['DBri'], ls='--', c='red')
        ax.plot(WD['DCgr'], WD['DCri'], ls='--', c='cyan')
        # ax.scatter(BHB['gr'], BHB['ri'], s=0.01, c='red')

    if (zoom==False):
        ax.scatter(WDMD['DAd_gr'], WDMD['DAd_ri'], s=0.01, c='black')
        
    plt.tight_layout()
    if zoom:
        plt.savefig('plot3diagsBAzoom.png')
        print('made plot: plot3diagsBAzoom.png')
    else:
        plt.savefig('plot3diagsBA.png')
        print('made plot: plot3diagsBA.png')
    plt.show() 
    plt.close("all")
    return



def plot3diagsData(df1, df2, df3, df4, L0, L1, L2, WD=""):

    fig=plt.figure(2,figsize=(12,4))
    fig.subplots_adjust(wspace=0.2,hspace=0.2,top=0.97,bottom=0.1,left=0.09,right=0.95)
    plt.rcParams['font.size'] = 12

    def plotPanel(ax, df1, df2, df3, df4, Xname, Yname, xLabel, yLabel, xMin, xMax, yMin, yMax):
        ax.scatter(df1[Xname], df1[Yname],s=0.01, c='red')
        ax.scatter(df2[Xname], df2[Yname],s=0.012, c='blue')
        ax.scatter(df3[Xname], df3[Yname],s=0.014, c='cyan')
        ax.scatter(df4[Xname], df4[Yname],s=0.016, c='green')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        # plt.title(title, x=0.30, y=1.008)
        
    cLocus = 'yellow'
  
    ax=fig.add_subplot(131)
    plotPanel(ax, df1, df2, df3, df4, 'gi', 'rmag', 'g-i', 'SDSS r mag', -1.0, 3.9, 22.0, 13.0)

    ax=fig.add_subplot(132)
    plotPanel(ax, df1, df2, df3, df4, 'ug', 'gr', 'u-g', 'g-r', -1.0, 4.0, -0.7, 1.9)
    ax.plot(L0['ug'], L0['gr'], ls='--', c=cLocus)
    ax.plot(L1['ug'], L1['gr'], ls='-', c=cLocus)
    ax.plot(L2['ug'], L2['gr'], ls='-.', c=cLocus)
    if (1):
        ax.plot(WD['DAug'], WD['DAgr'], ls='--', c=cLocus)
        ax.plot(WD['DBug'], WD['DBgr'], ls='--', c='red')
        ax.plot(WD['DCug'], WD['DCgr'], ls='--', c='cyan')
 
    ax=fig.add_subplot(133)
    plotPanel(ax, df1, df2, df3, df4, 'gr', 'ri', 'g-r', 'r-i', -0.7, 1.9, -0.5, 2.3)
    ax.plot(L0['gr'], L0['ri'], ls='--', c=cLocus)
    ax.plot(L1['gr'], L1['ri'], ls='-', c=cLocus)
    ax.plot(L2['gr'], L2['ri'], ls='-.', c=cLocus)
    if (1):
        ax.plot(WD['DAgr'], WD['DAri'], ls='--', c=cLocus)
        ax.plot(WD['DBgr'], WD['DBri'], ls='--', c='red')
        ax.plot(WD['DCgr'], WD['DCri'], ls='--', c='cyan')
        
    plt.tight_layout()
    plt.savefig('plot3diagsData.png')
    print('made plot: plot3diagsData.png')
    plt.show() 
    plt.close("all")
    return



def plot3HRdiags(df1, df2, df3, L0, L1, L2, Lok, WD):

    fig=plt.figure(2,figsize=(12,4))
    fig.subplots_adjust(wspace=0.2,hspace=0.2,top=0.97,bottom=0.1,left=0.09,right=0.95)
    plt.rcParams['font.size'] = 12

    def getMr(gi, FeH):
        MrFit = -5.06 + 14.32*gi -12.97*gi**2 + 6.127*gi**3 -1.267*gi**4 + 0.0967*gi**5
        ## offset for metallicity, valid for -2.5 < FeH < 0.2
        FeHoffset = 4.50 -1.11*FeH -0.18*FeH**2
        return MrFit + FeHoffset

    def plotPanel(ax, df1, df2, df3, df4, Xname, Yname, xLabel, yLabel, xMin, xMax, yMin, yMax):
        ax.scatter(df1[Xname], df1[Yname],s=0.01, c='blue')
        ax.scatter(df2[Xname], df2[Yname],s=0.01, c='red')
        ax.scatter(df3[Xname], df3[Yname],s=0.01, c='cyan')
        ax.scatter(df4[Xname], df4[Yname],s=0.01, c='green')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        # plt.title(title, x=0.30, y=1.008)

           
    ### locus color-coded by FeH
    ax=fig.add_subplot(131)
    ax.scatter(Lok['gi'], Lok['Mr'], s=0.01, c=Lok['FeH'], cmap=plt.cm.jet)
    ax.plot(L0['gi'], L0['Mr'], ls='--', c='black')
    ax.plot(L1['gi'], L1['Mr'], ls='-', c='black')
    ax.plot(L2['gi'], L2['Mr'], ls='-.', c='yellow')

    ax.set_xlim(-2.0,3.8)
    ax.set_ylim(15.0, -2.0)
    ax.set_xlabel('g-i')
    ax.set_ylabel('Mr')
    if (1):
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='--', c='yellow')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-.', c='red')
        ax.plot(WD['DCgr']+WD['DCri'], WD['Mr'], ls='--', c='cyan')


    ### subset with Gaia parallaxes (SNR>20)
    ax=fig.add_subplot(132)
    ax.scatter(df1['gi'], df1['Mr'],s=0.1, c='green') 
 
    # 3 locus sequences
    ax.plot(L0['gi'], L0['Mr'], ls='--', c='black')
    ax.plot(L1['gi'], L1['Mr'], ls='-', c='black')
    ax.plot(L2['gi'], L2['Mr'], ls='-.', c='black')    
    if (1):
        giGrid = np.linspace(0.2, 3.5, 300)
        Mr0 = getMr(giGrid, 0.0)
        Mr1 = getMr(giGrid, -1.0)
        Mr2 = getMr(giGrid, -2.0)
        ax.plot(giGrid, Mr0, ls='dotted', c='yellow')
        ax.plot(giGrid, Mr1, ls='dotted', c='yellow')
        ax.plot(giGrid, Mr2, ls='dotted', c='yellow')
    ax.set_xlim(-1.0,3.8)
    ax.set_ylim(15.0, -2.0)
    ax.set_xlabel('g-i')
    ax.set_ylabel('Mr')
    if (1):
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='--', c='yellow')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-.', c='red')
        ax.plot(WD['DCgr']+WD['DCri'], WD['Mr'], ls='--', c='cyan')


    ### subset with CBJ distances 
    ax=fig.add_subplot(133)
    ax.scatter(df2['gi'], df2['MrPho'],s=0.01, c='red')
    ax.scatter(df3['gi'], df3['MrPho'],s=0.01, c='cyan')

    # 3 locus sequences
    ax.plot(L0['gi'], L0['Mr'], ls='--', c='black')
    ax.plot(L1['gi'], L1['Mr'], ls='-', c='black')
    ax.plot(L2['gi'], L2['Mr'], ls='-.', c='black')
    ax.set_xlim(-1.0,3.8)
    ax.set_ylim(15.0, -2.0)
    ax.set_xlabel('g-i')
    ax.set_ylabel('Mr')
    if (1):
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DAgr']+WD['DAri'], WD['Mr'], ls='--', c='yellow')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-', c='black')
        ax.plot(WD['DBgr']+WD['DBri'], WD['Mr'], ls='-.', c='red')
        ax.plot(WD['DCgr']+WD['DCri'], WD['Mr'], ls='--', c='cyan')

        
    plt.tight_layout()
    plt.savefig('plot3HRdiags.png')
    print('made plot: plot3HRdiags.png')
    plt.show() 
    plt.close("all")
    return



def showHR(dataDF, sdssLdf, dfs, plotname='plot3HRdiags.png'):
    ## show Mr vs. g-i relations
    fig,ax = plt.subplots(1,1,figsize=(6,8))

    ax.scatter(dataDF['gi'], dataDF['Mr'], s=0.1, c='red')

    ax.plot(sdssLdf[0]['gi'], sdssLdf[0]['Mr'], ls='-', c='yellow')
    ax.plot(sdssLdf[0]['gi'], sdssLdf[0]['Mr'], ls='--', c='black')
    ax.plot(sdssLdf[1]['gi'], sdssLdf[1]['Mr'], ls='-', c='yellow')
    ax.plot(sdssLdf[1]['gi'], sdssLdf[1]['Mr'], ls='--', c='cyan')
    ax.plot(sdssLdf[2]['gi'], sdssLdf[2]['Mr'], ls='-', c='yellow')
    ax.plot(sdssLdf[2]['gi'], sdssLdf[2]['Mr'], ls='--', c='blue')

    ax.plot(dfs[0]['gi'], dfs[0]['Mr'], c='black')
    ax.plot(dfs[1]['gi'], dfs[1]['Mr'], c='cyan')
    ax.plot(dfs[2]['gi'], dfs[2]['Mr'], c='blue')

    ax.set_xlim(-1.0,4.0)
    ax.set_ylim(16.0, -3.0)
    ax.set_xlabel('SDSS (g-i)')
    ax.set_ylabel('Mr')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return

    
def showUGR(dataDF, sdssLdf, dfs, plotname='showUGR.png'):
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))

    ax.scatter(dataDF['ug'], dataDF['gr'], s=0.1, c='red')

    ax.plot(sdssLdf[0]['ug'], sdssLdf[0]['gr'], ls='-', c='yellow')
    ax.plot(sdssLdf[0]['ug'], sdssLdf[0]['gr'], ls='--', c='black')
    ax.plot(sdssLdf[1]['ug'], sdssLdf[1]['gr'], ls='-', c='yellow')
    ax.plot(sdssLdf[1]['ug'], sdssLdf[1]['gr'], ls='--', c='cyan')
    ax.plot(sdssLdf[2]['ug'], sdssLdf[2]['gr'], ls='-', c='yellow')
    ax.plot(sdssLdf[2]['ug'], sdssLdf[2]['gr'], ls='--', c='blue')

    ax.plot(dfs[0]['ug'], dfs[0]['gr'], c='black')
    ax.plot(dfs[1]['ug'], dfs[1]['gr'], c='cyan')
    ax.plot(dfs[2]['ug'], dfs[2]['gr'], c='blue')

    ax.set_xlim(-0.7,3.5)
    ax.set_ylim(-0.7, 1.8)
    ax.set_xlabel('u-g')
    ax.set_ylabel('g-r')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return

    
def showGRI(dataDF, sdssLdf, dfs, plotname='showGRI.png'):
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))

    ax.scatter(dataDF['gr'], dataDF['ri'], s=0.1, c='red')

    ax.plot(sdssLdf[0]['gr'], sdssLdf[0]['ri'], ls='-', c='yellow')
    ax.plot(sdssLdf[0]['gr'], sdssLdf[0]['ri'], ls='--', c='black')
    ax.plot(sdssLdf[1]['gr'], sdssLdf[1]['ri'], ls='-', c='yellow')
    ax.plot(sdssLdf[1]['gr'], sdssLdf[1]['ri'], ls='--', c='cyan')
    ax.plot(sdssLdf[2]['gr'], sdssLdf[2]['ri'], ls='-', c='yellow')
    ax.plot(sdssLdf[2]['gr'], sdssLdf[2]['ri'], ls='--', c='blue')

    ax.plot(dfs[0]['gr'], dfs[0]['ri'], c='black')
    ax.plot(dfs[1]['gr'], dfs[1]['ri'], c='cyan')
    ax.plot(dfs[2]['gr'], dfs[2]['ri'], c='blue')
    
    ax.set_xlim(-0.7,1.8)
    ax.set_ylim(-0.7, 2.2)
    ax.set_xlabel('g-r')
    ax.set_ylabel('r-i')

    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return

    
def showRIZ(dataDF, sdssLdf, dfs, plotname='showRIZ.png'):
    
    fig,ax = plt.subplots(1,1,figsize=(8,8))

    ax.scatter(dataDF['ri'], dataDF['iz'], s=0.1, c='red')

    ax.plot(sdssLdf[0]['ri'], sdssLdf[0]['iz'], ls='-', c='yellow')
    ax.plot(sdssLdf[0]['ri'], sdssLdf[0]['iz'], ls='--', c='black')
    ax.plot(sdssLdf[1]['ri'], sdssLdf[1]['iz'], ls='-', c='yellow')
    ax.plot(sdssLdf[1]['ri'], sdssLdf[1]['iz'], ls='--', c='cyan')
    ax.plot(sdssLdf[2]['ri'], sdssLdf[2]['iz'], ls='-', c='yellow')
    ax.plot(sdssLdf[2]['ri'], sdssLdf[2]['iz'], ls='--', c='blue')

    ax.plot(dfs[0]['ri'], dfs[0]['iz'], c='black')
    ax.plot(dfs[1]['ri'], dfs[1]['iz'], c='cyan')
    ax.plot(dfs[2]['ri'], dfs[2]['iz'], c='blue')
    
    ax.set_xlim(-0.7,2.7)
    ax.set_ylim(-0.7, 1.8)
    ax.set_xlabel('r-i')
    ax.set_ylabel('i-z')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return


def show4diagsDSED(age, ages, FeHlist, DSEDiso, FeHlocus3vals, Lcomparison, dataDF):
    DSEDlist = DSEDiso[ages.index(age)]
    DSEDs = []
    for FeH in FeHlocus3vals:
        DSEDs.append(DSEDlist[FeHlist.index(FeH)])
    showHR(dataDF, Lcomparison, DSEDs)
    showUGR(dataDF, Lcomparison, DSEDs)
    showGRI(dataDF, Lcomparison, DSEDs)
    showRIZ(dataDF, Lcomparison, DSEDs)
    return

 


def compare2isochrones(dataDF, isoDF1, isoDF2, alpha1=0.05, alpha2=0.8, title="", plotname='compare2isochrones.png'):

    fig,ax = plt.subplots(3,2,figsize=(12,12))

    plt.rcParams.update({'font.size': 12})
    plt.rc('axes', labelsize=14)    

    ax[0,0].scatter(dataDF['gi'], dataDF['Mr'], s=0.1, c='black')
    ax[0,0].scatter(isoDF1['gi'], isoDF1['Mr'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[0,0].set_xlim(-1.0, 4.2)
    ax[0,0].set_ylim(16.5, -3.2)
    ax[0,0].set_xlabel('g-i')
    ax[0,0].set_ylabel('Mr')

    ax[0,1].scatter(dataDF['gi'], dataDF['Mr'], s=0.1, c='black')
    ax[0,1].scatter(isoDF2['gi'], isoDF2['Mr'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[0,1].set_xlim(-1.0, 4.2)
    ax[0,1].set_ylim(16.5, -3.2)
    ax[0,1].set_xlabel('g-i')
    ax[0,1].set_ylabel('Mr')
    ax[0,0].set_title(title, loc='center')

    ax[1,0].scatter(dataDF['ug'], dataDF['gr'], s=0.1, c='black')
    if (1):
        flag = (isoDF1['Mr']>0.5)
        isoDF1x = isoDF1[flag]
        ax[1,0].scatter(isoDF1x['ug'], isoDF1x['gr'], s=0.1, c=isoDF1x['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    else:
        ax[1,0].scatter(isoDF1['ug'], isoDF1['gr'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[1,0].set_xlim(-1.0,4.2)
    ax[1,0].set_ylim(-0.5, 2.2)
    ax[1,0].set_xlabel('u-g')
    ax[1,0].set_ylabel('g-r')

    ax[1,1].scatter(dataDF['ug'], dataDF['gr'], s=0.1, c='black')
    ax[1,1].scatter(isoDF2['ug'], isoDF2['gr'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[1,1].set_xlim(-1.0,4.2)
    ax[1,1].set_ylim(-0.5, 2.2)
    ax[1,1].set_xlabel('u-g')
    ax[1,1].set_ylabel('g-r')

    ax[2,0].scatter(dataDF['gr'], dataDF['ri'], s=0.1, c='black')
    ax[2,0].scatter(isoDF1['gr'], isoDF1['ri'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[2,0].set_xlim(-0.5, 2.2)
    ax[2,0].set_ylim(-0.5, 2.4)
    ax[2,0].set_xlabel('g-r')
    ax[2,0].set_ylabel('r-i')

    ax[2,1].scatter(dataDF['gr'], dataDF['ri'], s=0.1, c='black')
    ax[2,1].scatter(isoDF2['gr'], isoDF2['ri'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[2,1].set_xlim(-0.5, 2.2)
    ax[2,1].set_ylim(-0.5, 2.4)
    ax[2,1].set_xlabel('g-r')
    ax[2,1].set_ylabel('r-i')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return



def compare2isochrones2(dataDF, isoDF1, isoDF2, alpha1=0.05, alpha2=0.8, title="", plotname='compare2isochrones.png'):

    fig,ax = plt.subplots(3,2,figsize=(12,12))

    plt.rcParams.update({'font.size': 12})
    plt.rc('axes', labelsize=14)    

    ax[0,0].scatter(dataDF['gi'], dataDF['Mr'], s=0.1, c='black')
    ax[0,0].scatter(isoDF1['gi'], isoDF1['Mr'], s=0.1, c=isoDF1['Mr'], cmap=plt.cm.jet, alpha=alpha1)
    ax[0,0].set_xlim(-1.0, 4.2)
    ax[0,0].set_ylim(16.5, -3.2)
    ax[0,0].set_xlabel('g-i')
    ax[0,0].set_ylabel('Mr')

    ax[0,1].scatter(dataDF['gi'], dataDF['Mr'], s=0.1, c='black')
    ax[0,1].scatter(isoDF2['gi'], isoDF2['Mr'], s=0.1, c=isoDF2['Mr'], cmap=plt.cm.jet, alpha=alpha2)
    ax[0,1].set_xlim(-1.0, 4.2)
    ax[0,1].set_ylim(16.5, -3.2)
    ax[0,1].set_xlabel('g-i')
    ax[0,1].set_ylabel('Mr')
    ax[0,0].set_title(title, loc='center')

    ax[1,0].scatter(dataDF['ug'], dataDF['gr'], s=0.1, c='black')
    if (1):
        flag = (isoDF1['Mr']>0.5)
        isoDF1x = isoDF1[flag]
        ax[1,0].scatter(isoDF1x['ug'], isoDF1x['gr'], s=0.1, c=isoDF1x['Mr'], cmap=plt.cm.jet, alpha=alpha1)
    else:
        ax[1,0].scatter(isoDF1['ug'], isoDF1['gr'], s=0.1, c=isoDF1['Mr'], cmap=plt.cm.jet, alpha=alpha1)
    ax[1,0].set_xlim(-1.0,4.2)
    ax[1,0].set_ylim(-0.5, 2.2)
    ax[1,0].set_xlabel('u-g')
    ax[1,0].set_ylabel('g-r')

    ax[1,1].scatter(dataDF['ug'], dataDF['gr'], s=0.1, c='black')
    ax[1,1].scatter(isoDF2['ug'], isoDF2['gr'], s=0.1, c=isoDF2['Mr'], cmap=plt.cm.jet, alpha=alpha2)
    ax[1,1].set_xlim(-1.0,4.2)
    ax[1,1].set_ylim(-0.5, 2.2)
    ax[1,1].set_xlabel('u-g')
    ax[1,1].set_ylabel('g-r')

    ax[2,0].scatter(dataDF['gr'], dataDF['ri'], s=0.1, c='black')
    ax[2,0].scatter(isoDF1['gr'], isoDF1['ri'], s=0.1, c=isoDF1['Mr'], cmap=plt.cm.jet, alpha=alpha1)
    ax[2,0].set_xlim(-0.5, 2.2)
    ax[2,0].set_ylim(-0.5, 2.4)
    ax[2,0].set_xlabel('g-r')
    ax[2,0].set_ylabel('r-i')

    ax[2,1].scatter(dataDF['gr'], dataDF['ri'], s=0.1, c='black')
    ax[2,1].scatter(isoDF2['gr'], isoDF2['ri'], s=0.1, c=isoDF2['Mr'], cmap=plt.cm.jet, alpha=alpha2)
    ax[2,1].set_xlim(-0.5, 2.2)
    ax[2,1].set_ylim(-0.5, 2.4)
    ax[2,1].set_xlabel('g-r')
    ax[2,1].set_ylabel('r-i')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return






def compare2isochronesColorMr(dataDF, isoDF1, isoDF2, alpha1=0.05, alpha2=0.8, title="", plotname='compare2isochronesColorMr.png'):

    fig,ax = plt.subplots(5,2,figsize=(12,12))

    MrMin = -3.2
    MrMax = 16.5
    ax[0,0].scatter(dataDF['Mr'], dataDF['ug'], s=0.1, c='black')
    ax[0,0].scatter(isoDF1['Mr'], isoDF1['ug'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[0,0].set_xlim(MrMin, MrMax)
    ax[0,0].set_ylim(-0.5, 4.0)
    ax[0,0].set_xlabel('Mr')
    ax[0,0].set_ylabel('u-g')
    ax[0,0].set_title(title)

    ax[0,1].scatter(dataDF['Mr'], dataDF['ug'], s=0.1, c='black')
    ax[0,1].scatter(isoDF2['Mr'], isoDF2['ug'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[0,1].set_xlim(MrMin, MrMax)
    ax[0,1].set_ylim(-0.5, 4.0)
    ax[0,1].set_xlabel('Mr')
    ax[0,1].set_ylabel('u-g')
   
    ax[1,0].scatter(dataDF['Mr'], dataDF['gr'], s=0.1, c='black')
    ax[1,0].scatter(isoDF1['Mr'], isoDF1['gr'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[1,0].set_xlim(MrMin, MrMax)
    ax[1,0].set_ylim(-0.5, 2.2)
    ax[1,0].set_xlabel('Mr')
    ax[1,0].set_ylabel('g-r')

    ax[1,1].scatter(dataDF['Mr'], dataDF['gr'], s=0.1, c='black')
    ax[1,1].scatter(isoDF2['Mr'], isoDF2['gr'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[1,1].set_xlim(MrMin, MrMax)
    ax[1,1].set_ylim(-0.5, 2.2)
    ax[1,1].set_xlabel('Mr')
    ax[1,1].set_ylabel('g-r')

    ax[2,0].scatter(dataDF['Mr'], dataDF['ri'], s=0.1, c='black')
    ax[2,0].scatter(isoDF1['Mr'], isoDF1['ri'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[2,0].set_xlim(MrMin, MrMax)
    ax[2,0].set_ylim(-0.5, 2.2)
    ax[2,0].set_xlabel('Mr')
    ax[2,0].set_ylabel('r-i')

    ax[2,1].scatter(dataDF['Mr'], dataDF['ri'], s=0.1, c='black')
    ax[2,1].scatter(isoDF2['Mr'], isoDF2['ri'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha1)
    ax[2,1].set_xlim(MrMin, MrMax)
    ax[2,1].set_ylim(-0.5, 2.2)
    ax[2,1].set_xlabel('Mr')
    ax[2,1].set_ylabel('r-i')

    ax[3,0].scatter(dataDF['Mr'], dataDF['iz'], s=0.1, c='black')
    ax[3,0].scatter(isoDF1['Mr'], isoDF1['iz'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[3,0].set_xlim(MrMin, MrMax)
    ax[3,0].set_ylim(-0.5, 1.6)
    ax[3,0].set_xlabel('Mr')
    ax[3,0].set_ylabel('i-z')

    ax[3,1].scatter(dataDF['Mr'], dataDF['iz'], s=0.1, c='black')
    ax[3,1].scatter(isoDF2['Mr'], isoDF2['iz'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[3,1].set_xlim(MrMin, MrMax)
    ax[3,1].set_ylim(-0.5, 1.6)
    ax[3,1].set_xlabel('Mr')
    ax[3,1].set_ylabel('i-z')

    ax[4,0].scatter(dataDF['Mr'], dataDF['gi'], s=0.1, c='black')
    ax[4,0].scatter(isoDF1['Mr'], isoDF1['gi'], s=0.1, c=isoDF1['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[4,0].set_xlim(MrMin, MrMax)
    ax[4,0].set_ylim(-0.5, 3.8)
    ax[4,0].set_xlabel('Mr')
    ax[4,0].set_ylabel('g-i')

    ax[4,1].scatter(dataDF['Mr'], dataDF['gi'], s=0.1, c='black')
    ax[4,1].scatter(isoDF2['Mr'], isoDF2['gi'], s=0.1, c=isoDF2['FeH'], cmap=plt.cm.jet, alpha=alpha2)
    ax[4,1].set_xlim(MrMin, MrMax)
    ax[4,1].set_ylim(-0.5, 3.8)
    ax[4,1].set_xlabel('Mr')
    ax[4,1].set_ylabel('g-i')

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return


 



def compare2isochronesColorMrAlongLocus(df1, df2, alpha=0.8, title="", plotname='noname.png', tMin = -10, tMax = 272):

    fig,ax = plt.subplots(7,2,figsize=(12,12))

    def plotPanel(xP, yP, df, xName, yName, cName, xMin, xMax, yMin, yMax, xLabel, yLabel, alpha):
        ax[xP,yP].scatter(df[xName], df[yName], s=0.3, c=df[cName], cmap=plt.cm.jet, alpha=alpha)
        ax[xP,yP].set_xlim(xMin, xMax)
        ax[xP,yP].set_ylim(yMin, yMax)
        ax[xP,yP].set_xlabel(xLabel)
        ax[xP,yP].set_ylabel(yLabel)

    # derivatives dMr/dtLoc
    def dMdt(df):
        dMdt = 0*df['Mr']
        for i in range(1,len(dMdt)):
            dMdt[i] = (df['Mr'][i]-df['Mr'][i-1])/(df['tLoc'][i]-df['tLoc'][i-1])      
        dMdt[0] = dMdt[1]
        return dMdt

    tLocMin = tMin
    tLocMax = tMax
    ax[0,0].set_title(title)

    plotPanel(0, 0, df1, 'tLoc', 'ug', 'FeH', tLocMin, tLocMax, -0.7, 4.2, 'tLocus', 'u-g', alpha) 
    plotPanel(0, 1, df2, 'tLoc', 'ug', 'FeH', tLocMin, tLocMax, -0.7, 4.2, 'tLocus', 'u-g', alpha) 

    plotPanel(1, 0, df1, 'tLoc', 'gr', 'FeH', tLocMin, tLocMax, -0.9, 2.2, 'tLocus', 'g-r', alpha) 
    plotPanel(1, 1, df2, 'tLoc', 'gr', 'FeH', tLocMin, tLocMax, -0.9, 2.2, 'tLocus', 'g-r', alpha) 
     
    plotPanel(2, 0, df1, 'tLoc', 'ri', 'FeH', tLocMin, tLocMax, -0.7, 2.9, 'tLocus', 'r-i', alpha) 
    plotPanel(2, 1, df2, 'tLoc', 'ri', 'FeH', tLocMin, tLocMax, -0.7, 2.9, 'tLocus', 'r-i', alpha) 

    plotPanel(3, 0, df1, 'tLoc', 'iz', 'FeH', tLocMin, tLocMax, -0.7, 2.0, 'tLocus', 'i-z', alpha) 
    plotPanel(3, 1, df2, 'tLoc', 'iz', 'FeH', tLocMin, tLocMax, -0.7, 2.0, 'tLocus', 'i-z', alpha) 

    plotPanel(4, 0, df1, 'tLoc', 'gi', 'FeH', tLocMin, tLocMax, -1.7, 5.1, 'tLocus', 'g-i', alpha) 
    plotPanel(4, 1, df2, 'tLoc', 'gi', 'FeH', tLocMin, tLocMax, -1.7, 5.1, 'tLocus', 'g-i', alpha) 
  
    plotPanel(5, 0, df1, 'tLoc', 'Mr', 'FeH', tLocMin, tLocMax, 20.1, -5.1, 'tLocus', 'Mr', alpha) 
    plotPanel(5, 1, df2, 'tLoc', 'Mr', 'FeH', tLocMin, tLocMax, 20.1, -5.1, 'tLocus', 'Mr', alpha) 

    # derivatives dMr/dtLoc
    if (1):
        df1['dMdt'] = dMdt(df1)
        df2['dMdt'] = dMdt(df2)
        plotPanel(6, 0, df1, 'tLoc', 'dMdt', 'FeH', tLocMin, tLocMax, -10.7, 10.3, 'tLocus', 'dMr/dtLoc', alpha) 
        plotPanel(6, 1, df2, 'tLoc', 'dMdt', 'FeH', tLocMin, tLocMax, -10.7, 10.3, 'tLocus', 'dMr/dtLoc', alpha) 

    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return





def compare2isochronesColorMrAlongLocus2(df1, df2, alpha=0.8, title="", plotname='compare2isochronesColorMrAlongLocus.png'):

    fig,ax = plt.subplots(7,2,figsize=(12,12))

    def plotPanel(xP, yP, df, xName, yName, cName, xMin, xMax, yMin, yMax, xLabel, yLabel, alpha):
        ax[xP,yP].scatter(df[xName], df[yName], s=0.3, c=df[cName], cmap=plt.cm.jet, alpha=alpha)
        ax[xP,yP].set_xlim(xMin, xMax)
        ax[xP,yP].set_ylim(yMin, yMax)
        ax[xP,yP].set_xlabel(xLabel)
        ax[xP,yP].set_ylabel(yLabel)

    # derivatives dMr/dtLoc
    def dMdt(df):
        dMdt = 0*df['Mr']
        for i in range(1,len(dMdt)):
            dMdt[i] = (df['Mr'][i]-df['Mr'][i-1])/(df['tLoc'][i]-df['tLoc'][i-1])      
        dMdt[0] = dMdt[1]
        return dMdt

    tLocMin = -10
    tLocMax = 272
    ax[0,0].set_title(title)

    plotPanel(0, 0, df1, 'tLoc', 'ug', 'Mr', tLocMin, tLocMax, -0.7, 4.2, 'tLocus', 'u-g', alpha) 
    plotPanel(0, 1, df2, 'tLoc', 'ug', 'Mr', tLocMin, tLocMax, -0.7, 4.2, 'tLocus', 'u-g', alpha) 

    plotPanel(1, 0, df1, 'tLoc', 'gr', 'Mr', tLocMin, tLocMax, -0.9, 2.2, 'tLocus', 'g-r', alpha) 
    plotPanel(1, 1, df2, 'tLoc', 'gr', 'Mr', tLocMin, tLocMax, -0.9, 2.2, 'tLocus', 'g-r', alpha) 
     
    plotPanel(2, 0, df1, 'tLoc', 'ri', 'Mr', tLocMin, tLocMax, -0.7, 2.9, 'tLocus', 'r-i', alpha) 
    plotPanel(2, 1, df2, 'tLoc', 'ri', 'Mr', tLocMin, tLocMax, -0.7, 2.9, 'tLocus', 'r-i', alpha) 

    plotPanel(3, 0, df1, 'tLoc', 'iz', 'Mr', tLocMin, tLocMax, -0.7, 2.0, 'tLocus', 'i-z', alpha) 
    plotPanel(3, 1, df2, 'tLoc', 'iz', 'Mr', tLocMin, tLocMax, -0.7, 2.0, 'tLocus', 'i-z', alpha) 

    plotPanel(4, 0, df1, 'tLoc', 'gi', 'Mr', tLocMin, tLocMax, -1.7, 4.1, 'tLocus', 'g-i', alpha) 
    plotPanel(4, 1, df2, 'tLoc', 'gi', 'Mr', tLocMin, tLocMax, -1.7, 4.1, 'tLocus', 'g-i', alpha) 
  
    plotPanel(5, 0, df1, 'tLoc', 'Mr', 'Mr', tLocMin, tLocMax, 15.1, -5.1, 'tLocus', 'Mr', alpha) 
    plotPanel(5, 1, df2, 'tLoc', 'Mr', 'Mr', tLocMin, tLocMax, 15.1, -5.1, 'tLocus', 'Mr', alpha) 

    # derivatives dMr/dtLoc
    df1['dMdt'] = dMdt(df1)
    df2['dMdt'] = dMdt(df2)
    plotPanel(6, 0, df1, 'tLoc', 'dMdt', 'Mr', tLocMin, tLocMax, -0.7, 0.3, 'tLocus', 'dMr/dtLoc', alpha) 
    plotPanel(6, 1, df2, 'tLoc', 'dMdt', 'Mr', tLocMin, tLocMax, -0.7, 0.3, 'tLocus', 'dMr/dtLoc', alpha) 

    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return





def compare2isochronesColorDistance(df1, df2, alpha=0.8, title="", plotname='compare2isochronesColorMrAlongLocus.png'):

    fig,ax = plt.subplots(7,2,figsize=(12,12))

    def plotPanel(xP, yP, df, xName, yName, cName, xMin, xMax, yMin, yMax, xLabel, yLabel, alpha):
        ax[xP,yP].scatter(df[xName], df[yName], s=0.3, c=df[cName], cmap=plt.cm.jet, alpha=alpha)
        ax[xP,yP].set_xlim(xMin, xMax)
        ax[xP,yP].set_ylim(yMin, yMax)
        ax[xP,yP].set_xlabel(xLabel)
        ax[xP,yP].set_ylabel(yLabel)

#   tLoc    Mr    FeH     ug      gr      ri      iz   
#   3.46   3.00 -1.50   0.959   0.316   0.116   0.030
    ugT = 0.959
    grT = 0.316
    riT = 0.116
    izT = 0.030 

    df = df1
    df['colorDist'] = np.sqrt((df['ug']-ugT)**2 + (df['gr']-grT)**2 + (df['ri']-riT)**2 + (df['iz']-izT)**2)
    df = df2
    df['colorDist'] = np.sqrt((df['ug']-ugT)**2 + (df['gr']-grT)**2 + (df['ri']-riT)**2 + (df['iz']-izT)**2)
    
    tLocMin = 0
    tLocMax = 17
    ax[0,0].set_title(title)

    plotPanel(0, 0, df1, 'tLoc', 'ug', 'colorDist', tLocMin, tLocMax, -0.2, 4.4, 'tLocus', 'u-g', alpha) 
    plotPanel(0, 1, df2, 'tLoc', 'ug', 'colorDist', tLocMin, tLocMax, -0.2, 4.4, 'tLocus', 'u-g', alpha) 

    plotPanel(1, 0, df1, 'tLoc', 'gr', 'colorDist', tLocMin, tLocMax, -0.2, 2.2, 'tLocus', 'g-r', alpha) 
    plotPanel(1, 1, df2, 'tLoc', 'gr', 'colorDist', tLocMin, tLocMax, -0.2, 2.2, 'tLocus', 'g-r', alpha) 
     
    plotPanel(2, 0, df1, 'tLoc', 'ri', 'colorDist', tLocMin, tLocMax, -0.3, 2.9, 'tLocus', 'r-i', alpha) 
    plotPanel(2, 1, df2, 'tLoc', 'ri', 'colorDist', tLocMin, tLocMax, -0.3, 2.9, 'tLocus', 'r-i', alpha) 

    plotPanel(3, 0, df1, 'tLoc', 'iz', 'colorDist', tLocMin, tLocMax, -0.3, 2.0, 'tLocus', 'i-z', alpha) 
    plotPanel(3, 1, df2, 'tLoc', 'iz', 'colorDist', tLocMin, tLocMax, -0.3, 2.0, 'tLocus', 'i-z', alpha) 

    plotPanel(4, 0, df1, 'tLoc', 'gi', 'colorDist', tLocMin, tLocMax, -0.7, 4.4, 'tLocus', 'g-i', alpha) 
    plotPanel(4, 1, df2, 'tLoc', 'gi', 'colorDist', tLocMin, tLocMax, -0.7, 4.4, 'tLocus', 'g-i', alpha) 
  
    plotPanel(5, 0, df1, 'Mr', 'colorDist', 'colorDist', -4, 10, 0, 0.1, 'Mr', 'CD', alpha) 
    plotPanel(5, 1, df2, 'Mr', 'colorDist', 'colorDist', -4, 10, 0, 0.8, 'Mr', 'CD', alpha) 

    plotPanel(6, 0, df1, 'tLoc', 'colorDist', 'FeH', tLocMin, tLocMax, 0, 0.1, 'tLocus', 'CD', alpha) 
    plotPanel(6, 1, df2, 'tLoc', 'colorDist', 'FeH', tLocMin, tLocMax, 0, 0.8, 'tLocus', 'CD', alpha) 

    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return


# compute closest distance between the two locii, as well as corresponding dMr and dFeH 
def getCD(df1, df2):
    colors = ['ug', 'gr', 'ri', 'iz']
    df1['CD'] = 0*df1['ug']
    df1['CDdMr'] = 0*df1['ug']
    df1['CDdFeH'] = 0*df1['ug']
    for i in range(0,len(df1)):
        CD = 0
        for c in colors:
            CD += (df2[c]-df1[c][i])**2
        df1['CD'][i] = np.sqrt(np.min(CD))
        df1['CDdMr'][i] = df2['Mr'][np.argmin(CD)] - df1['Mr'][i]
        df1['CDdFeH'][i] = df2['FeH'][np.argmin(CD)] - df1['FeH'][i] 


        

def plotRGdegeneracy(df, alpha=0.8, title="", plotname='RGdegeneracy.png'):

    fig, ax = plt.subplots(1,3,figsize=(10,3))

    def plotPanel(nPanel, df, xName, yName, cName, xMin, xMax, yMin, yMax, xLabel, yLabel, alpha):
        #ax[nPanel].plot(df[xName], df[yName])
        ax[nPanel].scatter(df[xName], df[yName], s=1.3, c=df[cName], cmap=plt.cm.jet, alpha=alpha)
        ax[nPanel].set_xlim(xMin, xMax)
        ax[nPanel].set_ylim(yMin, yMax)
        ax[nPanel].set_xlabel(xLabel)
        ax[nPanel].set_ylabel(yLabel)
 
    MrMin = 0.5
    MrMax = 4.0
    ax[0].set_title(title)

    plotPanel(0, df, 'Mr', 'CD', 'FeH', MrMin, MrMax, 0.0, 0.03, 'Mr', 'min. color distance', alpha) 
    plotPanel(1, df, 'Mr', 'CDdMr', 'FeH', MrMin, MrMax, 0.0, 5.0, 'Mr', 'error in Mr', alpha) 
    plotPanel(2, df, 'Mr', 'CDdFeH', 'FeH', MrMin, MrMax, -0.2, 0.8, 'Mr', 'error in [Fe/H]', alpha) 
   
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return



###  analyze Gaia-Stripe82 sample

### test old photoFeH/photoMr:
#  give it a sample definition and analyze a set of quantities:
#   1) 0.2<g-r<0.6
#      generate photoFeH for ugShift, then generate dFeH and photoMr and
#         compare to MrPi0 for piSNR>10: vs. piSNR and umag, gi, photoFeH
#         compare to MrPho0 vs. umag, gi, photoFeH
#   2) all, umag < 21
#      generate photoFeH for ugShift, then generate photoMr and
#         compare to MrPho0 vs. gi



def analyzeGaiaStripe82(df, ugShift=0.0, piSNRmin=10):

    ### definitions of performance quantities 
    # first photoFeH with ugShift=0
    df['photoFeH0'] = lt.photoFeH(df['ug0'], df['gr0'])  
    # and then with provided ugShift
    df['photoFeH'] = lt.photoFeH(df['ug0'] + ugShift, df['gr0'])  
    # and photoMr
    df['photoMr0'] = lt.getMr(df['gi0'], df['photoFeH0'])
    df['photoMr'] = lt.getMr(df['gi0'], df['photoFeH'])
    df['dFeH'] = df['photoFeH0'] - df['FeHEst'] 
    df['dMrBayes'] = df['MrEst'] - df['photoMr0'] 

    ### selection of subsamples 
    # select bright blue stars
    B = df[(df['gr0']>0.2)&(df['gr0']<0.6)&(df['u_mMed']<21)]
    print('from', len(df), 'stars, selected', len(B), 'with 0.2<g-r<0.6 and u<21')
    # and now select parallax sample
    Pi = B[(B['piSNR']>piSNRmin)&(B['MrPi0']>4)]
    print('from', len(B), 'stars, selected', len(Pi), 'with piSNR>', piSNRmin, 'and MrPi>4')
    
    ### # performance analysis
    # 1) Pi sample
    Pi['dMr0'] = Pi['MrPi0'] - Pi['photoMr0'] 
    Pi['dMr'] = Pi['MrPi0'] - Pi['photoMr'] 
    Pi['dMrBayes2'] = Pi['MrEst'] - Pi['MrPi0']
    print(' ')
    print('Pi sample:')
    print('agreement Bayes FeH with photom FeH:', bt.getMedianSigG(bt.basicStats(Pi, 'dFeH'))) 
    print('agreement photom Mr with MrPi0:', bt.getMedianSigG(bt.basicStats(Pi, 'dMr0'))) 
    print('agreement photom Mr with Bayes Mr:', bt.getMedianSigG(bt.basicStats(Pi, 'dMrBayes'))) 
    print('agreement MrPi0 with Bayes Mr:', bt.getMedianSigG(bt.basicStats(Pi, 'dMrBayes2'))) 
    #print('agreement with MrPi0, using ugShift=', ugShift, ':', bt.getMedianSigG(bt.basicStats(Pi, 'dMr'))) 

    # 2) B sample
    B['dMr'] = B['MrPho0'] - B['photoMr0'] 
    B['dMrBayes2'] =  B['MrEst'] - B['MrPho0']
    print(' ')
    print('B sample:')
    print('agreement Bayes FeH with photmo FeH:', bt.getMedianSigG(bt.basicStats(B, 'dFeH'))) 
    print('agreement photom Mr with MrPho0:', bt.getMedianSigG(bt.basicStats(B, 'dMr'))) 
    print('agreement photom Mr with Bayes Mr:', bt.getMedianSigG(bt.basicStats(B, 'dMrBayes'))) 
    print('agreement MrPi0 with Bayes Mr:', bt.getMedianSigG(bt.basicStats(B, 'dMrBayes2'))) 
   

    ### plots
    # 1) 4-panel plot (1) of dMr0 from Pi sample vs. umag, piSNR, gi, photoFeH0 
    # 2) 4-panel plot (2) of dMr from B sample vs. umag, ug, gi, photoFeH0 
    return B, Pi

def gaiaStripe82plot2(df, title="", plotname='gaiaStripe82plot2.png'):

    fig,ax = plt.subplots(2,2,figsize=(8,8))
    alpha = 0.9 
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=18)

    def plotBinnedMedians(i, j, x, y, xMin, xMax, Nbin):
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, xMin, xMax, Nbin) 
        ax[i,j].scatter(xBin, medianBin, s=40, c='red')
        ax[i,j].plot(xBin, medianBin+sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot(xBin, medianBin-sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot([xMin, xMax], [0,0], c='yellow')

    symbSize = 0.01
    
    ax[0,0].scatter(df['u_mMed'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,0].set_xlim(15, 21.2)
    ax[0,0].set_ylim(-1.0, 1.0)
    ax[0,0].set_xlabel('u mag')
    ax[0,0].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(0, 0, df['u_mMed'], df['dMr'], 16, 21, 15)
    
    ax[0,1].scatter(df['ug0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,1].set_xlim(0.6, 2.0)
    ax[0,1].set_ylim(-1.0, 1.0)
    ax[0,1].set_xlabel('u-g')
    ax[0,1].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(0, 1, df['ug0'], df['dMr'], 0.8, 1.8, 10)

    ax[1,0].scatter(df['gr0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,0].set_xlim(0.15, 0.65)
    ax[1,0].set_ylim(-1.0, 1.0)
    ax[1,0].set_xlabel('g-r')
    ax[1,0].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(1, 0, df['gr0'], df['dMr'], 0.20, 0.6, 8)

    ax[1,1].scatter(df['photoFeH0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,1].set_xlim(-2.2, 0.2)
    ax[1,1].set_ylim(-1.0, 1.0)
    ax[1,1].set_xlabel('photo [Fe/H]')
    ax[1,1].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(1, 1, df['photoFeH0'], df['dMr'], -2.0, 0.0, 8)

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return


    

def gaiaStripe82plot1(df, title="", plotname='gaiaStripe82plot1.png'):

    fig,ax = plt.subplots(2,2,figsize=(8,8))
    alpha = 0.9 
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=18)

    def plotBinnedMedians(i, j, x, y, xMin, xMax, Nbin):
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, xMin, xMax, Nbin) 
        ax[i,j].scatter(xBin, medianBin, s=40, c='red')
        ax[i,j].plot(xBin, medianBin+sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot(xBin, medianBin-sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot([xMin, xMax], [0,0], c='yellow')   

        
    ax[0,0].scatter(df['u_mMed'], df['dMr0'], s=0.3, c=df['MrPi0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,0].set_xlim(15.0, 20.0)
    ax[0,0].set_ylim(-1.0, 1.0)
    ax[0,0].set_xlabel('u mag')
    ax[0,0].set_ylabel('MrGeo - photoMr')
    plotBinnedMedians(0, 0, df['u_mMed'], df['dMr0'], 16, 19, 12)
    
    ax[0,1].scatter(df['piSNR'], df['dMr0'], s=0.3, c=df['MrPi0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,1].set_xlim(0.0, 100.0)
    ax[0,1].set_ylim(-1.0, 1.0)
    ax[0,1].set_xlabel('parallax SNR')
    ax[0,1].set_ylabel('MrGeo - photoMr')
    plotBinnedMedians(0, 1, df['piSNR'], df['dMr0'], 10, 90, 8)

    ax[1,0].scatter(df['gr0'], df['dMr0'], s=0.3, c=df['MrPi0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,0].set_xlim(0.15, 0.65)
    ax[1,0].set_ylim(-1.0, 1.0)
    ax[1,0].set_xlabel('g-r')
    ax[1,0].set_ylabel('MrGeo - photoMr')
    plotBinnedMedians(1, 0, df['gr0'], df['dMr0'], 0.25, 0.6, 7)

    ax[1,1].scatter(df['photoFeH0'], df['dMr0'], s=0.3, c=df['MrPi0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,1].set_xlim(-2.2, 0.2)
    ax[1,1].set_ylim(-1.0, 1.0)
    ax[1,1].set_xlabel('photo [Fe/H]')
    ax[1,1].set_ylabel('MrGeo - photoMr')
    plotBinnedMedians(1, 1, df['photoFeH0'], df['dMr0'], -2.0, 0.0, 8)

    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return



def gaiaStripe82plot2(df, title="", plotname='gaiaStripe82plot2.png'):

    fig,ax = plt.subplots(2,2,figsize=(8,8))
    alpha = 0.9 
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=18)

    def plotBinnedMedians(i, j, x, y, xMin, xMax, Nbin):
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, xMin, xMax, Nbin) 
        ax[i,j].scatter(xBin, medianBin, s=40, c='red')
        ax[i,j].plot(xBin, medianBin+sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot(xBin, medianBin-sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot([xMin, xMax], [0,0], c='yellow')

    symbSize = 0.01
    
    ax[0,0].scatter(df['u_mMed'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,0].set_xlim(15, 21.2)
    ax[0,0].set_ylim(-1.0, 1.0)
    ax[0,0].set_xlabel('u mag')
    ax[0,0].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(0, 0, df['u_mMed'], df['dMr'], 16, 21, 15)
    
    ax[0,1].scatter(df['ug0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,1].set_xlim(0.6, 2.0)
    ax[0,1].set_ylim(-1.0, 1.0)
    ax[0,1].set_xlabel('u-g')
    ax[0,1].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(0, 1, df['ug0'], df['dMr'], 0.8, 1.8, 10)

    ax[1,0].scatter(df['gr0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,0].set_xlim(0.15, 0.65)
    ax[1,0].set_ylim(-1.0, 1.0)
    ax[1,0].set_xlabel('g-r')
    ax[1,0].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(1, 0, df['gr0'], df['dMr'], 0.20, 0.6, 8)

    ax[1,1].scatter(df['photoFeH0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,1].set_xlim(-2.2, 0.2)
    ax[1,1].set_ylim(-1.0, 1.0)
    ax[1,1].set_xlabel('photo [Fe/H]')
    ax[1,1].set_ylabel('MrPho - photoMr')
    plotBinnedMedians(1, 1, df['photoFeH0'], df['dMr'], -2.0, 0.0, 8)

    feh = np.linspace(-2.2, 0.2, 220)
    x = (-1.2-feh)/0.1 - 0.5
    fit = 0.8*(1/(1+np.exp(-x)) - 0.5)
    ax[1,1].plot(feh, fit, c='cyan')

    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return




def gaiaStripe82plot3(df, title="", plotname='gaiaStripe82plot3.png'):

    fig,ax = plt.subplots(2,2,figsize=(8,8))
    alpha = 0.9 
    plt.rcParams.update({'font.size': 14})
    plt.rc('axes', labelsize=18)

    def plotBinnedMedians(i, j, x, y, xMin, xMax, Nbin):
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, xMin, xMax, Nbin) 
        ax[i,j].scatter(xBin, medianBin, s=40, c='red')
        ax[i,j].plot(xBin, medianBin+sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot(xBin, medianBin-sigGbin*np.sqrt(nPts), c='red')
        ax[i,j].plot([xMin, xMax], [0,0], c='yellow')

    symbSize = 0.01
    
    ax[0,0].scatter(df['u_mMed'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,0].set_xlim(15, 21.2)
    ax[0,0].set_ylim(-2.0, 2.0)
    ax[0,0].set_xlabel('u mag')
    ax[0,0].set_ylabel('MrPho - Bayes Mr')
    plotBinnedMedians(0, 0, df['u_mMed'], df['dMr'], 16, 21, 15)
    
    ax[0,1].scatter(df['ug0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[0,1].set_xlim(0.6, 2.8)
    ax[0,1].set_ylim(-2.0, 2.0)
    ax[0,1].set_xlabel('u-g')
    ax[0,1].set_ylabel('MrPho - Bayes Mr')
    plotBinnedMedians(0, 1, df['ug0'], df['dMr'], 0.8, 2.7, 19)

    ax[1,0].scatter(df['gi0'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,0].set_xlim(0.15, 2.65)
    ax[1,0].set_ylim(-2.0, 2.0)
    ax[1,0].set_xlabel('g-i')
    ax[1,0].set_ylabel('MrPho - Bayes Mr')
    plotBinnedMedians(1, 0, df['gi0'], df['dMr'], 0.20, 2.5, 23)

    ax[1,1].scatter(df['FeHEst'], df['dMr'], s=symbSize, c=df['MrPho0'], cmap=plt.cm.jet, alpha=alpha)
    ax[1,1].set_xlim(-2.5, 0.5)
    ax[1,1].set_ylim(-2.0, 2.0)
    ax[1,1].set_xlabel('Bayes [Fe/H]')
    ax[1,1].set_ylabel('MrPho - Bayes Mr')
    plotBinnedMedians(1, 1, df['FeHEst'], df['dMr'], -2.0, 0.0, 8)

    feh = np.linspace(-2.2, 0.2, 220)
    x = (-1.2-feh)/0.1 - 0.5
    fit = 0.8*(1/(1+np.exp(-x)) - 0.5)
    ax[1,1].plot(feh, fit, c='cyan')
    
    plt.tight_layout()
    plt.savefig(plotname)
    print('made plot:', plotname)
    plt.show() 
    plt.close("all")
    return


          


def testBayesBSphotoGaiaStripe82(df):

    # FeH: Bayes vs. photo
    df['photoFeH'] = lt.photoFeH(df['ug0'], df['gr0'])  
    df['dBphotoFeH'] = df['FeHEst'] - df['photoFeH']
    print('agreement between Bayes and photo FeH:', bt.getMedianSigG(bt.basicStats(df, 'dBphotoFeH'))) 

    # Mr: MrPi0 vs. photo with Bayes FeH 
    df['MrPiBayesFeH'] = lt.getMr(df['gi0'], df['FeHEst'])
    df['dMrPiBayesFeH'] = df['MrPi0'] - df['MrPiBayesFeH']
    print('agreement between MrPi0 and photo Mr with Bayes FeH:', bt.getMedianSigG(bt.basicStats(df, 'dMrPiBayesFeH'))) 

    # Mr: MrPho0 vs. photo with Bayes FeH 
    df['MrPhoBayesFeH'] = lt.getMr(df['gi0'], df['FeHEst'])
    df['dMrPhoBayesFeH'] = df['MrPho0'] - df['MrPhoBayesFeH']
    print('agreement between MrPho0 and photo Mr with Bayes FeH:', bt.getMedianSigG(bt.basicStats(df, 'dMrPhoBayesFeH'))) 

    # Mr: Bayes vs. photo with Bayes FeH 
    df['photoMrBayesFeH'] = lt.getMr(df['gi0'], df['FeHEst'])
    df['dphotoMrBayesFeH'] = df['MrEst'] - df['photoMrBayesFeH']
    print('agreement between Bayes Mr and photo Mr with Bayes FeH:', bt.getMedianSigG(bt.basicStats(df, 'dphotoMrBayesFeH'))) 

    # Mr: Bayes vs. photo with photo FeH 
    df['photoMrphotoFeH'] = lt.getMr(df['gi0'], df['photoFeH'])
    df['dphotoMrphotoFeH'] = df['MrEst'] - df['photoMrphotoFeH']
    print('agreement between Bayes Mr and photo Mr with photo FeH:', bt.getMedianSigG(bt.basicStats(df, 'dphotoMrphotoFeH'))) 
