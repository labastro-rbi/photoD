
import numpy as np
import pylab as plt
import scipy.stats as stats
from scipy.stats import norm
from astroML.stats import binned_statistic_2d


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
