
import numpy as np
import pylab as plt
import scipy.stats as stats
from scipy.stats import norm
from astroML.stats import binned_statistic_2d


def plot3diagsData(df1, df2, df3, df4, L0, L1, L2):

    fig=plt.figure(2,figsize=(12,4))
    fig.subplots_adjust(wspace=0.2,hspace=0.2,top=0.97,bottom=0.1,left=0.09,right=0.95)
    plt.rcParams['font.size'] = 12

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
        
    cLocus = 'yellow'
  
    ax=fig.add_subplot(131)
    plotPanel(ax, df1, df2, df3, df4, 'gi', 'rmag', 'g-i', 'SDSS r mag', -1.0, 3.9, 22.0, 13.0)

    ax=fig.add_subplot(132)
    plotPanel(ax, df1, df2, df3, df4, 'ug', 'gr', 'u-g', 'g-r', -1.0, 4.0, -0.7, 1.9)
    ax.plot(L0['ug'], L0['gr'], ls='--', c=cLocus)
    ax.plot(L1['ug'], L1['gr'], ls='-', c=cLocus)
    ax.plot(L2['ug'], L2['gr'], ls='-.', c=cLocus)


    ax=fig.add_subplot(133)
    plotPanel(ax, df1, df2, df3, df4, 'gr', 'ri', 'g-r', 'r-i', -0.7, 1.9, -0.5, 2.3)
    ax.plot(L0['gr'], L0['ri'], ls='--', c=cLocus)
    ax.plot(L1['gr'], L1['ri'], ls='-', c=cLocus)
    ax.plot(L2['gr'], L2['ri'], ls='-.', c=cLocus)
        
    plt.tight_layout()
    plt.savefig('plot3diagsData.png')
    print('made plot: plot3diagsData.png')
    plt.show() 
    plt.close("all")
    return



def plot3HRdiags(df1, df2, df3, L0, L1, L2, Lok):

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

    ax.set_xlim(-1.0,3.8)
    ax.set_ylim(15.0, -2.0)
    ax.set_xlabel('g-i')
    ax.set_ylabel('Mr')

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

    plt.tight_layout()
    plt.savefig('plot3HRdiags.png')
    print('made plot: plot3HRdiags.png')
    plt.show() 
    plt.close("all")
    return


