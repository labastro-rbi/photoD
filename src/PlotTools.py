import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
from astroML.stats import binned_statistic_2d
import LocusTools as lt 

# def makeDMhistogram(x, xMin=1, xMax=22):
# def plot2Dmap(x, y, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):
# def replot2Dmap(Xgrid, Ygrid, Z, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):
# def restore2Dmap(npz, logScale=False):
# def show2Dmap(Xgrid, Ygrid, Z, metadata, xLabel, yLabel, logScale=False):
# def showFlat2Dmap(Z, metadata, xLabel, yLabel, logScale=False):
# def show3Flat2Dmaps(Z1, Z2, Z3, md, xLab, yLab, x0=-99, y0=-99, logScale=False, minFac=1000, cmap='Blues'):
# def showMargPosteriors(x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, trueX1, trueX2): 


# make more sanity plots 
def makeDMhistogram(x, xMin=1, xMax=22):

    ### PLOTTING ###
    plot_kwargs = dict(color='k', linestyle='none', marker='.', markersize=1)
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.98, wspace=0.4, hspace=0.4)

    xOK = x[(x>xMin)&(x<xMax)]
    hist, bins = np.histogram(xOK, bins=50)
    center = (bins[:-1]+bins[1:])/2
    ax1 = plt.subplot(3,1,1)
    ax1.plot(center, hist, drawstyle='steps')   
    ax1.set_xlim(xMin, xMax)
    ax1.set_xlabel(r'$\mathrm{Distance Modulus}$')
    ax1.set_ylabel(r'$\mathrm{dN/dDM}$')

    # save
    plt.savefig('../plots/DMhistogram.png')
    plt.show() 
    return


def plot2Dmap(x, y, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, addMedians=True, logScale=False):
   
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xgrid = np.linspace(xMin, xMax, nXbin)
    ygrid = np.linspace(yMin, yMax, nYbin)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    print('nXbin=', nXbin)
    
    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(vmin=0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")
        
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xlim(xMin, xMax)
    plt.ylim(yMin, yMax)

    if (0):
         # plt.plot([3, 11.5], [3, 11.5], ls='--', lw=1, c='red')
         plt.plot([2.5, 10.5], [0, 0], ls='--', lw=1, c='red')
   
    if (addMedians):     
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, xMin, xMax, nXbin)
        plt.scatter(xBin, medianBin, s=40, c='red')
        plt.plot(xBin, medianBin+sigGbin*np.sqrt(nPts), c='yellow')
        plt.plot(xBin, medianBin-sigGbin*np.sqrt(nPts), c='yellow')
        plt.plot([xMin, xMax], [0,0], c='black')   
        return (xBin, medianBin)
    else:
        # save
        plt.savefig('../plots/2Dmap.png')
        print('made ../plots/2Dmap.png')
        plt.show() 
        return 

def replot2Dmap(Xgrid, Ygrid, Z, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    return 



def makePriorMosaic(priors, rObs, mapIndex, rootname, Npriors, Nrmag, titles, coords, xLabel='[Fe/H]', yLabel='$M_r$', logScale=False):

    import matplotlib.cm as cm
    
    if (logScale):
        from matplotlib.colors import LogNorm

    maps = {}
    for i in range(0, Npriors):
        prior = priors[i]
        for j in range(0, Nrmag):
             npz = prior[mapIndex[j]]
             maps[i,j] = npz['kde'].reshape(npz['xGrid'].shape)
             if ((i==0)&(j==0)):
                xMin = npz['metadata'][0]
                xMax = npz['metadata'][1]
                nXbin = npz['metadata'][2]
                yMin = npz['metadata'][3]
                yMax = npz['metadata'][4]
                nYbin = npz['metadata'][5]

    fig, axs = plt.subplots(Npriors, Nrmag, figsize=(9, 6))
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.98, wspace=0.1, hspace=0.1)
     
    # cb = plt.colorbar()
    # plt.colorbar(cmap=cm.hot, vmin=1.2, vmax=4.3)

    for i in range(0, Npriors):
        for j in range(0, Nrmag):
            if (logScale):
                # plot image
                axs[i,j].imshow(maps[i,j],
                               origin='lower', aspect='auto',
                               extent=[xMin, xMax, yMin, yMax], cmap='Blues',                             
                               norm=LogNorm(0.001, vmax=maps[0,0].max()))
            else: 
                axs[i,j].imshow(maps[i,j],
                               origin='lower', aspect='auto',
                               extent=[xMin, xMax, yMin, yMax], cmap='Blues') 
            axs[i,j].plot([xMin, xMax], [5.0, 5.0], c='red', ls='--', lw=1)
            axs[i,j].plot([0, 0], [yMin, yMax], c='red', ls='--', lw=1)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            if (i==Npriors-1):
                axs[i,j].set(xlabel=xLabel)
                axs[i,j].set_xticks([-2, -1, 0])
            if (j==0):
                axs[i,j].set(ylabel=yLabel)
                axs[i,j].set_yticks([0, 5, 10, 15])
                axs[i,j].text(-2.4, 15.2, coords[i])
            if (i==0):
                axs[i,j].set(title=titles[j])

    plt.savefig(rootname+'priorMosaic.png')
    plt.show() 
    return 



def restore2Dmap(npz, logScale=False):
    
    im = npz['kde'].reshape(npz['xGrid'].shape)
    xMin = npz['metadata'][0]
    xMax = npz['metadata'][1]
    nXbin = npz['metadata'][2]
    yMin = npz['metadata'][3]
    yMax = npz['metadata'][4]
    nYbin = npz['metadata'][5]
    xLabel = npz['labels'][0]
    yLabel = npz['labels'][1]
    
    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(im,
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=im.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(im,
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    return


def show2Dmap(Xgrid, Ygrid, Z, metadata, xLabel, yLabel, logScale=False):

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = metadata[2]
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = metadata[5]

    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('x.png')
    plt.show() 


def showFlat2Dmap(Z, metadata, xLabel, yLabel, logScale=False):

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = metadata[2]
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = metadata[5]
    
    Xpts = nXbin.astype(int)
    Ypts = nYbin.astype(int)
    im = Z.reshape((Xpts, Ypts))

    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(im.T,
               origin='upper', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(im.T,
               origin='upper', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('x.png')
    plt.show() 
    
  
    
def show3Flat2Dmaps(Z1, Z2, Z3, md, xLab, yLab, x0=-99, y0=-99, logScale=False, minFac=1000, cmap='Blues'):

    # unpack metadata
    xMin = md[0]
    xMax = md[1]
    nXbin = md[2]
    yMin = md[3]
    yMax = md[4]
    nYbin = md[5]
    # set local variables and
    myExtent=[xMin, xMax, yMin, yMax]
    Xpts = nXbin.astype(int)
    Ypts = nYbin.astype(int)
    # reshape flattened input arrays to get "images"
    im1 = Z1.reshape((Xpts, Ypts))
    im2 = Z2.reshape((Xpts, Ypts))
    im3 = Z3.reshape((Xpts, Ypts))
    print('pts:', Xpts, Ypts)
    
    showTrue = False
    if ((x0>-99)&(y0>-99)):
        showTrue = True
        
    def oneImage(ax, image, extent, minFactor, title, showTrue, x0, y0, logScale=True, cmap='Blues'):
        im = image/image.max()
        ImMin = im.max()/minFactor
        if (logScale):
            cmap = ax.imshow(im.T,
               origin='upper', aspect='auto', extent=extent,
               cmap=cmap, norm=LogNorm(ImMin, vmax=im.max()))
            ax.set_title(title)
        else:
            cmap = ax.imshow(im.T,
               origin='upper', aspect='auto', extent=extent,
               cmap=cmap)
            ax.set_title(title)
        if (showTrue):
            ax.scatter(x0, y0, s=150, c='red') 
            ax.scatter(x0, y0, s=40, c='yellow') 
        return cmap
 
    fig, axs = plt.subplots(1,3,figsize=(14,4))

    # plot  
    from matplotlib.colors import LogNorm
    cmap = oneImage(axs[0], im1, myExtent, minFac, 'Prior', showTrue, x0, y0, logScale=logScale)
    cmap = oneImage(axs[1], im2, myExtent, minFac, 'Likelihood', showTrue, x0, y0, logScale=logScale)
    cmap = oneImage(axs[2], im3, myExtent, minFac, 'Posterior', showTrue, x0, y0, logScale=logScale)

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    if (0):
        cb = fig.colorbar(cmap, ax=cax)
        if (logScale):
            cb.set_label("density on log scale")
        else:
            cb.set_label("density on linear scale")

    for ax in axs.flat:
        ax.set(xlabel=xLab, ylabel=yLab)

    plt.savefig('../plots/bayesPanels.png')
    plt.show() 

    
def sigGzi(x):
    return 0.741*(np.percentile(x,75)-np.percentile(x,25))


def qp(df, Qname, Dname='Mr'):
    # define custom colormaps: Set pixels with no sources to white
    cmap = plt.cm.copper
    cmap.set_bad('w', 1.)
    cmap_multicolor = plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)

    print('plots for', Qname)
    dFeH_count, xedges, yedges = binned_statistic_2d(df['Mr'], df['FeH'], df[Qname], 'count', bins=50)
    # dFeH_mean, xedges, yedges = binned_statistic_2d(df['Mr'], df['FeH'], df[Qname], 'mean', bins=25)
    # try median
    dFeH_mean, xedges, yedges = binned_statistic_2d(df['Mr'], df['FeH'], df[Qname], 'median', bins=25)
    dFeH_sig, xedges, yedges = binned_statistic_2d(df['Mr'], df['FeH'], df[Qname], sigGzi, bins=25)
    
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.4, left=0.1, right=0.95, bottom=0.12, top=0.95)

    ax = plt.subplot(131, yticks=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    plt.imshow(dFeH_mean.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlabel(r'$\mathrm{Mr}$')
    plt.ylabel(r'$\mathrm{[Fe/H]}$')
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])

    cb = plt.colorbar(ticks=np.linspace(-0.5, 0.5, 5), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (Dname=='Mr'):
        cb.set_label(r'$\mathrm{mean\ \Delta M_r\ in\ pixel}$')
    else:
        if (Dname=='Ar'):
            cb.set_label(r'$\mathrm{mean\ \Delta A_r\ in\ pixel}$')
        else:
            cb.set_label(r'$\mathrm{mean\ \Delta Fe/H]\ in\ pixel}$') 

    plt.clim(-0.5, 0.5)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])


    ax = plt.subplot(132, yticks=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    plt.imshow(dFeH_sig.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlabel(r'$\mathrm{Mr}$')
    plt.ylabel(r'$\mathrm{[Fe/H]}$')
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])

    cb = plt.colorbar(ticks=np.linspace(-0.5, 0.5, 5), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (Dname=='Mr'):
        cb.set_label(r'$\mathrm{\sigma_G(M_r)\ in\ pixel}$')
    else:
        if (Dname=='Ar'):
            cb.set_label(r'$\mathrm{\sigma_G(A_r)\ in\ pixel}$')     
        else:
            cb.set_label(r'$\mathrm{\sigma_G[Fe/H]\ in\ pixel}$')
    plt.clim(0, 0.5)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    plt.show() 

def plot3qp(df, q, Dname='Mr'):
    qp(df, q, Dname)    
    qp(df, q+'ML', Dname)
    qp(df, q+'2', Dname)




    
# like qp but add the third panel with chi2 
def qpB(df, Qname, Dname='Mr', cmd=False, estQ=False):
    # define custom colormaps: Set pixels with no sources to white
    cmap = plt.cm.copper
    cmap.set_bad('w', 1.)
    cmap_multicolor = plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)

    print('plots for', Qname)
    if (cmd):
        xAxis = 'gi'
        yAxis = 'umag'
        yTicks = [16, 18, 20, 22, 24, 26, 28]
        xlabel = '$\mathrm{g-i}$'
        ylabel = '$\mathrm{u mag}$'
    else:
        if estQ:
            xAxis = 'MrEst'
            yAxis = 'FeHEst'
            yTicks = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5] 
            xlabel = 'Mr estimate'
            ylabel = '[Fe/H] estimate'
        else:
            xAxis = 'Mr'
            yAxis = 'FeH'
            yTicks = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5] 
            xlabel = '$\mathrm{Mr}$'
            ylabel = '$\mathrm{[Fe/H]}$'
        
    # counts
    dFeH_count, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df[Qname], 'count', bins=50)
    # median
    dFeH_mean, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df[Qname], 'median', bins=25)
    # scatter
    dFeH_sig, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df[Qname], sigGzi, bins=25)
    # chi2 
    dFeH_chi2, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df[Qname+'Norm'], sigGzi, bins=25)
    
    
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.4, left=0.1, right=0.95, bottom=0.12, top=0.95)

    ## mean value 
    ax = plt.subplot(131, yticks=yTicks)
    plt.imshow(dFeH_mean.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (cmd):
        plt.xlim(0.0, 4.0)
        plt.ylim(27, 15)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])

        
    if (Dname=='Mr'):
        cb = plt.colorbar(ticks=np.linspace(-0.4, 0.4, 5), pad=0.22, format=r'$%.1f$', orientation='horizontal')
        cb.set_label(r'$\mathrm{mean\ \Delta M_r\ in\ pixel}$')
        plt.clim(-0.4, 0.4)
    else:
        if (Dname=='Ar'):
            cb = plt.colorbar(ticks=np.linspace(-0.3, 0.3, 7), pad=0.22, format=r'$%.1f$', orientation='horizontal')
            cb.set_label(r'$\mathrm{mean\ \Delta A_r\ in\ pixel}$')
            plt.clim(-0.15, 0.15)
        else:
            cb = plt.colorbar(ticks=np.linspace(-0.4, 0.4, 5), pad=0.22, format=r'$%.1f$', orientation='horizontal')
            cb.set_label(r'$\mathrm{mean\ \Delta [Fe/H]\ in\ pixel}$') 
            plt.clim(-0.4, 0.4)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    ### sigmaG scatter 
    ax = plt.subplot(132, yticks=yTicks)
    plt.imshow(dFeH_sig.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    if (cmd):
        plt.xlim(0.0, 4.0)
        plt.ylim(28, 15)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
 
    cb = plt.colorbar(ticks=np.linspace(0, 0.5, 6), pad=0.22, format=r'$%.1f$', orientation='horizontal')

    if (Dname=='Mr'):
        plt.clim(0, 0.5)
        cb.set_label(r'$\mathrm{\sigma_G(M_r)\ in\ pixel}$')
    else:
        if (Dname=='Ar'):
            plt.clim(0, 0.1)
            cb.set_label(r'$\mathrm{\sigma_G(A_r)\ in\ pixel}$')     
        else:
            plt.clim(0, 0.5)
            cb.set_label(r'$\mathrm{\sigma_G[Fe/H]\ in\ pixel}$')
  
    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    

    ### chi2 
    ax = plt.subplot(133, yticks=yTicks)
    plt.imshow(dFeH_chi2.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    if (cmd):
        plt.xlim(0.0, 4.0)
        plt.ylim(28, 15)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    cb = plt.colorbar(ticks=np.linspace(0, 2, 5), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (Dname=='Mr'):
        cb.set_label(r'$\mathrm{\chi^2(M_r)\ in\ pixel}$')
    else:
        if (Dname=='Ar'):
            cb.set_label(r'$\mathrm{\chi^2(A_r)\ in\ pixel}$')     
        else:
            cb.set_label(r'$\mathrm{\chi^2[Fe/H]\ in\ pixel}$')
    plt.clim(0, 2)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    if estQ:
        outname = '../plots/qpB_estQ_'+Dname+'.png'
    else:
        outname = '../plots/qpB_'+Dname+'.png'
        
    plt.savefig(outname)
    print('made plot:', outname)
    plt.show() 
    plt.close("all")
    return 



    
    
# like qpB but plot the three fitted quantities (Mr, FeH, Ar) in the u vs. g-i cmd 
def qpBcmd(df, color='gi', mag='umag', scatter=False):

    cmd = True
    # define custom colormaps: Set pixels with no sources to white
    cmap = plt.cm.copper
    cmap.set_bad('w', 1.)
    cmap_multicolor = plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)

    if (cmd):
        xAxis = color
        yAxis = mag
        yTicks = [16, 18, 20, 22, 24, 26, 28]
        xlabel = '$\mathrm{g-i}$'
        ylabel = '$\mathrm{u mag}$'
        xMin = 0.0
        xMax = 4.0
        yMin = 27
        yMax = 14
    else:
        xAxis = 'Mr'
        yAxis = 'FeH'
        yTicks = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5] 
        xlabel = '$\mathrm{Mr}$'
        ylabel = '$\mathrm{[Fe/H]}$'
        
    # counts
    dFeH_count, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['Mr'], 'count', bins=50)
    # median
    dFeH_mean1, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['Mr'], 'mean', bins=25)
    dFeH_mean2, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['FeH'], 'mean', bins=25)
    dFeH_mean3, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['Ar'], 'mean', bins=25)
    # scatter
    dFeH_sig1, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['Mr'], sigGzi, bins=25)
    dFeH_sig2, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['FeH'], sigGzi, bins=25)
    dFeH_sig3, xedges, yedges = binned_statistic_2d(df[xAxis], df[yAxis], df['Ar'], sigGzi, bins=25)
     
    
    # Create figure and subplots
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.4, left=0.1, right=0.95, bottom=0.12, top=0.95)

    ## Mr 
    ax = plt.subplot(131, yticks=yTicks)
    plt.imshow(dFeH_mean1.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (cmd):
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])

    cb = plt.colorbar(ticks=np.linspace(-1, 13, 8), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (scatter):
        cb.set_label(r'$\mathrm{\sigma_G(M_r)\ in\ pixel}$')
    else:
        cb.set_label(r'$\mathrm{mean\ M_r\ in\ pixel}$')
    plt.clim(-1, 13)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    ### [Fe/H]
    ax = plt.subplot(132, yticks=yTicks)
    plt.imshow(dFeH_mean2.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    if (cmd):
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
 
    cb = plt.colorbar(ticks=np.linspace(-2.0, 0.0, 5), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (scatter):
        cb.set_label(r'$\mathrm{\sigma_G([Fe/H])\ in\ pixel}$')
    else:
        cb.set_label(r'$\mathrm{mean\ [Fe/H]\ in\ pixel}$')
    plt.clim(-2, 0)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    

    ### Ar
    ax = plt.subplot(133, yticks=yTicks)
    plt.imshow(dFeH_mean3.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    if (cmd):
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
    else:
        plt.xlim(xedges[0], xedges[-1])
        plt.ylim(yedges[0], yedges[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    cb = plt.colorbar(ticks=np.linspace(0, 0.5, 6), pad=0.22,
                  format=r'$%.1f$', orientation='horizontal')
    if (scatter):
        cb.set_label(r'$\mathrm{\sigma_G(Ar)\ in\ pixel}$')
    else:
        cb.set_label(r'$\mathrm{mean\ Ar\ in\ pixel}$')
    plt.clim(0, 0.25)

    # density contours 
    levels = np.linspace(0, np.log10(dFeH_count.max()), 5)[2:]
    plt.contour(np.log10(dFeH_count.T), levels, colors='w',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.contour(np.log10(dFeH_count.T), levels, colors='k', linestyles='dashed',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    plt.savefig('../plots/qpBcmd.png')
    print('made plot: ../plots/qpBcmd.png')
    plt.show() 
    plt.close("all")
    return 

