import pandas as pd
from matplotlib import pyplot as plt
import tools
import tensorflow as tf
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib


def create_predictions(model_path):
    model = tf.keras.models.load_model(model_path)
    x_train, y_train, x_val, y_val, x_test, y_test = tools.create_simulated_data.read_npy("./data/"+model_path.split("_")[-1]+".npy",
                                                                            remove=False)
    p_test = model(x_test)
    return x_test, y_test, p_test


def mean_maha_dist(y, p):
    weights = np.array([0.1, 0.02, 0.1])
    maha_dist = 0
    for i in range(len(y)):
        maha_dist += (np.square((np.array(y[i]).flatten() - np.array(p[i]).flatten()))).mean() / weights[i]**2
    return maha_dist


def data_to_pandas(data):
    x_train, y_train, x_val, y_val, x_test, y_test = data
    x_train = np.concatenate([i for i in x_train], axis=-1)
    y_train = np.concatenate([i for i in y_train], axis=-1)
    #y_train = np.array(y_train).T
    x_val = np.concatenate([i for i in x_val], axis=-1)
    #y_val = np.array(y_val).T
    y_val = np.concatenate([i for i in y_val], axis=-1)
    x_test = np.concatenate([i for i in x_test], axis=-1)
    #y_test = np.array(y_test).T
    y_test = np.concatenate([i for i in y_test], axis=-1)
    train_x_df = pd.DataFrame(x_train, columns=["rmag", "ug", "gr", "ri", "iz",
                                                "rmagErr", "ugErr", "grErr", "riErr", "izErr"])
    val_x_df = pd.DataFrame(x_val, columns=["rmag", "ug", "gr", "ri", "iz",
                                            "rmagErr", "ugErr", "grErr", "riErr", "izErr"])
    test_x_df = pd.DataFrame(x_test, columns=["rmag", "ug", "gr", "ri", "iz",
                                              "rmagErr", "ugErr", "grErr", "riErr", "izErr"])
    train_y_df = pd.DataFrame(y_train, columns=["Mr", "Ar", "FeH"])
    val_y_df = pd.DataFrame(y_val, columns=["Mr", "Ar", "FeH"])
    test_y_df = pd.DataFrame(y_test, columns=["Mr", "Ar", "FeH"])
    return train_x_df, train_y_df, val_x_df, val_y_df, test_x_df, test_y_df

def density_color_mask(x, y, small_size):
    x = np.array(x)
    y = np.array(y)
    xy = (np.vstack([x.ravel(), y.ravel()]))
    if x.shape[0] > small_size:
        small_sample = np.random.choice(range(x.shape[0]), size=small_size)
        z = gaussian_kde(xy[:, small_sample])(xy)
    else:
        z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z

def plot_correlation(true, prediction, axis, name="", small_size=15000):
    #cmaps = ['spring', 'summer', 'autumn', 'winter', 'cool', 'pink', 'Wistia']
    if type(true) is tuple or type(true) is list:
        cmaps = ['Purples', 'Greens', 'Reds', 'Blues', 'Oranges', 'YlOrBr',
                 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
                 'PuBuGn', 'BuGn', 'YlGn']
        axis.set_xlabel(xlabel="true")
        axis.set_ylabel(ylabel="estimated")
        limit = (true[0].min(), true[0].max())
        if (type(name) is not list) and (type(name) is not tuple or len(name) == 1):
            name = [str(i) for i in range(len(true))]
        for i in range(len(true)):
            limit = (min(limit[0], min(true[i].min(), prediction[i].min())),
                     max(limit[1], max(true[i].max(), prediction[i].max())))
            t, p, z = density_color_mask(true[i], prediction[i], small_size)
            axis.scatter(np.array(t).flatten(),
                         np.array(p).flatten(),
                         c=z, label=name[i], s=1, cmap=cmaps[i])
    else:
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        axis.set_xlabel(xlabel=name + " true")
        axis.set_ylabel(ylabel=name + " estimated")
        limit = (min(true.min(), prediction.min()),
                 max(true.max(), prediction.max()))
        t, p, z = density_color_mask(true, prediction, small_size)
        axis.scatter(np.array(t).flatten(),
                     np.array(p).flatten(),
                     c=z, label=name, s=1, cmap=cmaps[0])
    axis.plot ([limit[0],limit[1]], [limit[0],limit[1]], c='red', linestyle="dashed", alpha=0.9)
    lgnd = axis.legend(scatterpoints=1)
    axis.set_facecolor("white")
    limit = (limit[0]-0.05*(limit[1]-limit[0]), limit[1]+0.05*(limit[1]-limit[0]))
    axis.set(xlim=limit, ylim=limit)
    for i, handle in enumerate(lgnd.legendHandles[0:-1]):
        handle.set_sizes([10.0])
        handle.set_color(matplotlib.cm.get_cmap(cmaps[i])(0.7))
    axis.xaxis.set_ticks_position('top')
    axis.yaxis.set_ticks_position('right')
    return axis


def sigGzi(x, axis=None):
    return 0.741*(np.percentile(x, 75, axis=axis)-np.percentile(x, 25, axis=axis))

def sigma(x, axis=None):
    return (np.percentile(x, 84, axis=axis)-np.percentile(x, 16, axis=axis))

def plot_pike(predictedValue, xSample, xErrSample, axis, statistics="median",
               xname="u-g", value_name="", bins= 35, density=False, yname=None):
    from astroML.stats import binned_statistic_2d
    # here we pixelize the ugErr vs. ug diagram and compute various FeH statistics (using true values)
    value_mean, xedges, yedges = binned_statistic_2d(xSample, xErrSample, predictedValue, statistics, bins=bins)

    # Define custom colormaps: Set pixels with no sources to white
    cmap_multicolor = plt.cm.jet.copy()
    cmap_multicolor.set_bad('w', 1.)

    im = axis.imshow(value_mean.T, origin='lower',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    axis.set_xlim(xedges[0], xedges[-1])
    axis.set_ylim(yedges[0], yedges[-1])
    axis.set_xlabel(xname)
    if yname is not None:
        axis.set_ylabel(yname)
    else:
        axis.set_ylabel(xname+' error')
    cb = plt.colorbar(im, ticks=np.linspace(value_mean[~np.isnan(value_mean)].min(),
                                        value_mean[~np.isnan(value_mean)].max(), 3), pad=0.22,
                      format=r'$%.1f$', orientation='horizontal', ax=axis)
    if type(statistics) is str:
        if statistics == "median":
            stat_name = "med"
        elif statistics == "mean":
            stat_name = "mean"
        contour_label = fr''+stat_name+'($'+value_name+'$) in px'
    else:
        contour_label = r'$\sigma$($'+value_name+'$) in px'
    cb.set_label(contour_label)
    if density:
        N, xedges, yedges = binned_statistic_2d(xSample, xErrSample, predictedValue, 'count', bins=bins)
        levels = np.linspace(0, np.log10(N.max()), 7)[2:]
        axis.contour(np.log10(N.T), levels, colors='k', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    return axis

def plot_2Dmap(x, y, axis, xname="", yname="", subsample=3000, xlimit = None, ylimit = None):
    cmaps = 'Blues'
    axis.set_xlabel(xlabel=xname)
    axis.set_ylabel(ylabel=yname)
    x_s, y_s, z = density_color_mask(x, y, subsample)
    # figuring out what limits to have on axis
    if type(xlimit) is float:
        xlimit = (x_s[z>((1-xlimit)*(z.max()-z.min()))+z.min()].min(),
                  x_s[z>((1-xlimit)*(z.max()-z.min()))+z.min()].max())
    elif type(xlimit) is tuple or type(xlimit) is list or type(xlimit) is np.array:
        pass
    else:
        xlimit = 1
        xlimit = (x_s[z>((1-xlimit)*(z.max()-z.min()))+z.min()].min(),
                  x_s[z>((1-xlimit)*(z.max()-z.min()))+z.min()].max())
    if type(ylimit) is float:
        ylimit = (y_s[z>((1-ylimit)*(z.max()-z.min()))+z.min()].min(),
                  y_s[z>((1-ylimit)*(z.max()-z.min()))+z.min()].max())
    elif type(ylimit) is tuple or type(ylimit) is list or type(ylimit) is np.array:
        pass
    else:
        ylimit = 1
        ylimit = (y_s[z>((1-ylimit)*(z.max()-z.min()))+z.min()].min(),
                  y_s[z>((1-ylimit)*(z.max()-z.min()))+z.min()].max())

    # plotting the points with color
    im = axis.scatter(np.array(x_s).flatten(),
                      np.array(y_s).flatten(),
                      c=z, s=1, cmap=cmaps)
    axis.set_facecolor("white")
    axis.set(xlim=xlimit, ylim=ylimit)
    cb = plt.colorbar(im, ticks=[])
    cb.set_label("density on linear scale")
    return axis

def fitMedians(x, y, xMin, xMax, Nbin, min_points_for_median=100):
    # first generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1))
    xBin = np.linspace(0, 1, Nbin)
    nPts = 0*np.linspace(0, 1, Nbin)
    medianBin = 0*np.linspace(0, 1, Nbin)
    sigGbin = -1+0*np.linspace(0, 1, Nbin)
    for i in range(0, Nbin):
        xBin[i] = 0.5*(xEdge[i]+xEdge[i+1])
        yAux = y[(x>xEdge[i])&(x<=xEdge[i+1])]
        if (yAux.size > 0):
            nPts[i] = yAux.size
            medianBin[i] = np.median(yAux)
            # robust estimate of standard deviation: 0.741*(q75-q25)
            sigmaG = 0.741*(np.percentile(yAux, 75)-np.percentile(yAux, 25))
            # uncertainty of the median: sqrt(pi/2)*st.dev/sqrt(N)
            sigGbin[i] = np.sqrt(np.pi/2)*sigmaG/np.sqrt(nPts[i])
        else:
            nPts[i] = yAux.size
            medianBin[i] = np.nan
            sigGbin[i] = np.nan
    xBin = xBin[nPts > min_points_for_median]
    medianBin = medianBin[nPts > min_points_for_median]
    sigGbin = sigGbin[nPts > min_points_for_median]
    return xBin, medianBin, sigGbin

def plot_medians_on_2Dmap(x, y, axis):
    xlimit = axis.get_xlim()
    xBin, medianBin, sigGbin = fitMedians(x, y, xlimit[0], xlimit[1], 60)
    axis.plot([xlimit[0],xlimit[1]], [0,0], c='black', linestyle="dashed", alpha=0.2)
    axis.errorbar(xBin, medianBin, yerr=sigGbin, c='red', xerr=None, linestyle='', marker=".", markersize=2, label="Median")
    lgnd = axis.legend()
    return axis