import os
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def density_color_mask(x, y, small_size):
    x = np.array(x)
    y = np.array(y)
    xy = (np.vstack([x.ravel(), y.ravel()]))
    if x.shape[0] > small_size:
        small_sample = np.random.choice(range(x.shape[0]), size=small_size)
        z = gaussian_kde(xy[:, small_sample], bw_method='silverman')(xy)
    else:
        z = gaussian_kde(xy, bw_method='silverman')(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z


def plot_correlation(true, prediction, axis, name="", small_size=15000, true_name="true"):
    if type(true) is tuple or type(true) is list:
        cmaps = ['Purples', 'Greens', 'Reds', 'Blues', 'Oranges', 'YlOrBr',
                 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
                 'PuBuGn', 'BuGn', 'YlGn']
        axis.set_xlabel(xlabel=true_name)
        axis.set_ylabel(ylabel="NN estimated")
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
        axis.set_xlabel(xlabel=name + " " + true_name)
        axis.set_ylabel(ylabel=name + " NN estimated")
        limit = (min(true.min(), prediction.min()),
                 max(true.max(), prediction.max()))
        t, p, z = density_color_mask(true, prediction, small_size)
        axis.scatter(np.array(t).flatten(),
                     np.array(p).flatten(),
                     c=z, label=name, s=1, cmap=cmaps[0])
    axis.plot([limit[0], limit[1]], [limit[0], limit[1]], c='red', linestyle="dashed", alpha=0.9)
    lgnd = axis.legend(scatterpoints=1, loc='lower right')
    axis.set_facecolor("white")
    limit = (limit[0] - 0.05 * (limit[1] - limit[0]), limit[1] + 0.05 * (limit[1] - limit[0]))
    axis.set(xlim=limit, ylim=limit)
    for i, handle in enumerate(lgnd.legend_handles[0:-1]):
        handle.set_sizes([10.0])
        handle.set_color(matplotlib.cm.get_cmap(cmaps[i])(0.7))
    return axis


def sigzi(x, axis=None):
    return 0.741 * (np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis))


def sigma(x, axis=None):
    return np.percentile(x, 84, axis=axis) - np.percentile(x, 16, axis=axis)


def plot_pike(predicted_value, x_sample, xerr_sample, axis, statistics="median",
              xname="u-g", value_name="", bins=35, density=False, yname=None):
    from astroML.stats import binned_statistic_2d

    # here we pixelize the ugErr vs. ug diagram and compute various FeH statistics (using true values)
    value_mean, xedges, yedges = binned_statistic_2d(x_sample, xerr_sample, predicted_value, statistics, bins=bins)

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
        axis.set_ylabel(xname + ' error')
    cb = plt.colorbar(im, ticks=np.linspace(value_mean[~np.isnan(value_mean)].min(),
                                            value_mean[~np.isnan(value_mean)].max(), 5), pad=0.22,
                      format=r'$%.1f$', orientation='horizontal', ax=axis)
    if type(statistics) is str:
        if statistics == "median":
            contour_label = r'$\mathrm{med\ [' + value_name + r']\ in\ px}$'
        else:
            contour_label = fr'$' + statistics + '\ [' + value_name + ']\ in\ px$'
    else:
        contour_label = r'$\mathrm{\sigma_G \, for [' + value_name + r']\ in\ px}$'
    cb.set_label(contour_label)
    if density:
        n, xedges, yedges = binned_statistic_2d(x_sample, xerr_sample, predicted_value, 'count', bins=bins)
        levels = np.linspace(0, np.log10(n.max()), 7)[2:]
        axis.contour(np.log10(n.T), levels, colors='k', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    return axis


def pike_plot_multiple(predicted_value, x_sample, xerr_sample, true_value, xname="u-g", value_name="", bins=35,
                       density=False, true_name="true"):
    original_rcparams = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        'font.size': 14,  # General font size
        'axes.labelsize': 15,  # X and Y axis label size
        'xtick.labelsize': 15,  # X axis tick size
        'ytick.labelsize': 15,  # Y axis tick size
        'legend.fontsize': 15,  # Legend font size
        'figure.titlesize': 16  # Figure title size
    })
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    fig.tight_layout()
    deltavalue = predicted_value - true_value
    ax[0] = plot_pike(predicted_value, x_sample, xerr_sample, ax[0], statistics="median",
                      xname=xname, value_name=value_name, bins=bins, density=density)

    ax[1] = plot_pike(deltavalue, x_sample, xerr_sample, ax[1], statistics="median",
                      xname=xname, value_name=value_name + "_{NN} - " + value_name + "_{" + true_name + "}", bins=bins,
                      density=density)
    # Set colorbar limits to +-3 sigmaG of the (estimated - true)
    sample_sigmag = sigzi(deltavalue)
    ax[1].images[-1].set_clim(-3 * sample_sigmag, 3 * sample_sigmag)
    ticks = np.linspace(-3 * sample_sigmag, 3 * sample_sigmag, 5)
    ax[1].images[-1].colorbar.set_ticks(ticks)
    ax[1].images[-1].colorbar.set_ticklabels([r'$%.1f$' % t for t in ticks])
    # add median to the plot
    ax[0].set_title(
        r"med[$" + value_name + "_{NN} - " + value_name + "_{" + true_name + "}$] = " + str(round(np.median(deltavalue),
                                                                                                  3)))

    ax[2] = plot_pike(deltavalue, x_sample, xerr_sample, ax[2], statistics=sigzi,
                      xname=xname, value_name=value_name + "_{NN} - " + value_name + "_{" + true_name + "}",
                      bins=bins, density=density)
    # Set colorbar limits to [0, 3 sigmaG of the (estimated - true)]
    ax[2].images[-1].set_clim(0, 3 * sample_sigmag)
    ticks = np.linspace(0, 3 * sample_sigmag, 5)
    ax[2].images[-1].colorbar.set_ticks(ticks)
    ax[2].images[-1].colorbar.set_ticklabels([r'$%.1f$' % t for t in ticks])

    # add sigmaG to the plot
    ax[2].set_title(
        r"$\sigma _G$ [$" + value_name + "_{NN} - " + value_name + "_{" + true_name + "}$] = " + str(
            round(sample_sigmag,
                  3)))
    matplotlib.rcParams.update(original_rcparams)
    return fig


def plot_2d_map(x, y, axis, xname="", yname="", subsample=3000, xlimit=None, ylimit=None):
    cmaps = 'Blues'
    axis.set_xlabel(xlabel=xname)
    axis.set_ylabel(ylabel=yname)
    x_s, y_s, z = density_color_mask(x, y, subsample)
    # figuring out what limits to have on axis
    if type(xlimit) is float:
        xlimit = (x_s[z > ((1 - xlimit) * (z.max() - z.min())) + z.min()].min(),
                  x_s[z > ((1 - xlimit) * (z.max() - z.min())) + z.min()].max())
    elif type(xlimit) is tuple or type(xlimit) is list or type(xlimit) is np.array:
        pass
    else:
        xlimit = 1
        xlimit = (x_s[z > ((1 - xlimit) * (z.max() - z.min())) + z.min()].min(),
                  x_s[z > ((1 - xlimit) * (z.max() - z.min())) + z.min()].max())
    if type(ylimit) is float:
        ylimit = (y_s[z > ((1 - ylimit) * (z.max() - z.min())) + z.min()].min(),
                  y_s[z > ((1 - ylimit) * (z.max() - z.min())) + z.min()].max())
    elif type(ylimit) is tuple or type(ylimit) is list or type(ylimit) is np.array:
        pass
    else:
        ylimit = 1
        ylimit = (y_s[z > ((1 - ylimit) * (z.max() - z.min())) + z.min()].min(),
                  y_s[z > ((1 - ylimit) * (z.max() - z.min())) + z.min()].max())

    # plotting the points with color
    im = axis.scatter(np.array(x_s).flatten(),
                      np.array(y_s).flatten(),
                      c=z, s=1, cmap=cmaps)
    axis.set_facecolor("white")
    axis.set(xlim=xlimit, ylim=ylimit)
    cb = plt.colorbar(im, ticks=[])
    cb.set_label("density on linear scale")
    return axis


def fit_medians(x, y, x_min, x_max, n_bin, min_points_for_median=100):
    # first generate bins
    x_edge = np.linspace(x_min, x_max, (n_bin + 1))
    x_bin = np.linspace(0, 1, n_bin)
    n_pts = 0 * np.linspace(0, 1, n_bin)
    median_bin = 0 * np.linspace(0, 1, n_bin)
    sig_gbin = -1 + 0 * np.linspace(0, 1, n_bin)
    for i in range(0, n_bin):
        x_bin[i] = 0.5 * (x_edge[i] + x_edge[i + 1])
        y_aux = y[(x > x_edge[i]) & (x <= x_edge[i + 1])]
        if y_aux.size > 0:
            n_pts[i] = y_aux.size
            median_bin[i] = np.median(y_aux)
            # robust estimate of standard deviation: 0.741*(q75-q25)
            sigma_g = 0.741 * (np.percentile(y_aux, 75) - np.percentile(y_aux, 25))
            # uncertainty of the median: sqrt(pi/2)*st.dev/sqrt(n)
            sig_gbin[i] = np.sqrt(np.pi / 2) * sigma_g / np.sqrt(n_pts[i])
        else:
            n_pts[i] = y_aux.size
            median_bin[i] = np.nan
            sig_gbin[i] = np.nan
    x_bin = x_bin[n_pts > min_points_for_median]
    median_bin = median_bin[n_pts > min_points_for_median]
    sig_gbin = sig_gbin[n_pts > min_points_for_median]
    return x_bin, median_bin, sig_gbin


def plot_medians_on_2d_map(x, y, axis):
    xlimit = axis.get_xlim()
    x_bin, median_bin, sig_gbin = fit_medians(x, y, xlimit[0], xlimit[1], 60)
    axis.plot([xlimit[0], xlimit[1]], [0, 0], c='black', linestyle="dashed", alpha=0.2)
    axis.errorbar(x_bin, median_bin, yerr=sig_gbin, c='red', xerr=None, linestyle='', marker=".", markersize=2,
                  label="Median")
    __ = axis.legend(loc="lower left")
    return axis


def plot_2d_multiple1(x_test, p, y_test):
    original_rcparams = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        'font.size': 14,  # General font size
        'axes.labelsize': 16,  # X and Y axis label size
        'xtick.labelsize': 14,  # X axis tick size
        'ytick.labelsize': 14,  # Y axis tick size
        'legend.fontsize': 12,  # Legend font size
        'figure.titlesize': 18  # Figure title size
    })
    p = list(p)
    y_test = list(y_test)
    r_mag_sample, ug_sample, gr_sample, ri_sample, iz_sample = (x_test[:, i] for i in range(x_test.shape[1]))
    mrsample, arsample, feh_sample = tuple(
        [np.array(p[i]).reshape((-1)) for i in range(len(p))])  # cleaning shapes of the outputs
    mrsample_true, arsample_true, fehsample_true = (y_test[i][:] for i in range(len(y_test)))
    d_mr = mrsample - mrsample_true
    d_feh = feh_sample - fehsample_true
    fig, ax = plt.subplots(3, figsize=(7, 5))
    fig.set_facecolor('white')
    fig.tight_layout()
    u_mag_sample = ug_sample + gr_sample + r_mag_sample
    ax[2] = plot_2d_map(d_feh, d_mr, ax[2], xlimit=[-1, 1], ylimit=[-1, 1], xname="d[Fe/H]", yname="d[Mr]")
    ax[2] = plot_medians_on_2d_map(d_feh, d_mr, ax[2])
    ax[2].plot([0, 0], [ax[2].get_ylim()[0], ax[2].get_ylim()[1]], c='black', linestyle="dashed", alpha=0.2)

    ax[1] = plot_2d_map(mrsample, d_mr, ax[1], xlimit=[-1, 13], ylimit=[-1, 1], xname="Mr", yname="d[Mr]")
    ax[1] = plot_medians_on_2d_map(mrsample, d_mr, ax[1])

    ax[0] = plot_2d_map(u_mag_sample, d_feh, ax[0], xlimit=[16, 30], ylimit=[-1, 1], xname="u mag",
                        yname="d[Fe/H]")
    ax[0] = plot_medians_on_2d_map(u_mag_sample, d_feh, ax[0])
    _ = [fig.delaxes(i) for i in fig.axes[:] if i.get_label() == '<colorbar>']
    cb = fig.colorbar(fig.axes[0].collections[0], ax=ax.ravel().tolist(), ticks=[])
    cb.set_label("density on linear scale")
    matplotlib.rcParams.update(original_rcparams)
    return fig


def plot_2d_multiple(x_test, p, y_test):
    original_rcparams = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        'font.size': 14,  # General font size
        'axes.labelsize': 14,  # X and Y axis label size
        'xtick.labelsize': 12,  # X axis tick size
        'ytick.labelsize': 12,  # Y axis tick size
        'legend.fontsize': 12,  # Legend font size
        'figure.titlesize': 18  # Figure title size
    })
    p = list(p)
    y_test = list(y_test)
    r_mag_sample, ug_sample, gr_sample, ri_sample, iz_sample = (x_test[:, i] for i in range(x_test.shape[1]))
    mrsample, arsample, feh_sample = tuple(
        [np.array(p[i]).reshape((-1)) for i in range(len(p))])  # cleaning shapes of the outputs
    mrsample_true, arsample_true, fehsample_true = (y_test[i][:] for i in range(len(y_test)))
    d_mr = mrsample - mrsample_true
    d_feh = feh_sample - fehsample_true

    # Use GridSpec to control the layout
    fig = plt.figure(figsize=(7, 5))
    fig.set_facecolor('white')
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05])  # Allocate space for colorbar

    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[2, 0])

    u_mag_sample = ug_sample + gr_sample + r_mag_sample
    ax2 = plot_2d_map(d_feh, d_mr, ax2, xlimit=[-1, 1], ylimit=[-1, 1], xname="d[Fe/H]", yname="d[Mr]")
    ax2 = plot_medians_on_2d_map(d_feh, d_mr, ax2)
    ax2.plot([0, 0], [ax2.get_ylim()[0], ax2.get_ylim()[1]], c='black', linestyle="dashed", alpha=0.2)

    ax1 = plot_2d_map(mrsample, d_mr, ax1, xlimit=[-1, 13], ylimit=[-1, 1], xname="Mr", yname="d[Mr]")
    ax1 = plot_medians_on_2d_map(mrsample, d_mr, ax1)

    ax0 = plot_2d_map(u_mag_sample, d_feh, ax0, xlimit=[16, 30], ylimit=[-1, 1], xname="u mag", yname="d[Fe/H]")
    ax0 = plot_medians_on_2d_map(u_mag_sample, d_feh, ax0)

    _ = [fig.delaxes(i) for i in fig.axes[:] if i.get_label() == '<colorbar>']

    # Add the colorbar to the 4th column
    cb_ax = plt.subplot(gs[:, 1])
    cb = fig.colorbar(ax0.collections[0], cax=cb_ax, ticks=[])
    cb.set_label("density on linear scale")
    fig.tight_layout()
    matplotlib.rcParams.update(original_rcparams)
    return fig


def plot_expected_gauss(p, sigma_p, y_test, outputs, mask=None, nbins=300, true_name="true"):
    original_rcparams = matplotlib.rcParams.copy()
    matplotlib.rcParams.update({
        'font.size': 30,  # General font size
        'axes.labelsize': 32,  # X and Y axis label size
        'xtick.labelsize': 18,  # X axis tick size
        'ytick.labelsize': 18,  # Y axis tick size
        'legend.fontsize': 12,  # Legend font size
        'figure.titlesize': 18  # Figure title size
    })
    p = list(p)
    sigma_p = list(sigma_p)
    y_test = list(y_test)
    if mask is not None:
        p = [p[i][mask] for i in range(len(p))]
        sigma_p = [sigma_p[i][mask] for i in range(len(sigma_p))]
        y_test = [y_test[i][mask] for i in range(len(y_test))]
    fig, ax = plt.subplots(1, 3, figsize=(3 * 5, 5))
    for i in range(len(ax)):
        ax[i].set_xlabel(
            r"$\frac{" + outputs[i] + "_{NN} - " + outputs[i] + "_{" + true_name + "}}{ \sigma_{" + outputs[i] + "} }$")
        ax[i].set_yticks([])
        points_norm = (p[i][sigma_p[i] != 0] - y_test[i][sigma_p[i] != 0]) / sigma_p[i][sigma_p[i] != 0]
        points_norm = points_norm[(points_norm > -3 * sigzi(points_norm)) &
                                  (points_norm < 3 * sigzi(points_norm))]
        hist, bins = np.histogram(points_norm, bins=nbins)
        ax[i].bar(bins[:-1], hist, width=np.diff(bins))
        norm_x = np.linspace(-3 * sigzi(points_norm), 3 * sigzi(points_norm), 100)
        norm_y = scipy.stats.norm.pdf(norm_x, 0, 1) * points_norm.shape[0] * np.diff(bins)[0]
        ax[i].plot(norm_x, norm_y, color="red", label="N(0,1)")
        ax[i].legend(loc="upper right")
    matplotlib.rcParams.update(original_rcparams)
    return fig


def get_model_metrics(x, y, p, sigma_p, save_path=None, show_plot=True, true_name="true"):
    fig_correlation, ax = plt.subplots(figsize=(5, 5))
    names = ["Mr", "Ar", "FeH"]
    name = ["", "", ""]
    for i in range(len(names)):
        name[i] = names[i] + " (corr=" + str(round(np.corrcoef(y[:, i], p[:, i])[0, 1], 2)) + ")"
    plot_correlation([y[:, i] for i in range(y.shape[-1])], [p[:, i] for i in range(p.shape[-1])],
                     ax, name=name, small_size=200, true_name=true_name)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_path[-1] != "/":
            save_path += "/"
        fig_correlation.savefig(save_path + "correlation.png", dpi=300, bbox_inches='tight')

    fig_correlation_separate, ax = plt.subplots(len(names), figsize=(5, len(names) * 4))
    fig_correlation_separate.set_facecolor('white')
    for i in range(len(names)):
        name = names[i]
        ax[i] = plot_correlation(y[:, i], p[:, i], ax[i], name=name, small_size=200, true_name=true_name)
    if save_path is not None:
        fig_correlation_separate.savefig(save_path + "correlation_separate.png", dpi=300, bbox_inches='tight')

    mrsample_true, arsample_true, fehsample_true = (y[:, i] for i in range(y.shape[-1]))

    r_mag_sample, ug_sample, gr_sample, ri_sample, iz_sample = (x[0][:, i] for i in range(x[0].shape[1]))
    r_mag_err_sample, ug_err_sample, gr_err_sample, ri_err_sample, iz_err_sample = (x[1][:, i] for i in
                                                                                    range(x[1].shape[1]))
    mrsample, arsample, fehsample = tuple(p[:, i] for i in range(p.shape[-1]))  # cleaning shapes of the outputs
    figar = pike_plot_multiple(arsample[ug_err_sample < 0.5], ug_sample[ug_err_sample < 0.5],
                               ug_err_sample[ug_err_sample < 0.5],
                               arsample_true[ug_err_sample < 0.5], value_name="Ar", true_name=true_name)
    figmr = pike_plot_multiple(mrsample[ug_err_sample < 0.5], ug_sample[ug_err_sample < 0.5],
                               ug_err_sample[ug_err_sample < 0.5],
                               mrsample_true[ug_err_sample < 0.5], value_name="Mr", true_name=true_name)
    figfeh = pike_plot_multiple(fehsample[ug_err_sample < 0.5], ug_sample[ug_err_sample < 0.5],
                                ug_err_sample[ug_err_sample < 0.5],
                                fehsample_true[ug_err_sample < 0.5], value_name="Fe/H", true_name=true_name)
    if save_path is not None:
        figar.savefig(save_path + "pike_ar.png", dpi=300, bbox_inches='tight')
        figmr.savefig(save_path + "pike_mr.png", dpi=300, bbox_inches='tight')
        figfeh.savefig(save_path + "pike_feh.png", dpi=300, bbox_inches='tight')

    fig_error_2d = plot_2d_multiple(x[0], (p[:, i] for i in range(p.shape[-1])), (y[:, i] for i in range(y.shape[-1])))
    if save_path is not None:
        plt.savefig(save_path + "error_2d.png", dpi=300, bbox_inches='tight')
    fig_chi_squared = plot_expected_gauss((p[:, i] for i in range(p.shape[-1])),
                                          (sigma_p[:, i] for i in range(sigma_p.shape[-1])),
                                          (y[:, i] for i in range(y.shape[-1])), names, true_name=true_name)
    if save_path is not None:
        plt.savefig(save_path + "chi_squared.png", dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig_correlation, fig_correlation_separate, figar, figmr, figfeh, fig_error_2d, fig_chi_squared


if __name__ == "__main__":
    print(matplotlib.rcParams)
