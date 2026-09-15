import numpy as np
import matplotlib.pyplot as plt
import tools
import scipy.stats
import tensorflow as tf
import pydot
from IPython.display import Image, display

#some useful graphs generating function for later
def pike_plot_multiple (predictedValue, xSample, xErrSample, trueValue, xname="u-g", value_name="", bins= 35, density=False):
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    deltavalue = predictedValue - trueValue
    ax[0] = tools.eval_tools.plot_pike(predictedValue, xSample, xErrSample, ax[0], statistics="median",
                                       xname=xname, value_name=value_name, bins=bins, density=density)
    ax[0].xaxis.set_ticks_position('top')
    ax[0].yaxis.set_ticks_position('right')

    ax[1] = tools.eval_tools.plot_pike(deltavalue, xSample, xErrSample, ax[1], statistics="median",
                                       xname=xname, value_name=value_name + "_{e} - "+ value_name +"_{t}", bins = bins, density=density)
    # Set colorbar limits to +-3 sigmaG of the (estimated - true)
    sample_sigmag = tools.eval_tools.sigGzi(deltavalue)
    ax[1].images[-1].set_clim(-3 * sample_sigmag, 3 * sample_sigmag)
    ticks = np.unique(np.round(np.linspace(-3 * sample_sigmag, 3 * sample_sigmag, 3), 2))
    ax[1].images[-1].colorbar.set_ticks(ticks)
    ax[1].images[-1].colorbar.set_ticklabels([r'$%.2f$' % t for t in ticks])
    ax[1].xaxis.set_ticks_position('top')
    ax[1].yaxis.set_ticks_position('right')
    # add median to the plot
    #ax[1].set_title(r"$\mu$($"+value_name + "_{e} - "+ value_name +"_{t}$)="+str(round(np.median(deltavalue), 2)))
    print(r"$\mu$($"+value_name + "_{e} - "+ value_name +"_{t}$)="+str(round(np.median(deltavalue), 2)))

    ax[2] = tools.eval_tools.plot_pike(deltavalue, xSample, xErrSample, ax[2], statistics=tools.eval_tools.sigGzi,
                                       xname=xname, value_name=value_name + "_{e} - "+ value_name +"_{t}", bins = bins, density=density)
    # Set colorbar limits to [0, 3 sigmaG of the (estimated - true)]
    ax[2].images[-1].set_clim(0, 3 * sample_sigmag)
    ticks = np.unique(np.round(np.linspace(0, 3 * sample_sigmag, 2), 2))
    ax[2].images[-1].colorbar.set_ticks(ticks)
    ax[2].images[-1].colorbar.set_ticklabels([r'$%.2f$' % t for t in ticks])
    ax[2].xaxis.set_ticks_position('top')
    ax[2].yaxis.set_ticks_position('right')

    #add sigmaG to the plot
    #ax[2].set_title(r"$\sigma$($"+value_name + "_{e} - "+ value_name +"_{t}$)="+str(round(sample_sigmag,2)))
    print(r"$\sigma$($"+value_name + "_{e} - "+ value_name +"_{t}$)="+str(round(sample_sigmag,2)))
    fig.tight_layout()

    return fig

def plot_2D_multiple (x_test, p, y_test):
    r_magSample, ugSample, grSample, riSample, izSample = (x_test[0][:,i] for i in range (x_test[0].shape[1]))
    r_magErrSample, ugErrSample, grErrSample, riErrSample, izErrSample = (x_test[1][:,i] for i in range (x_test[1].shape[1]))
    Mrsample, Arsample, FeHsample = tuple ([np.array(p[i]).reshape((-1)) for i in range (len(p))]) #cleaning shapes of the outputs
    Mrsample_true, Arsample_true, FeHsample_true = (y_test[i][:,0] for i in range (len(y_test)))
    dMr =  Mrsample - Mrsample_true
    dAr = Arsample - Arsample_true
    dFeH = FeHsample - FeHsample_true
    fig, ax = plt.subplots(3, figsize=(6, 6))
    fig.set_facecolor('white')
    fig.tight_layout()
    u_magSample = ugSample+grSample+r_magSample
    ax[2] = tools.eval_tools.plot_2Dmap(dFeH, dMr, ax[2], xlimit=[-1,1], ylimit=[-1,1], xname="d[Fe/H]", yname="dMr")
    ax[2] = tools.eval_tools.plot_medians_on_2Dmap(dFeH, dMr, ax[2])
    ax[2].plot([0,0], [ax[2].get_ylim()[0], ax[2].get_ylim()[1]], c='black', linestyle="dashed", alpha=0.2)

    ax[1] = tools.eval_tools.plot_2Dmap(Mrsample, dMr, ax[1], xlimit=[-1,13], ylimit=[-1,1], xname="Mr", yname="dMr")
    ax[1] = tools.eval_tools.plot_medians_on_2Dmap(Mrsample, dMr, ax[1])

    ax[0] = tools.eval_tools.plot_2Dmap(u_magSample, dFeH, ax[0], xlimit=[16,30], ylimit=[-1,1], xname="u mag", yname="d[Fe/H]")
    ax[0] = tools.eval_tools.plot_medians_on_2Dmap(u_magSample, dFeH, ax[0])
    _ = [fig.delaxes(i) for i in fig.axes[:] if i.get_label() == '<colorbar>']
    cb = fig.colorbar(fig.axes[0].collections[0], ax=ax.ravel().tolist(), ticks=[])
    cb.set_label("density on linear scale")

    return fig

def plot_sigma_histogram (ax, sigma, limits, name, mask=None):
    if mask is not None:
        sigma = sigma[mask]
    ax.set_xlabel(r"estimated $\sigma_{"+name+"}$")
    ax.hist (sigma, bins=300, range=limits)
    return ax

def plot_expected_gauss(p, sigma_p, y_test, outputs, ax, mask=None, nbins=300):
    if mask is not None:
        p = [p[i][mask] for i in range(len(p))]
        sigma_p = [sigma_p[i][mask] for i in range(len(sigma_p))]
        y_test = [y_test[i][mask] for i in range(len(y_test))]
    #fig, ax = plt.subplots(1, 3, figsize=(3*5,5))
    outputs = ["Mr", "Ar", "FeH"]
    for i in range(len(ax)):
        ax[i].set_xlabel(r"$\frac{"+outputs[i]+"_{est} - "+outputs[i]+"_{true}}{ \sigma_{"+outputs[i]+"} }$")
        ax[i].set_yticks([])
        points_norm = (p[i][sigma_p[i]!=0]-y_test[i][sigma_p[i]!=0])/sigma_p[i][sigma_p[i]!=0]
        points_norm = points_norm[(points_norm>-3*tools.eval_tools.sigGzi(points_norm)) & (points_norm<3*tools.eval_tools.sigGzi(points_norm))]
        hist, bins = np.histogram(points_norm, bins=nbins)
        ax[i].bar(bins[:-1], hist, width=np.diff(bins))
        norm_x = np.linspace(-3*tools.eval_tools.sigGzi(points_norm), 3*tools.eval_tools.sigGzi(points_norm),100)
        norm_y = scipy.stats.norm.pdf(norm_x, 0, 1) * points_norm.shape[0] * np.diff(bins)[0]
        ax[i].plot (norm_x, norm_y, color="red", label="N(0,1)")
        ax[i].legend()
    return ax

def plot_magnitudes_error (sims):
    fig, ax = plt.subplots(5, 5, figsize=(10,10), sharex="col", sharey="row")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    magnitudes = tools.create_simulated_data.magnitudes_from_colors(np.array(sims[0][0]))
    errors = tools.create_simulated_data.magnitude_from_colors_errror(np.array(sims[0][1]))
    mags = {"r" : magnitudes[:, 0], "u" : magnitudes[:, 1], "g" : magnitudes[:, 2], "i" : magnitudes[:, 3], "z" : magnitudes[:, 4]}
    errs = {"r" : errors[:, 0], "u" : errors[:, 1], "g" : errors[:, 2], "i" : errors[:, 3], "z" : errors[:, 4]}
    mags_mixed = {"r" : sims[0][0][:, 0], "ug" : sims[0][0][:, 1], "gr" : sims[0][0][:, 2], "ri" : sims[0][0][:, 3], "iz" : sims[0][0][:, 4]}
    errs_mixed = {"r" : sims[0][1][:, 0], "ug" : sims[0][1][:, 1], "gr" : sims[0][1][:, 2], "ri" : sims[0][1][:, 3], "iz" : sims[0][1][:, 4]}
    for j, name_mixed in enumerate(mags_mixed.keys()):
        for i, name_single in enumerate(mags.keys()):
            if i == len(mags.keys())-1:
                ax[j,i].set_ylabel(name_mixed+" error")
                ax[j,i].yaxis.set_label_position('right')
            if j == 0:
                ax[j,i].set_xlabel(name_single+" magnitude")
                ax[j,i].xaxis.set_label_position('top')
            #ax[i, j] = tools.eval_tools.plot_2Dmap(mags[name_single], errs_mixed[name_mixed], ax[i, j], xname=name_single+" magnitude", yname=name_mixed+" error", subsample=100)
            ax[j,i].scatter(mags[name_single], errs_mixed[name_mixed], s=0.1)
    return fig

def plot_sigma_dependency(p, sigma_p, y_test):
    num_bins = 100
    fig, ax = plt.subplots(2, 3, figsize=(3 * 5, 2 * 3))
    outputs = ["Mr", "Ar", "FeH"]
    for i in range(len(p)):
        bins = np.linspace(y_test[i].min(), y_test[i].max(), num_bins)
        coverage = np.array([(np.abs(y_test[i][(y_test[i] >= bins[j]) & (y_test[i] <= bins[j + 1])] - p[i][
            (y_test[i] >= bins[j]) & (y_test[i] <= bins[j + 1])]) < sigma_p[i][
                                  (y_test[i] >= bins[j]) & (y_test[i] <= bins[j + 1])]).sum() /
                             y_test[i][(y_test[i] >= bins[j]) & (y_test[i] <= bins[j + 1])].shape[0] for j in
                             range(bins.shape[0] - 1)])
        bins = bins[:-1][~np.isnan(coverage)]
        coverage = coverage[~np.isnan(coverage)]
        ax[0][i].bar(bins + (bins[1] - bins[0]), coverage, width=bins[1] - bins[0])
        ax[0][i].set_xlabel(outputs[i])
        ax[0][i].set_ylabel("Coverage")
        ax[1][i].hist(y_test[i], bins=num_bins)
        ax[1][i].set_xlabel(outputs[i])
        ax[1][i].set_ylabel("Histogram")
    plt.tight_layout()
    return fig


def plotModel(model, inputs=[], outputs=[]):
    def plot_one_layer(layer, graph):
        sha = 'box'
        color = "black"
        layer_class = layer.__class__.__name__
        if type(layer.input) == list or type(layer.input) == tuple:
            input_shape = [list(i.shape[1:]) for i in layer.input]
        else:
            input_shape = list(layer.input.shape[1:])
        if type(layer.output) == list or type(layer.output) == tuple:
            output_shape = [list(i.shape[1:]) for i in layer.output]
        else:
            output_shape = list(layer.output.shape[1:])
        label = layer_class + "\nInput: " + str(input_shape) + "\nOutput: " + str(output_shape)
        if "activation" in layer.get_config().keys():
            label = label + "\nActivation: " + str(layer.get_config()["activation"])
        # add node
        node = pydot.Node(layer.name, color=color, shape=sha, label=label)
        graph.add_node(node)
        # add edges
        for inbound_nodes in layer.inbound_nodes:
            for inbound_layer, node_index, tensor_index, _ in inbound_nodes.iterate_inbound():
                if not ("InputLayer" in inbound_layer.__class__.__name__):
                    edge = pydot.Edge(inbound_layer.name, node, color='blue')
                    graph.add_edge(edge)
                else:
                    if not (layer.name in graph.inputs):
                        graph.inputs.append(layer.name)
                if layer.outbound_nodes == []:
                    if not (layer.name in graph.outputs):
                        graph.outputs.append(layer.name)
        return graph

    def plot_one_branch(submodel, graph):
        layers = submodel.layers
        if "Functional" in [layer.__class__.__name__ for layer in layers]:
            mask = np.array("Functional" == np.array([layer.__class__.__name__ for layer in layers]))
            for functional in np.array(layers)[mask]:
                for funct_layer in functional.layers:
                    funct_layer._name = functional.name + "_" + funct_layer.name
                graph = plot_one_branch(functional, graph)
                layers = [i for i in layers if i.__class__.__name__ != "Functional"]
        for i in range(len(layers)):
            if not ("InputLayer" in layers[i].__class__.__name__):
                graph = plot_one_layer(layers[i], graph)
        return graph

    def add_inputs(graph, inputs):
        sha = "plaintext"
        color = "black"
        for i in inputs:
            label = i
            node = pydot.Node(i, color=color, shape=sha, label=label)
            graph.add_node(node)
            for k in graph.inputs:
                edge = pydot.Edge(i, k, color='blue')
                graph.add_edge(edge)
        return graph

    def add_outputs(graph, outputs):
        sha = "plaintext"
        color = "black"
        for ii, i in enumerate(np.array(outputs).flatten()):
            label = i
            node = pydot.Node(i, color=color, shape=sha, label=label)
            graph.add_node(node)
            if len(graph.outputs) == len(outputs):
                edge = pydot.Edge(graph.outputs[ii], i, color='blue')
                graph.add_edge(edge)
            else:
                for k in graph.outputs:
                    edge = pydot.Edge(k, i, color='blue')
                    graph.add_edge(edge)
        return graph

    model = tf.keras.models.clone_model(model)
    graph = pydot.Dot('model', graph_type='digraph', bgcolor="white", rankdir="TB")
    graph.set_node_defaults(fontname='Courier', fontsize='10')
    graph.inputs = []
    graph.outputs = []
    graph = plot_one_branch(model, graph)
    graph = add_inputs(graph, inputs)
    graph = add_outputs(graph, outputs)
    return graph

def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)