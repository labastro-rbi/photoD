"""Uncertainties of the network compared with the uncertainties of the Bayesian estimates."""
import numpy as np
import sys

sys.path.append("../Environment")
import photod.model_tools as model_tools
import photod.plot_tools as plot_tools
from matplotlib import pyplot as plt

def read_and_plot(file_path, plot_path=""):
    # Read the data from the file
    sims_predict = model_tools.table_from_file(file_path)
    x, x_error = model_tools.inputs_from_table(sims_predict)
    y_bayes = model_tools.outputs_from_table(sims_predict)
    sigma_bayes = np.stack([sims_predict['MrUnc'], sims_predict['ArUnc'], sims_predict['FeHUnc']], axis=1)
    y_NN = np.stack([sims_predict['MrNN'], sims_predict['ArNN'], sims_predict['FeHNN']], axis=1)
    sigma_NN = np.stack([sims_predict['MrNNUnc'], sims_predict['ArNNUnc'], sims_predict['FeHNNUnc']], axis=1)
    sigma_NN_partial = np.stack([sims_predict['MrNNUncPart'], sims_predict['ArNNUncPart'], sims_predict['FeHNNUncPart']], axis=1)
    if ("MrTrue" in sims_predict.colnames) and ("ArTrue" in sims_predict.colnames) and (
            "FeHTrue" in sims_predict.colnames):
        y_true = np.stack([sims_predict['MrTrue'], sims_predict['ArTrue'], sims_predict['FeHTrue']], axis=1)
    else:
        y_true = None

    # Generate the plot
    names = [r"$\sigma_{Mr}$", r"$\sigma_{Ar}$", r"$\sigma_{FeH}$"]
    fig_correlation_separate, ax = plt.subplots(len(names), figsize=(5, len(names) * 4))
    fig_correlation_separate.set_facecolor('white')
    for i in range(len(names)):
        name = names[i]
        ax[i] = plot_tools.plot_correlation(sigma_bayes[:, i], sigma_NN_partial[:, i], ax[i], name=name,
                                            small_size=200, true_name="Bayes")
        print (name, "Bayes/NN:", np.mean(sigma_bayes[:, i]/sigma_NN_partial[:, i]))
    plt.show()
    return


# Read and plot the data
read_and_plot("../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest_NN.txt", plot_path="../outputs/simulation/")
