"""Coverage and QA plots for catalogs written by photod.pipeline (with the MrNN, ArNN, FeHNN columns)."""
import os
import numpy as np
import sys

sys.path.append("../Environment")
import photod.model_tools as model_tools
import photod.plot_tools as plot_tools


def read_and_plot(file_path, plot_path=""):
    # Read the data from the file
    sims_predict = model_tools.table_from_file(file_path)
    x, x_error = model_tools.inputs_from_table(sims_predict)
    y_bayes = model_tools.outputs_from_table(sims_predict)
    sigma_bayes = np.stack([sims_predict['MrUnc'], sims_predict['ArUnc'], sims_predict['FeHUnc']], axis=1)
    y_NN = np.stack([sims_predict['MrNN'], sims_predict['ArNN'], sims_predict['FeHNN']], axis=1)
    sigma_NN = np.stack([sims_predict['MrNNUnc'], sims_predict['ArNNUnc'], sims_predict['FeHNNUnc']], axis=1)
    sigma_NN_partial = np.stack(
        [sims_predict['MrNNUncPart'], sims_predict['ArNNUncPart'], sims_predict['FeHNNUncPart']], axis=1)
    if ("MrTrue" in sims_predict.colnames) and ("ArTrue" in sims_predict.colnames) and (
            "FeHTrue" in sims_predict.colnames):
        y_true = np.stack([sims_predict['MrTrue'], sims_predict['ArTrue'], sims_predict['FeHTrue']], axis=1)
    else:
        y_true = None

    # Compare the coverage
    coverage = [
        model_tools.coverage(y_bayes[:, i:i + 1], np.stack([y_NN[:, i], sigma_NN_partial[:, i]], axis=1)).numpy() for i
        in
        range(y_bayes.shape[1])]
    print("Coverage vs Bayes:", coverage)
    if y_true is not None:
        coverage = [
            model_tools.coverage(y_true[:, i:i + 1], np.stack([y_NN[:, i], sigma_NN[:, i]], axis=1)).numpy()
            for i in
            range(y_true.shape[1])]
        print("Coverage vs True:", coverage)
    # Generate the plots
    if plot_path == "":
        bayes_plot_path = None
        true_plot_path = None
    else:
        if plot_path[-1] != "/":
            plot_path += "/"
        bayes_plot_path = plot_path + "bayes/"
        true_plot_path = plot_path + "true/"
        os.makedirs(bayes_plot_path, exist_ok=True)

    (fig_correlation, fig_correlation_separate, figar,
     figmr, figfeh, fig_error_2d, fig_chi_squared) = plot_tools.get_model_metrics((x, x_error),
                                                                                  y_bayes, y_NN, sigma_NN_partial,
                                                                                  save_path=bayes_plot_path,
                                                                                  show_plot=True,
                                                                                  true_name="bayes")
    results_bayes = {
        "fig_correlation": fig_correlation,
        "fig_correlation_separate": fig_correlation_separate,
        "figar": figar,
        "figmr": figmr,
        "figfeh": figfeh,
        "fig_error_2d": fig_error_2d,
        "fig_chi_squared": fig_chi_squared
    }
    if y_true is not None:
        (fig_correlation, fig_correlation_separate, figar,
         figmr, figfeh, fig_error_2d, fig_chi_squared) = plot_tools.get_model_metrics((x, x_error),
                                                                                      y_true, y_NN, sigma_NN,
                                                                                      save_path=true_plot_path,
                                                                                      show_plot=True,
                                                                                      true_name="true")
        results_true = {
            "fig_correlation": fig_correlation,
            "fig_correlation_separate": fig_correlation_separate,
            "figar": figar,
            "figmr": figmr,
            "figfeh": figfeh,
            "fig_error_2d": fig_error_2d,
            "fig_chi_squared": fig_chi_squared
        }
    else:
        results_true = None
    return results_bayes, results_true


# Read and plot the data
print("simulation")
read_and_plot("../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest_NN.txt", plot_path="../outputs/simulation/")
print("real")
read_and_plot("../data/BayesMethod3D_SEGUEpatch-l110-KarloTest-short1_NN.txt", plot_path="../outputs/real/")
