import os

import numpy as np
import sys

sys.path.append("../../Environment")
import photod
import time
import tensorflow as tf

import matplotlib.pyplot as plt


def main(args):
    time_start = time.time()

    # Create output variable to contain the results
    class Results(object):
        def __init__(self):
            self.time = {}
            self.bayes_plots = {}
            self.true_plots = {}
            self.metrics_true = {}
            self.metrics_bayes = {}
            self.training_size = args.train_size
            self.evaluating_size = 0

    result = Results()

    # Read the data to the astropy table
    sims = photod.model_tools.table_from_file(args.input_training_path)
    result.time["data_reading"] = time.time() - time_start
    timer = time.time()

    # Get the input arrays
    x, x_error, y = photod.model_tools.tensor_slices_from_table(sims)
    y_error = photod.model_tools.outputs_error_from_table(sims)

    # Check if the simulation has columns without est in the name (to understand if the true values are present)
    if len([col for col in sims.colnames if "true" in str(col).lower()]) != 0:
        true_parameters_column_names = [col for col in sims.colnames if "est" not in str(col).lower()]
        _, _, true = photod.model_tools.tensor_slices_from_table(sims[true_parameters_column_names])
    else:
        true = None

    # Split the data into training and testing
    if args.train_size > x.shape[0]:
        raise ValueError("The training size is larger than the dataset size")
    if args.train_size > 0:
        train_index = np.random.choice(np.arange(0, x.shape[0]), args.train_size, replace=False)
        train_mask = np.zeros(x.shape[0], dtype=bool)
        train_mask[train_index] = True
        x_train = x[train_mask]
        x_error_train = x_error[train_mask]
        y_train = y[train_mask]
        y_error_train = y_error[train_mask]

        x_test = x[~train_mask]
        x_error_test = x_error[~train_mask]
        y_test = y[~train_mask]
        y_error_test = y_error[~train_mask]
        if true is not None:
            true_test = true[~train_mask]
        else:
            true_test = None
    else:
        train_index = np.arange(0, x.shape[0])
        train_mask = np.ones(x.shape[0], dtype=bool)
        x_train = x
        x_error_train = x_error
        y_train = y
        y_error_train = y_error
        x_test = None
        x_error_test = None
        y_test = None
        true_test = None
        y_error_test = None

    result.time["data_slicing"] = time.time() - timer
    timer = time.time()

    # Create a photometric distance model
    photod_model = photod.PhotoD(batch_size=args.batch_size, fit_in_memory=True)

    # Import a new model
    photod_model.create_model(photod_model.architecture)

    # Import input arrays
    result.training_size = x_train.shape[0]
    photod_model.import_tensors(x=x_train, x_error=x_error_train, y=y_train, y_error=y_error_train)
    result.time["prepare_model"] = time.time() - timer

    # Train a model
    result.time["training_photoD"] = photod_model.train_model(epochs=args.epochs,
                                                       iterations=args.iterations,
                                                       decay_epochs=args.decay_epochs,
                                                       decay_rate=args.decay_rate)
    result.time["training_error"] = photod_model.train_error_model(epochs=256,
                                                                   iterations=1,
                                                              decay_epochs=args.decay_epochs,
                                                              decay_rate=args.decay_rate)

    if args.input_prediction_path != "":
        sims_predict = photod.model_tools.table_from_file(args.input_prediction_path)
        if args.input_prediction_path == args.input_training_path:
            sims_predict = sims_predict[~train_mask]
        x_predict, x_error_predict = photod.model_tools.inputs_from_table(sims_predict)
        p_predict, sigma_p_predict, p_bayes_sigma_predict = photod_model.predict((x_predict, x_error_predict))
        sigma_true = np.sqrt(np.square(sigma_p_predict) + np.square(p_bayes_sigma_predict))

        col_names = ['MrNN', 'ArNN', 'FeHNN']
        for i, name in enumerate(col_names):
            sims_predict[name] = p_predict[:, i]
            sims_predict[name+"Unc"] = sigma_true[:, i]
            sims_predict[name+"UncPart"] = sigma_p_predict[:, i]
        photod.model_tools.table_to_file(sims_predict, args.output_prediction_path)

    # Evaluate the model
    if x_test is not None:
        timer = time.time()
        p, sigma_p, p_bayes_sigma = photod_model.predict((x_test, x_error_test))
        result.evaluating_size = x_test.shape[0]
        result.time["evaluation"] = time.time() - timer
        result.time["total"] = time.time() - time_start
    # Plot the results
        if args.do_metrics:
            # Calculate metrics
            with tf.device('/CPU:0'):
                loss_truth = 0
                loss_bayes = 0
                for k in range(len(photod_model.loss_weights)):
                    if true is not None:
                        loss_truth += photod_model.loss_weights[k] * photod_model.loss_fn(true_test[:, k][:, np.newaxis],
                                                                                          np.stack([p[:, k],
                                                                                                    sigma_p[:, k]],
                                                                                                   axis=1)).numpy().mean()
                    loss_bayes += photod_model.loss_weights[k] * photod_model.loss_fn(y_test[:, k][:, np.newaxis],
                                                                                      np.stack([p[:, k],
                                                                                                sigma_p[:, k]],
                                                                                               axis=1)).numpy().mean()
                mse_bayes = [
                    photod.model_tools.mse(y_test[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy().mean()
                    for i
                    in
                    range(y.shape[1])]
                cs_bayes = [
                    photod.model_tools.chi_squared(y_test[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy()
                    for
                    i in
                    range(y_test.shape[1])]
                result.metrics_bayes["loss"] = loss_bayes
                result.metrics_bayes["mse"] = mse_bayes
                result.metrics_bayes["chi2"] = cs_bayes
                print("NN vs Bayes:", result.metrics_bayes)
                if true is not None:
                    cs_truth = [
                        photod.model_tools.chi_squared(true_test[:, i:i + 1],
                                                       np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy()
                        for
                        i in
                        range(true_test.shape[1])]
                    mse_truth = [
                        photod.model_tools.mse(true_test[:, i:i + 1],
                                               np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy().mean()
                        for
                        i in
                        range(y.shape[1])]
                    result.metrics_true["loss"] = loss_truth
                    result.metrics_true["mse"] = mse_truth
                    result.metrics_true["chi2"] = cs_truth
                    print("NN vs True:", result.metrics_true)
        if args.do_plots:
            if args.plot_path == "":
                bayes_plot_path = None
                true_plot_path = None
                show_plot = True
            else:
                plot_path = args.plot_path
                if plot_path[-1] != "/":
                    plot_path += "/"
                bayes_plot_path = plot_path + "bayes/"
                true_plot_path = plot_path + "true/"
                os.makedirs(bayes_plot_path, exist_ok=True)
                os.makedirs(true_plot_path, exist_ok=True)
                show_plot = False
            (fig_correlation, fig_correlation_separate, figar,
             figmr, figfeh, fig_error_2d, fig_chi_squared) = photod.plot_tools.get_model_metrics((x_test, x_error_test),
                                                                                                 y_test, p, sigma_p,
                                                                                                 save_path=bayes_plot_path,
                                                                                                 show_plot=show_plot,
                                                                                                 true_name="bayes", )
            result.bayes_plots = {"fig_correlation": fig_correlation,
                                  "fig_correlation_separate": fig_correlation_separate,
                                  "figar": figar,
                                  "figmr": figmr,
                                  "figfeh": figfeh,
                                  "fig_error_2d": fig_error_2d,
                                  "fig_chi_squared": fig_chi_squared}
            if true is not None:
                sigma_true=np.sqrt(np.square(sigma_p)+np.square(p_bayes_sigma))
                (fig_correlation, fig_correlation_separate, figar,
                 figmr, figfeh, fig_error_2d, fig_chi_squared) = photod.plot_tools.get_model_metrics((x_test, x_error_test),
                                                                                                     true_test, p, sigma_true,
                                                                                                     save_path=true_plot_path,
                                                                                                     show_plot=show_plot,
                                                                                                     true_name="true", )
                result.true_plots = {"fig_correlation": fig_correlation,
                                     "fig_correlation_separate": fig_correlation_separate,
                                     "figar": figar,
                                     "figmr": figmr,
                                     "figfeh": figfeh,
                                     "fig_error_2d": fig_error_2d,
                                     "fig_chi_squared": fig_chi_squared}
    print(result.time["total"])
    return result


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Train a photometric distance model")
    parser.add_argument("--input_training_path", type=str,
                        help="Path to the file with the input data for training",
                        default="../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt")
                        #default="../../data/BayesMethod3D_SEGUEpatch-l110-KarloTest-short1.txt")
    parser.add_argument("--input_prediction_path", type=str,
                        help="Path to the file with the input data for prediction",
                        default="../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt")
                        #default="../../data/BayesMethod3D_SEGUEpatch-l110-KarloTest-short1.txt")
    parser.add_argument("--model_path", type=str,
                        help="Path to the file with the trained model",
                        default="../../PhotoD.keras")
    parser.add_argument("--output_prediction_path", type=str,
                        help="Path to the file to output for prediction",
                        #default="../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest_NN_noerror.txt")
                        default="../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest_NN.txt")
                        #default="../../data/BayesMethod3D_SEGUEpatch-l110-KarloTest-short1_NN.txt")
    parser.add_argument("--train_size", type=int, help="Size of the training set", default=10000)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1024)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=1024)
    parser.add_argument("--iterations", type=int, help="Number of iterations", default=2)
    parser.add_argument("--decay_epochs", type=int, help="Decay epochs", default=10)
    parser.add_argument("--decay_rate", type=float, help="Decay rate", default=0.7)
    parser.add_argument("--output_file_path", type=str, help="Path to the file with the output data",
                        default="")
    parser.add_argument("--do_plots", type=bool, help="Create the plots", default=True)
    parser.add_argument("--do_metrics", type=bool, help="Create the metrics", default=True)
    parser.add_argument("--plot_path", type=str, help="Path to the folder with the plots", default="../../outputs/simulation/")
    return parser


if __name__ == "__main__":
    main(arg_parser().parse_args())
