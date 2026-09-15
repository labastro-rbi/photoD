"""
How the network improves with the size of the training set.

For training sets of 100 to 30000 stars, trains the network on the Bayesian estimates and compares its
predictions for the remaining stars with the Bayesian estimates and with the true values. Metrics are written
to ../outputs/metrics_bayes1.txt and ../outputs/metrics_truth1.txt (plot them with training_size_plots.py).
"""
import sys

import numpy as np
import tensorflow as tf

sys.path.append("../Environment")
import photod


def find_minimum_size(input_training_path, output_file_truth="../outputs/metrics_truth1.txt",
                      output_file_bayes="../outputs/metrics_bayes1.txt", epochs=1024, iterations=2):
    header = ("reduce_size, time, "
              "loss, bayes_loss, "
              "chi_squaredMr, chi_squaredAr, chi_squaredFeH, "
              "bayes_chi_squaredMr, bayes_chi_squaredAr, bayes_chi_squaredFeH, "
              "mseMr, mseAr, mseFeH, "
              "bayes_mseMr, bayes_mseAr, bayes_mseFeH\n")
    for path in (output_file_truth, output_file_bayes):
        with open(path, "w") as f:
            f.write(header)

    # Read the data to the astropy table
    sims = photod.model_tools.table_from_file(input_training_path)

    # Get the input arrays
    x, x_error, y = photod.model_tools.tensor_slices_from_table(sims)
    y_error = photod.model_tools.outputs_error_from_table(sims)

    # True values are only in simulated catalogs (columns without "Est" in the name)
    if len([col for col in sims.colnames if "true" in str(col).lower()]) != 0:
        true_parameters_column_names = [col for col in sims.colnames if "est" not in str(col).lower()]
        _, _, true = photod.model_tools.tensor_slices_from_table(sims[true_parameters_column_names])
    else:
        true = None

    # Create array with train sizes
    reduce_sizes = np.arange(100, 30000, 100)

    # Split train and test set
    train_index = np.random.choice(np.arange(0, x.shape[0]), reduce_sizes.max(), replace=False)
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
        true_test = y_test

    # Evaluate bayes model
    photod_model = photod.PhotoD(batch_size=10, fit_in_memory=True)

    bayes_true_loss = 0
    with tf.device('/CPU:0'):
        for k in range(len(photod_model.loss_weights)):
            bayes_true_loss += photod_model.loss_weights[k] * photod_model.loss_fn(true_test[:, k][:, np.newaxis],
                                                                                   np.stack([y_test[:, k],
                                                                                             y_error_test[:, k]],
                                                                                            axis=1)).numpy().mean()
    bayes_true_cs = [photod.model_tools.chi_squared(true_test[:, i:i + 1],
                                                    np.stack([y_test[:, i], y_error_test[:, i]], axis=1)).numpy()
                     for i in range(true_test.shape[1])]
    bayes_true_mse = [photod.model_tools.mse(true_test[:, i:i + 1],
                                             np.stack([y_test[:, i], y_error_test[:, i]], axis=1)).numpy().mean()
                      for i in range(y.shape[1])]

    for size in reduce_sizes:
        # Create a photometric distance model with the architecture of the trained model
        photod_model = photod.PhotoD(batch_size=min(1024, int(size)), fit_in_memory=True)
        photod_model.load_model("../PhotoD.keras")
        photod_model.create_model(photod_model.architecture)

        # Import input arrays and train
        photod_model.import_tensors(x_train[:size], x_error_train[:size], y_train[:size], y_error_train[:size])
        training_time = photod_model.train_model(epochs=epochs, iterations=iterations, decay_epochs=20,
                                                 decay_rate=0.7)

        p, p_error, _ = photod_model.predict((x_test, x_error_test))

        # Calculate metrics
        cs_truth = [photod.model_tools.chi_squared(true_test[:, i:i + 1],
                                                   np.stack([p[:, i], p_error[:, i]], axis=1)).numpy()
                    for i in range(true_test.shape[1])]
        cs_bayes = [photod.model_tools.chi_squared(y_test[:, i:i + 1],
                                                   np.stack([p[:, i], p_error[:, i]], axis=1)).numpy()
                    for i in range(y_test.shape[1])]
        mse_truth = [photod.model_tools.mse(true_test[:, i:i + 1],
                                            np.stack([p[:, i], p_error[:, i]], axis=1)).numpy().mean()
                     for i in range(y.shape[1])]
        mse_bayes = [photod.model_tools.mse(y_test[:, i:i + 1],
                                            np.stack([p[:, i], p_error[:, i]], axis=1)).numpy().mean()
                     for i in range(y.shape[1])]
        with tf.device('/CPU:0'):
            loss_truth = 0
            loss_bayes = 0
            for k in range(len(photod_model.loss_weights)):
                loss_truth += photod_model.loss_weights[k] * photod_model.loss_fn(true_test[:, k][:, np.newaxis],
                                                                                  np.stack([p[:, k], p_error[:, k]],
                                                                                           axis=1)).numpy().mean()
                loss_bayes += photod_model.loss_weights[k] * photod_model.loss_fn(y_test[:, k][:, np.newaxis],
                                                                                  np.stack([p[:, k], p_error[:, k]],
                                                                                           axis=1)).numpy().mean()

        # Write model metrics to file
        with open(output_file_truth, "a") as f:
            f.write(", ".join(str(v) for v in [size, training_time, loss_truth, bayes_true_loss,
                                                *cs_truth, *bayes_true_cs, *mse_truth, *bayes_true_mse]) + "\n")
        with open(output_file_bayes, "a") as f:
            f.write(", ".join(str(v) for v in [size, training_time, loss_bayes, bayes_true_loss,
                                                *cs_bayes, *bayes_true_cs, *mse_bayes, *bayes_true_mse]) + "\n")


if __name__ == "__main__":
    find_minimum_size('../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt', epochs=1024, iterations=2)
