"""
create_model.py - Create and train PhotoD model.

"""
import time
import json
import numpy as np
import tensorflow as tf
from photod import model_tools as model_tools


def import_tensors(x, x_error, y, reduce=None, test_split=-1, fit_in_memory=True, seed=None):
    """
    Create a dataset from numpy arrays.

    This function creates a dataset from numpy arrays representing input data, input errors, and output parameters.
    The inputs represent color values, the input errors represent the uncertainties for each color, and the output
    parameters represent the desired target values.

    Parameters
    ----------
    x: numpy.ndarray
        Array containing input colors (rmag, ug, gr, ri, iz) with shape (n, 5), where n is the number of objects.
    x_error: numpy.ndarray
        Array containing errors for the input colors with shape (n, 5), representing uncertainties for each input color.
    y: numpy.ndarray
        Array containing output parameters (Mr, Ar, FeH) with shape (n, 3), where n is the number of objects.
    reduce: float or None, optional
        If provided, reduces the training dataset to a fraction of the original dataset size. If None, the dataset is
        not reduced. Default is None.
    test_split: float, optional
        Fraction of the dataset to be used for testing. A value between 0 and 1 specifies the proportion of data for the test set.
        If test_split is -1, no test set is created. Default is -1.
    fit_in_memory: bool, optional
        If True, the dataset is fully loaded into memory for training. If False, the dataset is streamed for larger
        datasets that don't fit into memory. Default is True.
    seed: int or None, optional
        Seed for the shuffles of the split and the reduction. Default is None.

    Returns
    -------
    train_set: tf.data.Dataset
        The training dataset.
    test_set: tf.data.Dataset
        The test dataset, if test_split > 0. Otherwise, returns the same dataset as train_set.
    """
    dataset = model_tools.dataset_from_tensor_slices(x, x_error, y)
    if 0 < test_split < 1:
        if fit_in_memory:
            train_set, test_set = model_tools.dataset_split_train_test(dataset,
                                                                       tf.data.experimental.cardinality(
                                                                           dataset).numpy(),
                                                                       test_split=test_split, seed=seed)
        else:
            train_set, test_set = model_tools.dataset_split_train_test(dataset, 10,
                                                                       test_split=test_split, seed=seed)
    else:
        train_set = dataset
        test_set = dataset

    if (reduce is not None) and (reduce != 1):
        if reduce > 1 or reduce < 0:
            reduce = reduce / tf.data.experimental.cardinality(train_set).numpy()
        if fit_in_memory:
            train_set = model_tools.dataset_reduce(train_set,
                                                   tf.data.experimental.cardinality(train_set).numpy(),
                                                   reduce, seed=seed)
        else:
            train_set = model_tools.dataset_reduce(train_set, 10, reduce, seed=seed)
    return train_set, test_set


def create_model(architecture, model_name="photoD_model", input_colors=5, output_neurons=2):
    """
    Build a custom PhotoD model based on a provided architecture.

    This function constructs a PhotoD model using a given architecture specification. The architecture defines the number
    of units in each layer and the activation functions for each layer. The model takes two inputs: input colors and their
    associated errors. Layers are sequentially added based on the architecture, and the model produces multiple outputs.

    Parameters
    ----------
    architecture: dict
        A dictionary specifying the architecture with two keys:
        - "units": a list of integers, where each integer represents the number of units in a corresponding layer.
        - "activations": a list of activation functions (as strings) to be applied in each layer.
        The length of both lists must be the same.
    model_name: str, optional
        A name for the model. It is used for display purposes only. Default is "photoD_model".
    input_colors: int, optional
        The number of input color features. LSST color inputs are typically r, u-g, g-r, r-i, and i-z. Default is 5.
    output_neurons: int, optional
        The number of output neurons per output layer. Default is 2.

    Returns
    -------
    model: tf.keras.Model
        A Keras model instance built according to the specified architecture.
    """

    input_x = tf.keras.layers.Input(shape=input_colors, name='x_input')
    input_xerr = tf.keras.layers.Input(shape=input_colors, name='xerr_input')
    x = tf.keras.layers.concatenate([input_x, input_xerr], name="concat_inputs")
    # x = tf.keras.layers.BatchNormalization(name="Normalization", center=True, scale=True)(x)
    assert len(architecture["units"]) == len(architecture["activations"])
    for i in range(len(architecture["units"]) - 1):
        x = tf.keras.layers.Dense(units=architecture["units"][i], activation=architecture["activations"][i],
                                  name="dense" + str(i))(x)
        x = tf.keras.layers.BatchNormalization(name="Normalization" + str(i))(x)
    output = []
    for i in range(architecture["units"][-1]):
        output.append(tf.keras.layers.Dense(units=output_neurons,
                                            activation=architecture["activations"][-1],
                                            name="output" + str(i))(x))
    model = tf.keras.Model(inputs=(input_x, input_xerr), outputs=output, name=model_name)
    return model


class PhotoD:
    """
    Neural network estimates of Mr, Ar and [Fe/H] from LSST photometry.

    Holds two networks with the same inputs: the rmag, u-g, g-r, r-i, i-z values and their errors.
    The main network predicts, for each of Mr, Ar and [Fe/H], a value and its uncertainty. It is trained on the
    Bayesian estimates (or true values) of a training catalog with the marginal posterior density loss.
    The error network is trained on the uncertainties of the Bayesian estimates, so that predict() can also
    return the part of the uncertainty that the Bayesian method adds.

    Typical use:
        model = PhotoD()
        model.import_csv("catalog.txt", test_split=0.2)
        model.create_model()
        model.train_model(epochs=1024)
        model.train_error_model(epochs=256)
        p, sigma_p, bayes_sigma = model.predict((x, x_error))

    Parameters
    ----------
    model_type : str
        "single" or "multi". Default is "multi".
    batch_size : int
        Batch size for training and prediction. Default is 2048.
    fit_in_memory : bool
        Keep the whole dataset in memory. Default is True.
    seed : int, optional
        Seed for TensorFlow.
    """

    def __init__(self, model_type="multi", batch_size=2048, fit_in_memory=True, seed=None):
        self.metrics = None
        self.model_type = model_type
        self.batch_size = batch_size
        self.architecture = None
        self.model = None
        self.train_set = None
        self.test_set = None
        self.fit_in_memory = fit_in_memory
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            self.strategy = tf.distribute.MirroredStrategy(['/gpu:' + str(i) for i in range(len(gpu_devices))],
                                                           cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        else:
            self.strategy = tf.distribute.get_strategy()
        if seed is not None:
            tf.random.set_seed(seed)
        self.default_arhitecture = {'units': [45, 45, 42, 33, 36, 3],
                                    'activations': ['gelu', 'softsign', 'softsign', 'sigmoid', 'gelu', 'linear']}
        self.loss_weights = [1 / (0.1 ** 2), 1 / (0.02 ** 2), 1 / (0.1 ** 2)]
        self.loss_fn = model_tools.marginal_posterior_density_loss(1.0)
        # Error model
        self.error_architecture = None
        self.error_model = None
        self.error_train_set = None
        self.error_test_set = None
        self.error_loss_fn = tf.keras.losses.MeanSquaredError()

    def import_csv(self, path, reduce=None, test_split=-1):
        """
        Imports a dataset from a CSV file and creates training and test sets.

        Parameters
        ----------
        path : str
            Path to the CSV file containing the dataset.
        reduce : float, optional
            The fraction to reduce the dataset by, for faster training. Default is None.
        test_split : float, optional
            Fraction of the dataset to use for testing. Default is -1 (no test set split).

        Returns
        -------
        None
        """
        sims = model_tools.table_from_file(path)
        x, x_error, y = model_tools.tensor_slices_from_table(sims)
        y_error = model_tools.outputs_error_from_table(sims)
        self.import_tensors(x, x_error, y, y_error, reduce=reduce, test_split=test_split)

    def import_tensors(self, x, x_error, y, y_error, reduce=None, test_split=-1):
        """
        Imports tensor data directly, creating training and test sets.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        x_error : numpy.ndarray
            Errors for input features.
        y : numpy.ndarray
            Target labels.
        y_error : numpy.ndarray
            Errors for target labels.
        reduce : float, optional
            The fraction to reduce the dataset by. Default is None.
        test_split : float, optional
            Fraction of the dataset to use for testing. Default is -1 (no test set split).

        Returns
        -------
        None
        """
        # the same seed for both, so that the error datasets hold the same stars as the main datasets
        seed = int(np.random.randint(2 ** 31 - 1))
        self.train_set, self.test_set = import_tensors(x, x_error, y, reduce=reduce,
                                                       test_split=test_split,
                                                       fit_in_memory=self.fit_in_memory, seed=seed)
        self.error_train_set, self.error_test_set = import_tensors(x, x_error, y_error,
                                                                   reduce=reduce, test_split=test_split,
                                                                   fit_in_memory=self.fit_in_memory, seed=seed)

    def create_model(self, architecture=None, error_architecture=None, model_name="photoD_model"):
        """
        Creates a new PhotoD model.

        Uses the provided architecture to create a PhotoD model and an error model. If no architecture is provided,
        default architecture is used.

        Parameters
        ----------
        architecture : dict, optional
            Dictionary specifying the number of units and activation functions in each layer.
        error_architecture : dict, optional
            Dictionary specifying the architecture for the error model. Default is the same as the main model.
        model_name : str, optional
            Name of the model for saving/loading. Default is "photoD_model".

        Returns
        -------
        None
        """
        if architecture is None:
            architecture = self.default_arhitecture
        self.architecture = architecture
        self.model = create_model(architecture, model_name=model_name, input_colors=5, output_neurons=2)
        if error_architecture is None:
            error_architecture = self.default_arhitecture
        self.error_architecture = error_architecture
        self.error_model = create_model(error_architecture, model_name=model_name + "_error",
                                        input_colors=5, output_neurons=1)

    def save_model(self, path):
        """
        Saves the PhotoD model to a specified path.

        Parameters
        ----------
        path : str
            File path where the model will be saved. If the file extension is not '.keras', it will be added.

        Returns
        -------
        None
        """
        if self.model is None:
            raise Exception("Model not created. Use create_model() or load_model() method.")
        if path[-6:] != ".keras":
            path += ".keras"
        try:
            self.model.save(path, include_optimizer=False, save_format="keras")
        except ValueError:
            # newer versions of Keras take no options for the .keras format
            self.model.save(path)

    def save_error_model(self, path):
        """
        Saves the error model to a specified path.

        Parameters
        ----------
        path : str
            File path where the error model will be saved. If the file extension is not '.keras', it will be added.

        Returns
        -------
        None
        """
        if self.error_model is None:
            raise Exception("Error model not created. Use create_model() or load_model() method.")
        if path[-6:] != ".keras":
            path += ".keras"
        try:
            self.error_model.save(path, include_optimizer=False, save_format="keras")
        except ValueError:
            # newer versions of Keras take no options for the .keras format
            self.error_model.save(path)

    def load_model(self, path):
        """
        Loads a PhotoD model from the specified file.

        Parameters
        ----------
        path : str
            Path to a saved Keras model.

        Returns
        -------
        None
        """
        self.model = tf.keras.models.load_model(path, compile=False)
        self.architecture = model_tools.extract_arhitecture(self.model)
        self.architecture["units"][-1] = int(self.architecture["units"][-1] / 2)

    def load_error_model(self, path):
        """
        Loads an error model from the specified file.

        Parameters
        ----------
        path : str
            Path to a saved Keras model.

        Returns
        -------
        None
        """
        self.error_model = tf.keras.models.load_model(path, compile=False)
        self.error_architecture = model_tools.extract_arhitecture(self.error_model)

    def train_model(self, epochs, iterations=5, decay_epochs=10, decay_rate=0.9, logs_dir=None):
        """
        Trains the PhotoD model using the provided dataset.

        Implements early stopping, learning rate decay, and termination on NaN. Training is repeated for a specified
        number of iterations, each time reinitializing the model and saving the best version.

        Parameters
        ----------
        epochs : int
            Number of epochs to train in each iteration.
        iterations : int, optional
            Number of times to reinitialize and train the model. Default is 5.
        decay_epochs : int, optional
            Number of epochs before reducing the learning rate. Default is 10.
        decay_rate : float, optional
            Factor to reduce the learning rate by. Default is 0.9.
        logs_dir : str, optional
            Directory to store training logs for TensorBoard.

        Returns
        -------
        time : float
            The time taken for training in seconds.
        """
        if self.model is None:
            raise Exception("Model not created. Use create_model() or load_model() method.")
        if self.train_set is None:
            raise Exception("Train set not created. Use import_dataset() method.")
        dataset = self.train_set
        batch_size = self.batch_size
        min_loss = 100000000
        start_time = time.time()
        if self.fit_in_memory:
            train, val = model_tools.dataset_split_train_val(dataset, tf.data.experimental.cardinality(dataset).numpy(),
                                                             val_split=0.2)
        else:
            train, val = model_tools.dataset_split_train_val(dataset, tf.data.experimental.cardinality(dataset).numpy(),
                                                             val_split=0.2)
        train = model_tools.dataset_batch(train, batch_size, self.strategy)
        val = model_tools.dataset_batch(val, batch_size, self.strategy)
        if self.fit_in_memory:
            train = train.cache()
            val = val.cache()
        for i in range(iterations):
            if iterations > 1:
                print("#####################Iteration: ", i + 1, "/", iterations, "#####################")
            max_lr = 0.01
            with self.strategy.scope():
                model = tf.keras.models.clone_model(self.model)
                model.compile(optimizer=tf.keras.optimizers.Adam(max_lr, clipvalue=10.0),
                              loss=self.loss_fn,
                              # metrics=[model_tools.coverage, model_tools.mse, model_tools.kl_divergence],
                              loss_weights=self.loss_weights)
            earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10 * decay_epochs,
                                                                verbose=0,
                                                                restore_best_weights=True, min_delta=1e-4)
            terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
            reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=decay_rate,
                                                                        patience=decay_epochs, verbose=0)
            kb = [earlystopping_kb, terminateonnan_kb, reducelronplateau_kb]
            if logs_dir is not None:
                kb.append(tf.keras.callbacks.TensorBoard(logs_dir,
                                                         histogram_freq=1,
                                                         profile_batch='10,20'))
            model.fit(train, epochs=epochs,
                      validation_data=val,
                      callbacks=kb, verbose=2)
            val_loss = model.evaluate(val, verbose=0)[0]
            if val_loss < min_loss:
                min_loss = val_loss
                self.model = model
        return time.time() - start_time

    def train_error_model(self, epochs, iterations=5, decay_epochs=10, decay_rate=0.9, logs_dir=None):
        """
        Trains the error model using the provided dataset.

        Similar to `train_model`, it supports early stopping and learning rate decay. Trains for a specified number of
        iterations.

        Parameters
        ----------
        epochs : int
            Number of epochs to train in each iteration.
        iterations : int, optional
            Number of times to reinitialize and train the error model. Default is 5.
        decay_epochs : int, optional
            Number of epochs before reducing the learning rate. Default is 10.
        decay_rate : float, optional
            Factor to reduce the learning rate by. Default is 0.9.
        logs_dir : str, optional
            Directory to store training logs for TensorBoard.

        Returns
        -------
        time : float
            The time taken for training in seconds.
        """

        if self.error_model is None:
            raise Exception("Error model not created. Use create_model() or load_model() method.")
        if self.error_train_set is None:
            raise Exception("Train set for the error not created. Use import_dataset() method.")
        dataset = self.error_train_set
        batch_size = self.batch_size
        min_loss = 100000000
        start_time = time.time()
        if self.fit_in_memory:
            train, val = model_tools.dataset_split_train_val(dataset, tf.data.experimental.cardinality(dataset).numpy(),
                                                             val_split=0.2)
        else:
            train, val = model_tools.dataset_split_train_val(dataset, tf.data.experimental.cardinality(dataset).numpy(),
                                                             val_split=0.2)
        train = model_tools.dataset_batch(train, batch_size, self.strategy)
        val = model_tools.dataset_batch(val, batch_size, self.strategy)
        if self.fit_in_memory:
            train = train.cache()
            val = val.cache()
        for i in range(iterations):
            if iterations > 1:
                print("#####################Iteration: ", i + 1, "/", iterations, "#####################")
            max_lr = 0.01
            with self.strategy.scope():
                model = tf.keras.models.clone_model(self.error_model)
                model.compile(optimizer=tf.keras.optimizers.Adam(max_lr, clipvalue=10.0),
                              loss=self.error_loss_fn,
                              # metrics=[model_tools.coverage, model_tools.mse, model_tools.kl_divergence],
                              loss_weights=self.loss_weights)
            earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10 * decay_epochs,
                                                                verbose=0,
                                                                restore_best_weights=True, min_delta=1e-4)
            terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
            reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=decay_rate,
                                                                        patience=decay_epochs, verbose=0)
            kb = [earlystopping_kb, terminateonnan_kb, reducelronplateau_kb]
            if logs_dir is not None:
                kb.append(tf.keras.callbacks.TensorBoard(logs_dir,
                                                         histogram_freq=1,
                                                         profile_batch='10,20'))
            model.fit(train, epochs=epochs,
                      validation_data=val,
                      callbacks=kb, verbose=2)
            val_loss = model.evaluate(val, verbose=0)[0]
            if val_loss < min_loss:
                min_loss = val_loss
                self.error_model = model
        return time.time() - start_time

    def test_model(self, plot=False):
        """
        Tests the PhotoD model on the test dataset.

        Evaluates the model using various metrics such as Kullback-Leibler divergence, mean squared error, and chi-squared.
        Returns predictions, true values, and their respective errors.

        Parameters
        ----------
        plot : bool, optional
            If True, generates plots of the results. Default is False.

        Returns
        -------
        x : numpy.ndarray
            Input data.
        y : numpy.ndarray
            True output values.
        y_error : numpy.ndarray
            Errors in the true output values.
        p : numpy.ndarray
            Predicted output values.
        sigma_p : numpy.ndarray
            Errors in the predicted output values.
        bayes_sigma : numpy.ndarray
            Bayesian uncertainty in the predictions.
        """

        if self.model is None:
            raise Exception("Model not created. Use create_model() or load_model() method.")
        if self.test_set is None:
            raise Exception("Test set not created. Use import_dataset() method.")
        dataset = self.test_set
        error_dataset = self.error_test_set
        x, y = model_tools.dataset_split_x_y(dataset)
        _, y_error = model_tools.dataset_split_x_y(error_dataset)
        p, sigma_p, bayes_sigma = self.predict(x)
        y = np.stack([np.array(y[i]).flatten() for i in
                      range(len(y))], axis=1)
        y_error = np.stack([np.array(y_error[i]).flatten() for i in
                            range(len(y_error))], axis=1)
        kl_div = [model_tools.kl_divergence(y[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy() for i in
                  range(y.shape[1])]
        mse = [model_tools.mse(y[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy().mean() for i in
               range(y.shape[1])]
        coverage = [model_tools.coverage(y[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy() for i in
                    range(y.shape[1])]
        cs = [model_tools.chi_squared(y[:, i:i + 1], np.stack([p[:, i], sigma_p[:, i]], axis=1)).numpy() for i in
              range(y.shape[1])]
        self.metrics = {"kl_div": kl_div, "mse": mse, "coverage": coverage, "chi_squared": cs}
        return x, y, y_error, p, sigma_p, bayes_sigma

    def predict(self, x):
        """
        Predict output parameters.

        Predict output parameters from input parameters. If model is not created, raise Exception. Input should be a
        tuple of two numpy arrays. First array is a numpy array of input parameters. Second array is a numpy array of
        input parameters errors. Both arrays should have shape (n, 5), where n is a number of objects. Output is a
        tuple of two numpy arrays. First array is a numpy array of output parameters. Second array is a numpy array of
        output parameters errors. Both arrays have shape (n, 3), where n is a number of objects.

        Parameters
        ----------
        x : tuple
            Tuple of two numpy arrays. First array is a numpy array of input parameters. Second array is a numpy array
            of input parameters errors. Both arrays should have shape (n, 5), where n is a number of objects.

        Returns
        -------
        p : numpy.ndarray
            Predicted output parameters.
        sigma_p : numpy.ndarray
            Errors in the predicted output parameters.
        bayes_sigma_p : numpy.ndarray
            Predicted Bayesian uncertainty.
        """
        if self.model is None:
            raise Exception("Model not created. Use create_model() or load_model() method.")
        batch_size = self.batch_size
        prediction = self.model.predict(x, batch_size=batch_size, verbose=0)
        error_prediction = self.error_model.predict(x, batch_size=batch_size, verbose=0)
        bayes_sigma_p = np.stack([np.array(error_prediction[i]).reshape((-1, 1))[:, 0] for i in
                                  range(len(error_prediction))], axis=1)
        p = np.stack([np.array(prediction[i]).reshape((-1, 2))[:, 0] for i in
                      range(len(prediction))], axis=1)  # cleaning shapes of the outputs
        sigma_p = np.stack([np.abs(np.array(prediction[i])).reshape((-1, 2))[:, -1] for i in
                            range(len(prediction))], axis=1)  # cleaning shapes of the sigma outputs
        return p, sigma_p, bayes_sigma_p

    def save_arhitecture(self, path):
        """
        Saves the architecture of the PhotoD model to a file.

        Parameters
        ----------
        path : str
            File path to save the architecture dictionary. If not ending in '.json', the extension will be added.

        Returns
        -------
        None
        """
        if self.architecture is None:
            raise Exception("Architecture not created. Use create_model() method.")
        if path[-5:] != ".json":
            path += ".json"
        with open(path, 'w') as f:
            json.dump(self.architecture, f)


if __name__ == "__main__":
    pass
