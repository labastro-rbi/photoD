import tensorflow as tf
import numpy as np


def random_arhitecture(seed=None):
    """
    Generate random architecture for neural network.

    Creates random architecture for neural network. The architecture is a dictionary with keys "units" and "activations".
    The values are numpy arrays of integers and strings respectively. The length of the arrays is the number of layers
    in the neural network. The last element of the arrays is the number of output neurons and the output activation
    function. The rest of the elements are the number of neurons and activation functions for each layer. The number of
    layers is random between 1 and 10. The number of neurons is random between 10 and 100. The activation function is
    random from the list of activation functions in tensorflow.keras.activations.

    Parameters
    ----------
    seed: int
        Random seed for numpy.random.seed()

    Returns
    -------
    arh: dict
        Dictionary with keys "units" and "activations" and values numpy arrays of integers and strings respectively.
    """
    np.random.seed(seed)
    activations_choice = ["elu", "gelu", "hard_sigmoid",
                          "linear", "relu", "selu", "sigmoid", "softmax",
                          "softplus", "softsign", "swish", "tanh"]
    layers = np.random.randint(10)
    units = np.random.randint(low=10, high=100, size=layers)
    activations = np.random.choice(a=activations_choice, size=layers)
    units = np.append(units, np.array([3]))
    activations = np.append(activations, np.array(["linear"]))
    arh = {"units": units, "activations": activations}
    return arh


def extract_arhitecture(model):
    """
    Extract architecture from neural network.

    Extracts architecture from neural network. The architecture is a dictionary with keys "units" and "activations". The
    values are numpy arrays of integers and strings respectively. The length of the arrays is the number of layers in
    the neural network. The last element of the arrays is the number of output neurons and the output activation
    function. The rest of the elements are the number of neurons and activation functions for each layer.

    Parameters
    ----------
    model: tf.keras.Model
        Neural network model.

    Returns
    -------
    arh: dict
        Dictionary with keys "units" and "activations" and values numpy arrays of integers and strings respectively.
    """

    arh = {"units": [], "activations": []}
    output_units = 0
    output_activations = ""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            if "output" in layer.name:
                output_units += layer.units
                output_activations = layer.activation.__name__
            else:
                arh["units"].append(layer.units)
                arh["activations"].append(layer.activation.__name__)

    arh["units"].append(output_units)
    arh["activations"].append(output_activations)

    arh["units"] = np.array(arh["units"])
    arh["activations"] = np.array(arh["activations"])
    return arh


def table_from_file(file_path, verbose=True, remove_nan=True):
    """
    Create astropy table from csv file. Requires astropy to be installed.

    Creates astropy table from csv file. The output is an astropy table. The column names are taken from the first line
    of the csv file. The column names are separated by comma or space. The column names are used as the names of the
    columns in the astropy table. The csv has to contain the columns ['rmag', 'ug', 'gr', 'ri', 'iz'] or ['umag', 'gmag',
    'rmag', 'imag', 'zmag'] or ['ug0', 'gr0', 'ri0', 'iz0'] or ['umag', 'gmag', 'rmag', 'imag', 'zmag'] and ['uErr',
    'gErr', 'rErr', 'iErr', 'zErr'] or ['uerr', 'gerr', 'rerr', 'ierr', 'zerr']. Additionally the csv file must contain
    the columns ['MrEst', 'ArEst', 'FeHEst'] or ['MrTrue', 'ArTrue', 'FeHTrue']. The columns ['MrEst', 'ArEst', 'FeHEst']
    are the bayesian estimations of the absolute magnitude, extinction and metallicity. The columns ['MrTrue', 'ArTrue', '
    FeHTrue'] are the true values of the absolute magnitude, extinction and metallicity in the case the simulations are
    used.

    Parameters
    ----------
    file_path: str
        Path to csv file with the data.
    verbose: bool
        If True, print information about the file reading.

    Returns
    -------
    sims: astropy.table.Table
    """
    from astropy.table import Table
    # get column names from the first line of the file
    with open(file_path, "r") as f:
        line = f.readline()
        if "," in line:
            colnames = line.split(",")
        else:
            colnames = line.split()
    # Mr = rmag - Ar - DM - 5
    # rmag0, ug0...iz0: observed values without dust extinction (but include photometric noise, except for rmag0)
    # rmag, ug...iz: dust extinction included
    # uErr...zErr: photometric noise
    if verbose:
        print('READING FROM', file_path)
    sims = Table.read(file_path, format='ascii', names=colnames)
    # Remove NaNs
    if remove_nan:
        data = np.lib.recfunctions.structured_to_unstructured(np.array(sims))
        has_nan = np.any(np.isnan(data), axis=1)
        sims = sims[~has_nan]
    if verbose:
        print(np.size(sims), 'read from', file_path)
    return sims


def table_to_file(sims, path):
    """
    Save an Astropy Table to a file.

    Parameters
    ----------
    sims : `~astropy.table.Table`
        The Astropy Table to be saved.
    path : str
        The file path where the table will be saved. The file will be overwritten if it already exists.

    Returns
    -------
    None
    """

    from astropy.io import ascii
    ascii.write(sims, path, overwrite=True)


def inputs_from_table(sims):
    """
    Compute and add color and magnitude columns to the Astropy Table.

    This function will compute and add the color indices (ug, gr, ri, iz) and their associated errors
    if they are not present in the table. It will also compute the magnitude columns if they are not
    present, based on the color indices. The function will handle different column naming conventions
    for magnitude and error columns.

    Parameters
    ----------
    sims : `~astropy.table.Table`
        The Astropy Table containing photometric data. The table must have some of the following columns:
        - Magnitude columns: 'umag', 'gmag', 'rmag', 'imag', 'zmag'
        - Color indices: 'ug', 'gr', 'ri', 'iz'
        - Error columns: 'uErr', 'gErr', 'rErr', 'iErr', 'zErr' or 'uerr', 'gerr', 'rerr', 'ierr', 'zerr'

    Raises
    ------
    ValueError
        If neither required columns nor expected color/error columns are found in the table.

    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - x: An array of shape (n, 5) with columns for 'rmag', 'ug', 'gr', 'ri', 'iz'.
        - x_error: An array of shape (n, 5) with columns for 'rerr', 'ugerr', 'grerr', 'rierr', 'izerr'.
    """
    if not {"ug", "gr", "ri", "iz"}.issubset(sims.colnames) and {"umag", "gmag", "rmag", "imag", "zmag"}.issubset(
            sims.colnames):
        sims['ug'] = sims['umag'] - sims['gmag']
        sims['gr'] = sims['gmag'] - sims['rmag']
        sims['ri'] = sims['rmag'] - sims['imag']
        sims['iz'] = sims['imag'] - sims['zmag']
    elif {"ug", "gr", "ri", "iz"}.issubset(sims.colnames) and not {"umag", "gmag", "imag", "zmag"}.issubset(
            sims.colnames):
        sims['gmag'] = sims['rmag'] + sims['gr']
        sims['umag'] = sims['gmag'] + sims['ug']
        sims['imag'] = sims['rmag'] - sims['ri']
        sims['zmag'] = sims['imag'] - sims['iz']
    elif {'ug0', 'gr0', 'ri0', 'iz0'}.issubset(sims.colnames) and not {"umag", "gmag", "imag", "zmag"}.issubset(
            sims.colnames):
        sims['gmag'] = sims['rmag'] + sims['gr0']
        sims['umag'] = sims['gmag'] + sims['ug0']
        sims['imag'] = sims['rmag'] - sims['ri0']
        sims['zmag'] = sims['imag'] - sims['iz0']
        sims['ug'] = sims['ug0']
        sims['gr'] = sims['gr0']
        sims['ri'] = sims['ri0']
        sims['iz'] = sims['iz0']
    elif {"ug", "gr", "ri", "iz"}.issubset(sims.colnames) and {"umag", "gmag", "imag", "zmag"}.issubset(sims.colnames):
        pass
    elif {"ug0", "gr0", "ri0", "iz0"}.issubset(sims.colnames) and {"umag", "gmag", "imag", "zmag"}.issubset(
            sims.colnames):
        pass
    else:
        raise ValueError("Neither columns ['ug', 'gr', 'ri', 'iz'] nor ['umag', 'gmag', 'rmag', 'imag', 'zmag'] "
                         "not found in the file")
    # color errors
    if {"uErr", "gErr", "rErr", "iErr", "zErr"}.issubset(sims.colnames):
        sims['ugerr'] = np.sqrt(sims['uErr'] ** 2 + sims['gErr'] ** 2)
        sims['grerr'] = np.sqrt(sims['gErr'] ** 2 + sims['rErr'] ** 2)
        sims['rierr'] = np.sqrt(sims['rErr'] ** 2 + sims['iErr'] ** 2)
        sims['izerr'] = np.sqrt(sims['iErr'] ** 2 + sims['zErr'] ** 2)
        sims['rerr'] = sims['rErr']
    elif {"uerr", "gerr", "rerr", "ierr", "zerr"}.issubset(sims.colnames):
        sims['ugerr'] = np.sqrt(sims['uerr'] ** 2 + sims['gerr'] ** 2)
        sims['grerr'] = np.sqrt(sims['gerr'] ** 2 + sims['rerr'] ** 2)
        sims['rierr'] = np.sqrt(sims['rerr'] ** 2 + sims['ierr'] ** 2)
        sims['izerr'] = np.sqrt(sims['ierr'] ** 2 + sims['zerr'] ** 2)
    else:
        raise ValueError(
            "Neither columns ['uErr', 'gErr', 'rErr', 'iErr', 'zErr'] nor ['uerr', 'gerr', 'rerr', 'ierr', 'zerr'] "
            "not found in the file")
    x = np.array([sims[col].data for col in ['rmag', 'ug', 'gr', 'ri', 'iz']]).T
    x_error = np.array([sims[col].data for col in ['rerr', 'ugerr', 'grerr', 'rierr', 'izerr']]).T
    return x, x_error


def outputs_from_table(sims):
    """
    Extract output data columns from the Astropy Table.

    This function extracts columns related to stellar parameters from the table. It checks for the presence
    of different sets of column names related to stellar parameters and selects the appropriate columns.

    Parameters
    ----------
    sims : `~astropy.table.Table`
        The Astropy Table containing output data. The table must have some of the following columns:
        - Estimated values: 'MrEst', 'ArEst', 'FeHEst'
        - True values: 'MrTrue', 'ArTrue', 'FeHTrue'
        - Raw values: 'Mr', 'Ar', 'FeH'

    Raises
    ------
    ValueError
        If none of the expected columns are found in the table.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n, 3) containing the extracted stellar parameters: Mr, Ar, FeH.
    """
    if {"MrEst", "ArEst", "FeHEst"}.issubset(sims.colnames):
        y = tuple(sims[col].data for col in ['MrEst', 'ArEst', 'FeHEst'])
    elif {"MrTrue", "ArTrue", "FeHTrue"}.issubset(sims.colnames):
        y = tuple(sims[col].data for col in ['MrTrue', 'ArTrue', 'FeHTrue'])
    elif {"Mr", "Ar", "FeH"}.issubset(sims.colnames):
        y = tuple(sims[col].data for col in ['Mr', 'Ar', 'FeH'])
    else:
        raise ValueError(
            "Neither columns ['MrTrue', 'ArTrue', 'FeHTrue'] nor ['MrEst', 'ArEst', 'FeHEst'] not found in the file")
    return np.array(y).T


def outputs_error_from_table(sims):
    """
    Extract output error columns from the Astropy Table.

    This function extracts columns related to the uncertainties of stellar parameters from the table.
    It checks for the presence of the expected column names related to the errors and selects the appropriate columns.

    Parameters
    ----------
    sims : `~astropy.table.Table`
        The Astropy Table containing output error data. The table must have the following columns:
        - Error columns: 'MrUnc', 'ArUnc', 'FeHUnc'

    Raises
    ------
    ValueError
        If the required columns for errors are not found in the table.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n, 3) containing the extracted uncertainties: MrUnc, ArUnc, FeHUnc.
    """
    if {"MrUnc", "ArUnc", "FeHUnc"}.issubset(sims.colnames):
        y = tuple(sims[col].data for col in ['MrUnc', 'ArUnc', 'FeHUnc'])
    else:
        raise ValueError(
            "Columns ['MrUnc', 'ArUnc', 'FeHUnc'] not found in the file")
    return np.array(y).T


def tensor_slices_from_table(sims):
    """
    Create numpy tensors from astropy table.

    Creates numpy tensors from astropy table. The output is a tuple of two tuples. The first tuple contains the input
    data and the second tuple contains the output data. The input data is a tuple of two arrays. The first array is the
    input data and the second array is the error of the input data. The output data is a tuple of three arrays. The
    first array is the absolute magnitude, the second array is the extinction and the third array is the metallicity.

    Parameters
    ----------
    file_path: table
        Astropy table with the data.

    Returns
    -------
    dataset: tf.data.Dataset
    """
    x, x_error = inputs_from_table(sims)
    y = outputs_from_table(sims)
    return x, x_error, y


def dataset_from_tensor_slices(x, x_error, y):
    """
    Create tensorflow dataset from numpy arrays.

    Creates tensorflow dataset from numpy arrays. The input data x should be a numpy array of shape (n, n_colors) where
    n is the number of stars and n_colors is the number of colors. The input error data x_error_temp should be a numpy
    array of shape (n, n_colors) where n is the number of stars and n_colors is the number of colors. The output data y
    should be a numpy array of shape (n, n_param) where n is the number of stars and n_param is the number of outputs.
    The output data should be in the order of absolute magnitude, extinction and metallicity.

    Parameters
    ----------
    x: np.array
        Input data. Shape should be (n, n_colors) where n is the number of stars and n_colors is the number of colors.
    x_error: np.array
        Input error data. Shape should be (n, n_colors) where n is the number of stars and n_colors is the number of
        colors.
    y: np.array
        Output data. Shape should be (n, n_param) where n is the number of stars and n_param is the number of outputs.
        The output data should be in the order of absolute magnitude, extinction and metallicity.

    Returns
    -------
    dataset: tf.data.Dataset
    """
    if x.shape[0] != x_error.shape[0]:
        raise Exception("x and x_error should have the same number of stars.")
    if x.shape[0] != y.shape[0]:
        raise Exception("x and y should have the same number of stars.")
    if x.shape[1] != x_error.shape[1]:
        raise Exception("x and x_error should have the same number of colors.")
    dataset = tf.data.Dataset.from_tensor_slices(((x, x_error), tuple(y[:, i] for i in range(y.shape[-1]))))
    return dataset


def dataset_batch(dataset, batch_size, strategy=None):
    """
    Batch dataset and apply prefetching while taking care of the strategy.

    Batches dataset and applies prefetching while taking care of the strategy. If strategy is None, then the dataset is
    batched and prefetched without any strategy. If strategy is tf.distribute.MirroredStrategy(), then the dataset is
    batched and prefetched with strategy.experimental_distribute_dataset(dataset).

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to be batched and prefetched.
    batch_size: int
        Batch size.
    strategy: tf.distribute.Strategy or None
        Strategy to be used. If None, then no strategy is used.

    Returns
    -------
    dataset: tf.data.Dataset
        Batched and prefetched dataset.
    """

    if "MirroredStrategy" in strategy._tf_api_names[0]:
        dataset = dataset.batch(batch_size).prefetch(10)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
    else:
        dataset = dataset.batch(batch_size).prefetch(10)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
    return dataset


def dataset_split_train_test(dataset, ds_size, test_split=0.2, seed=None):
    """
    Split dataset into train and test dataset.

    Splits dataset into train and test dataset. The test dataset is taken from the end of the dataset after the whole
    dataset is shuffled. The train dataset is the rest of the dataset.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to be split.
    ds_size: int
        Size of the dataset. Usually this is the tf.data.experimental.cardinality(dataset).numpy().
    test_split: float
        Fraction of the dataset to be used for test dataset.
    seed: int or None
        Seed of the shuffle. The same seed gives the same split for datasets with the same number of rows.

    Returns
    -------
    train_dataset: tf.data.Dataset
        Train dataset.
    test_dataset: tf.data.Dataset
        Test dataset.
    """
    dataset = dataset.shuffle(buffer_size=ds_size, seed=seed, reshuffle_each_iteration=False)
    test_size = int(test_split * ds_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)
    return train_dataset, test_dataset


def dataset_reduce(dataset, ds_size, reduce_to=0.1, seed=None):
    """
    Reduce dataset size.

    Reduces dataset size to reduce_to fraction of the original dataset size. The dataset is shuffled before the size
    reduction.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to be reduced.
    ds_size: int
        Size of the dataset. Usually this is the tf.data.experimental.cardinality(dataset).numpy().
    reduce_to: float
        Fraction of the dataset to be used for the reduced dataset.
    seed: int or None
        Seed of the shuffle.

    Returns
    -------
    dataset: tf.data.Dataset
        Reduced dataset.
    """
    dataset = dataset.shuffle(buffer_size=ds_size, seed=seed, reshuffle_each_iteration=False)
    reduced_size = int(reduce_to * ds_size)
    dataset = dataset.take(reduced_size)
    return dataset


def dataset_split_train_val(dataset, ds_size, val_split=0.2):
    """
    Split training dataset into train and validation dataset.

    Splits training dataset into train and validation dataset. The validation dataset is taken from the end of the
    training dataset after the whole training dataset is shuffled. The train dataset is the rest of the dataset. The
    train and validation dataset is shuffled again after the split with reshuffle_each_iteration=True.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to be split.
    ds_size: int
        Size of the dataset. Usually this is the tf.data.experimental.cardinality(dataset).numpy().
    val_split: float
        Fraction of the dataset to be used for validation dataset.

    Returns
    -------
    train_dataset: tf.data.Dataset
        Train dataset.
    val_dataset: tf.data.Dataset
        Validation dataset.
    """

    dataset = dataset.shuffle(buffer_size=ds_size, reshuffle_each_iteration=False)
    train_size = int((1 - val_split) * ds_size)
    val_size = int(val_split * ds_size)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    train_dataset = train_dataset.shuffle(buffer_size=train_size, reshuffle_each_iteration=True)
    val_dataset = val_dataset.shuffle(buffer_size=val_size, reshuffle_each_iteration=True)
    return train_dataset, val_dataset


def marginal_posterior_density_loss(eta=1.):
    """
    Marginal posterior density loss function.

    Marginal posterior density loss function. The loss function is defined as:
    $L = (y_true - mu)^2 + eta * ((y_true - mu)^2 - sig^2)^2$
    Parameters
    ----------
    eta: float
        Hyperparameter eta.

    Returns
    -------
    loss: function
        Loss function.
    """

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        mu = y_pred[:, :1]  # first output neuron
        sig = y_pred[:, 1:]  # second output neuron
        return tf.math.square(y_true - mu) + eta * tf.math.square(tf.math.square(y_true - mu) - tf.math.square(sig))

    return loss


def logarithmic_marginal_posterior_density_loss(y_true, y_pred):
    """
    Logarithmic marginal posterior density loss function.

    Logarithmic marginal posterior density loss function. The loss function is defined as:
    $L = log(\sum(y_true - mu)^2) + log(\sum((y_true - mu)^2 - sig^2)^2)$
    Parameters
    ----------
    y_true: tf.Tensor
        True output.
    y_pred: tf.Tensor
        Predicted output.

    Returns
    -------
    loss: tf.Tensor
        Loss function.
    """
    mu = y_pred[:, :1]  # first output neuron
    sig = y_pred[:, 1:]  # second output neuron
    return tf.math.log(tf.reduce_sum((y_true - mu) ** 2)) + tf.math.log(
        tf.reduce_sum(((y_true - mu) ** 2 - sig ** 2) ** 2))


def kl_divergence(y_true, y_pred):
    f"""
    Kullback-Leibler divergence loss function.

    Kullback-Leibler divergence loss function. This loss function measures the divergence between the normal 
    distribution N(0,1) and the predicted distribution of the valus of the $(true - pred)/\sigma_p$. The loss function
    is defined as:
    $L = log(\sigma) + (1 + \mu^2)/2\sigma^2 - 1/2$
    where $\mu$ and $\sigma$ are the mean and standard deviation of the $(true - pred)/\sigma_p$ distribution.
    Parameters
    ----------
    y_true: tf.Tensor
        True output.
    y_pred: tf.Tensor
        Predicted output.

    Returns
    -------
    loss: tf.Tensor
        Loss function.
    """
    pred = y_pred[:, :1]  # first output neuron
    sig_pred = y_pred[:, 1:]  # second output neuron
    cs = (y_true - pred) / sig_pred
    sigma = 0.741 * (np.percentile(cs, 75) - np.percentile(cs, 25))
    mu = tf.math.reduce_mean(cs)
    return tf.math.log(sigma) + (1 + mu ** 2) / (2 * sigma ** 2) - 1 / 2


def chi_squared(y_true, y_pred):
    """
    Chi squared loss function.

    Chi squared loss function. This loss function measures the chi squared between the true and predicted values. The
    loss function is defined as:
    $L = (y_true - y_pred) / sig_pred$

    Parameters
    ----------
    y_true: tf.Tensor
        True output.
    y_pred: tf.Tensor
        Predicted output.

    Returns
    -------
    loss: tf.Tensor
        Loss function.
    """
    pred = y_pred[:, :1]  # first output neuron
    sig_pred = y_pred[:, 1:] + 1e-10  # second output neuron
    cs = (y_true - pred) / sig_pred
    return tf.convert_to_tensor(0.741 * (np.percentile(cs, 75) - np.percentile(cs, 25)), dtype=tf.float32)


def mse(y_true, y_pred):
    """
    Mean squared error loss function.
    Parameters
    ----------
    y_true: tf.Tensor
        True output.
    y_pred: tf.Tensor
        Predicted output.

    Returns
    -------
    loss: tf.Tensor
        Loss function.
    """
    y_mu = tf.convert_to_tensor(y_pred[:, :1])
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.math.squared_difference(y_mu, y_true)


def coverage(y_true, y_pred):
    """
    Coverage loss function.

    Coverage loss function. This loss function measures the coverage of the true values by the predicted interval. The
    interval is defined as $[mu - sig, mu + sig]$. If sigma predicted is correct then the coverage should be 0.68.
    Parameters
    ----------
    y_true: tf.Tensor
        True output.
    y_pred: tf.Tensor
        Predicted output.

    Returns
    -------
    loss: tf.Tensor
        Loss function.
    """
    mu = y_pred[:, :1]  # first output neuron
    sig = y_pred[:, 1:]  # second output neuron
    return tf.reduce_mean(tf.cast(tf.cast(tf.abs(y_true - mu), tf.float32) < tf.cast(tf.abs(sig), tf.float32),
                                  tf.float32))


def dataset_split_x_y(dataset):
    """
    Split dataset into x and y.

    Splits tf.data.Dataset into x and y arrays.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset to be split.

    Returns
    -------
    x: np.array
        Input data.
    y: np.array
        Output data.
    """
    x, y = list(dataset.batch(tf.data.experimental.cardinality(dataset).numpy()))[0]
    return x, y


if __name__ == "__main__":
    pass
