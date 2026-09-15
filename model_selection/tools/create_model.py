import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


def random_arhitecture(seed=None):
    """
    Create a random arhitecture dictionary for a neural network.
    :param seed: seed for the random number generator
    :return: random arhitecture dictionary
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
    Extract the arhitecture dictionary from a given model.
    :param model: model to extract the arhitecture from
    :return: arhitecture dictionary
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

def model_memory_usage(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    The model shapes are multipled by the batch size, but the weights are not.
    :param model: A Keras model.
    :param batch_size: The batch size you intend to run the model with. If you have already specified the batch size in
    the model itself, then pass `1` as the argument here.
    :return: An estimate of the Keras model's memory usage in bytes.
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += model_memory_usage(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory


def model_parameter_num(model):
    """
    Return the number of parameters of a given Keras model.
    :param model: model to evaluate
    :return: number of parameters
    """
    model_size = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]) + np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    zero_param = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights if np.prod(v.get_shape()) == 0]) + np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights if np.prod(v.get_shape()) == 0])
    return model_size, zero_param


class Sampling(tf.keras.layers.Layer):
    """
    Sampling layer for the SAMPLING models.
    """
    def call(self, inputs, **kwargs):
        """
        Call function of the layer. It samples from a normal distribution with mean and sigma given by the inputs.

        :param inputs: Mean and sigma of the normal distribution
        :param kwargs:
        :return: Sample from the normal distribution
        """
        mean, sigma = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        return tf.random.normal(shape=(batch, dim), mean=mean, stddev=sigma)

    def get_config(self):
        """
        Get the configuration of the layer.
        :return: configuration of the layer
        """
        config = super().get_config()
        #config["k"] = self.k
        return config


def correlation(y_true, y_pred):
    cov = tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)
    if tf.math.is_nan(cov):
        cov = tf.constant(0.)
    return cov

def marginal_posterior_density_loss(eta=1.):
    def loss(y_true, y_pred):
        mu = y_pred[:, :1]  # first output neuron
        sig = y_pred[:, 1:]  # second output neuron
        return tf.reduce_mean((y_true - mu) ** 2 + eta*((y_true - mu) ** 2 - sig ** 2) ** 2)
    return loss

def logarithmic_marginal_posterior_density_loss(y_true, y_pred):
    mu = y_pred[:, :1] # first output neuron
    sig = y_pred[:, 1:] # second output neuron
    return tf.math.log(tf.reduce_sum((y_true-mu)**2)) + tf.math.log(tf.reduce_sum(((y_true-mu)**2 - sig**2)**2))

def mse(y_true, y_pred):
    y_mu = tf.convert_to_tensor(y_pred[:, :1])
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(tf.math.squared_difference(y_mu, y_true), axis=-1)

def coverage (y_true, y_pred):
    mu = y_pred[:, :1] # first output neuron
    sig = y_pred[:, 1:] # second output neuron
    return tf.reduce_mean(tf.cast(tf.abs(y_true-mu)<tf.abs(sig), tf.float32))

class CorrelationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):

        loss_logs = [log for log in logs.keys() if ("_loss" in log) & ("val" not in log)]
        val_logs = [log for log in logs.keys() if ("_loss" in log) & ("val" in log) & (log != "val_loss")]
        for i, log in enumerate(loss_logs):
            logs["train_loss"+str(i+1)] = logs[log] * self.model.compiled_loss._loss_weights[i]
            del logs[log]
        for i, log in enumerate(val_logs):
            logs["val_loss"+str(i+1)] = logs[log] * self.model.compiled_loss._loss_weights[i]
            del logs[log]
        train_logs = [log for log in logs.keys() if ("correlation" in log) & ("val" not in log)]
        if len(train_logs) > 0:
            tmp_sum = 0
            for i, log in enumerate(train_logs):
                tmp_sum += logs[log]
                logs["train_corr"+str(i)] = logs[log]
                del logs[log]
            logs["TRAIN_CORRELATION"] = tmp_sum / len(train_logs)

        val_logs = [log for log in logs.keys() if ("correlation" in log) & ("val" in log)]

        if len(val_logs) > 0:
            tmp_sum = 0
            for i, log in enumerate(val_logs):
                tmp_sum += logs[log]
                logs["val_corr"+str(i)] = logs[log]
                del logs[log]
            logs["VAL_CORRELATION"] = tmp_sum / len(val_logs)
        return logs


def photozannv1(architecture, x_shape, model_name="simple", output_units=1):
    """
    Create a SIMPLE model with a number of outputs defined by the architecture
    :param architecture: dictionaries containing the number of units and the activation function for each layer
    :param x_shape: shape of the input layer
    :param model_name: Name of the model
    :return: model
    """
    input_x = tf.keras.layers.Input(shape=x_shape[1:], name='x_input')
    input_xerr = tf.keras.layers.Input(shape=x_shape[1:], name='xerr_input')
    x = input_x
    x = tf.keras.layers.BatchNormalization(name="Normalization", center=True, scale=True)(x)
    assert len(architecture["units"]) == len(architecture["activations"])
    for i in range(len(architecture["units"])-1):
        x = tf.keras.layers.Dense(units=architecture["units"][i], activation=architecture["activations"][i],
                                  name="dense" + str(i))(x)
    output = []
    for i in range(architecture["units"][-1]):
        output.append(tf.keras.layers.Dense(units=output_units,
                                            activation=architecture["activations"][-1],
                                            name="output" + str(i))(x))
    model = tf.keras.Model(inputs=(input_x, input_xerr), outputs=output, name=model_name)
    return model


def photozannv2(architecture, x_shape, model_name="naive", output_units=1):
    """
    Create an NAIVE model with a number of outputs defined by the architecture
    :param architecture: dictionaries containing the number of units and the activation function for each layer
    :param x_shape: shape of the input layer
    :param model_name: name of the model
    :return: model
    """
    input_x = tf.keras.layers.Input(shape=x_shape[1:], name='x_input')
    input_xerr = tf.keras.layers.Input(shape=x_shape[1:], name='xerr_input')
    x = tf.keras.layers.concatenate([input_x, input_xerr], name="concat_inputs")
    x = tf.keras.layers.BatchNormalization(name="Normalization", center=True, scale=True)(x)
    assert len(architecture["units"]) == len(architecture["activations"])
    for i in range(len(architecture["units"])-1):
        x = tf.keras.layers.Dense(units=architecture["units"][i], activation=architecture["activations"][i],
                                  name="dense" + str(i))(x)
    output = []
    for i in range(architecture["units"][-1]):
        output.append(tf.keras.layers.Dense(units=output_units,
                                            activation=architecture["activations"][-1],
                                            name="output" + str(i))(x))
    model = tf.keras.Model(inputs=(input_x, input_xerr), outputs=output, name=model_name)
    return model


def photozannv3(architecture, x_shape, model_name="sampling", output_units=1):
    """
    Create an SAMPLING model with a number of outputs defined by the architecture
    :param architecture: dictionaries containing the number of units and the activation function for each layer
    :param x_shape: shape of the input layer
    :param model_name: model name
    :return: model
    """
    input_x = tf.keras.layers.Input(shape=x_shape[1:], name='x_input')
    input_xerr = tf.keras.layers.Input(shape=x_shape[1:], name='xerr_input')
    x = Sampling(name="sampling")([input_x, input_xerr])
    x = tf.keras.layers.BatchNormalization(name="Normalization", center=True, scale=True)(x)
    assert len(architecture["units"]) == len(architecture["activations"])
    for i in range(len(architecture["units"])-1):
        x = tf.keras.layers.Dense(units=architecture["units"][i], activation=architecture["activations"][i],
                                  name="dense" + str(i))(x)
    output = []
    for i in range(architecture["units"][-1]):
        output.append(tf.keras.layers.Dense(units=output_units,
                                            activation=architecture["activations"][-1],
                                            name="output" + str(i))(x))
    model = tf.keras.Model(inputs=(input_x, input_xerr), outputs=output, name=model_name)
    return model

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    import tools
    x_train, y_train, x_val, y_val, x_test, y_test = tools.create_simulated_data.txt_file_split_to_npy("../data/TRILEGAL_three_pix_triout_V1.txt",
                                                                                                       seed=None)["sims"]
    arh = random_arhitecture(32)
    model1 = photozannv2(arh, x_train[0].shape)
    reconstructed_arh = extract_arhitecture(model1)
    print("arh")
    print(arh)
    print("reconstructed_arh")
    print(reconstructed_arh)
    print (arh == reconstructed_arh)
