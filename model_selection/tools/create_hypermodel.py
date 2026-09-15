import tensorflow as tf
import math
import numpy as np
import keras_tuner as kt
import tools

if __name__ == "__main__":
    import create_model
    import LRFinder
else:
    import tools.create_model as create_model
    import tools.LRFinder as LRFinder


def countparameterspace(arh):
    permutations = 1
    parc2 = 0
    for i1 in range(arh["Layers"][0], arh["Layers"][1], 1):
        parc = 1
        for i in range(i1):
            parc *= len(range(arh["Units"+str(i)][0], arh["Units"+str(i)][1], 1))
            parc *= len(arh["Activation"+str(i)])
        parc2 += parc
    permutations *= parc2
    return float(permutations)


def get_best_hyperparameters(tuner, num_trials=1):
    trials = [t
              for t in tuner.oracle.trials.values()
              if (t.status == kt.engine.trial.TrialStatus.COMPLETED) & (not np.isnan(t.score))]
    sorted_trials = sorted(trials, key=lambda trial: trial.score, reverse=tuner.oracle.objective.direction == "max")
    hyperparameters = [trial.hyperparameters for trial in sorted_trials if not np.isnan(trial.score)]
    return hyperparameters[:num_trials]


def createdefaulthyperarhitecture():
    activations_choice = ["elu", "gelu", "hard_sigmoid",
                          "linear", "relu", "selu", "sigmoid", "softmax",
                          "softplus", "softsign", "swish", "tanh"]
    hyperarh = {"Layers": [1, 10]}

    for i in range(hyperarh["Layers"][1]):
        hyperarh["Units" + str(i)] = [1, 50]
        hyperarh["Activations" + str(i)] = activations_choice
    return hyperarh


def create_architecture_dictionary(hp, hyperarh, output_units=3):
    architecture = {"units": [], "activations": []}

    if hyperarh is None:
        hyperarh = createdefaulthyperarhitecture()

    layers = hp.Int(name="Layers", min_value=hyperarh["Layers"][0], max_value=hyperarh["Layers"][1], step=1)
    for i in range(layers):
        units = hp.Int(name="Units" + str(i), min_value=hyperarh["Units" + str(i)][0],
                       max_value=hyperarh["Units" + str(i)][1], step=1)
        activations = hp.Choice(name="Activations" + str(i), values=hyperarh["Activations" + str(i)])
        architecture["units"].append(units)
        architecture["activations"].append(activations)
    architecture["units"].append(output_units)
    architecture["activations"].append("linear")
    return architecture


def create_hypertune_models(train_data, hyper_arh=None, version=1):
    def prepare_model(hp):
        loss_weights = None
        x, y = train_data[0], train_data[1]
        if (type(y) is tuple) or (type(y) is list):
            output_units = len(y)
        else:
            output_units = y.shape[-1]
        architecture_model = create_architecture_dictionary(hp, hyper_arh, output_units=output_units)
        if loss_weights is None:
            if version % 10 != 0:
                # single output models
                loss_weights = np.array([1])
            else:
                # multi output models
                #loss_weights = 1 / np.abs(np.array(y).mean(axis=1)).squeeze()
                # Mahalanobis distance as a loss function
                loss_weights = np.array([1/(0.1**2), 1/(0.02**2), 1/(0.1**2)])
        if version // 10 == 1:
            # simple model
            model = create_model.photozannv1(architecture_model, x[0].shape)
        elif version // 10 == 2:
            # naive model
            model = create_model.photozannv2(architecture_model, x[0].shape)
        elif version // 10 == 3:
            # sampling model
            model = create_model.photozannv3(architecture_model, x[0].shape)
        else:
            raise Exception('Version of the model must integer between [1,3]')
        model_size = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]) + \
                     np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        print("Model parameter size:", model_size)
        max_lr = 0.01
        """lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(max_lr,
                                                                     decay_steps=decay_epoch * math.ceil(y[0].shape[0] /
                                                                                                batch_size),
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)"""

        model.compile(optimizer=tf.keras.optimizers.Adam(max_lr, clipvalue=10.0),
                      loss='mse',
                      loss_weights=loss_weights)
        return model

    return prepare_model


def create_hypertune_probability_models(train_data, hyper_arh=None, version=1):
    def prepare_model(hp):
        loss_weights = None
        x, y = train_data[0], train_data[1]
        if (type(y) is tuple) or (type(y) is list):
            output_units = len(y)
        else:
            output_units = y.shape[-1]
        architecture_model = create_architecture_dictionary(hp, hyper_arh, output_units=output_units)
        if loss_weights is None:
            if version % 10 != 0:
                # single output models
                loss_weights = np.array([1])
            else:
                # multi output models
                #loss_weights = 1 / np.abs(np.array(y).mean(axis=1)).squeeze()
                # Mahalanobis distance as a loss function
                loss_weights = np.array([1/(0.1**2), 1/(0.02**2), 1/(0.1**2)])
        if version // 10 == 1:
            # simple model
            model = create_model.photozannv1(architecture_model, x[0].shape, output_units=2)
        elif version // 10 == 2:
            # naive model
            model = create_model.photozannv2(architecture_model, x[0].shape, output_units=2)
        elif version // 10 == 3:
            # sampling model
            model = create_model.photozannv3(architecture_model, x[0].shape, output_units=2)
        else:
            raise Exception('Version of the model must integer between [1,3]')
        model_size = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]) + \
                     np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        print("Model parameter size:", model_size)
        max_lr = 0.001
        """lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(max_lr,
                                                                     decay_steps=decay_epoch * math.ceil(y[0].shape[0] /
                                                                                                batch_size),
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)"""

        model.compile(optimizer=tf.keras.optimizers.Adam(max_lr, clipvalue=10.0),
                      loss=tools.create_model.logarithmic_marginal_posterior_density_loss,
                      loss_weights=loss_weights,
                      metrics=[tools.create_model.coverage, tools.create_model.mse])
        return model

    return prepare_model