import argparse
import sys
import os
import math
import keras_tuner as kt
import tensorflow as tf

sys.path.append("../")
import tools


def main(args):
    total_num_epochs = int(args.epochs * (
            math.log(args.epochs, int(args.factor)) ** 2) * args.hyperband_iterations)
    total_num_models = int((1 + math.log(args.epochs, int(args.factor))) * args.hyperband_iterations)
    print("Total number of epochs: ", int(total_num_epochs))
    print("Total number of models: ", total_num_models)
    x_train, y_train, x_val, y_val, x_test, y_test = tools.create_simulated_data.txt_file_split_to_npy(args.data_path,
                                                                                                       seed=args.seed)[
        "sims"]
    del x_test, y_test
    if "p" in str(args.version):
        version = int(args.version.split("p")[0])
        probbablistic = True
    else:
        version = int(args.version)
        probbablistic = False
    if version % 10 != 0:
        y_train = (y_train[(version % 10) - 1],)
        y_val = (y_val[(version % 10) - 1],)
    train_data = (x_train, y_train)
    gpus_devices = ["/gpu:" + str(j) for j, gpu in enumerate(tf.config.experimental.list_physical_devices('GPU'))]
    if len(gpus_devices) > 0:
        print("Using " + str(len(gpus_devices)) + " GPUs.")
        strategy = tf.distribute.MirroredStrategy(devices=gpus_devices)
    else:
        print("Using CPU.")
        strategy = tf.distribute.get_strategy()
    hyper_arh = tools.create_hypermodel.createdefaulthyperarhitecture()
    if probbablistic:
        print("Probabilistic model.")
        h_model = tools.create_hypermodel.create_hypertune_models(train_data,
                                                                  hyper_arh=hyper_arh,
                                                                  version=version)
    else:
        print("Deterministic model.")
        h_model = tools.create_hypermodel.create_hypertune_probability_models(train_data,
                                                                              hyper_arh=hyper_arh,
                                                                              version=version)
    tuner = kt.Hyperband(hypermodel=h_model,
                         objective='val_loss',
                         max_epochs=args.epochs,
                         factor=int(args.factor),
                         hyperband_iterations=args.hyperband_iterations,
                         directory=args.model_destination,
                         overwrite=args.overwrite,
                         project_name="photoZv" + str(args.version),
                         distribution_strategy=strategy)
    earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10*args.decay_epochs, verbose=1,
                                                        restore_best_weights=True, min_delta=1e-4)
    terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
    reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_rate,
                                                                patience=args.decay_epochs, verbose=1)
    if strategy is not tf.distribute.get_strategy():
        train_dataset = tools.create_simulated_data.create_tf_dataset(x_train, y_train, batch_size=args.batch_size)
        val_dataset = tools.create_simulated_data.create_tf_dataset(x_val, y_val, batch_size=args.batch_size)
        tuner.search(train_dataset, epochs=args.epochs, verbose=2, validation_data=val_dataset,
                     callbacks=[earlystopping_kb, terminateonnan_kb, reducelronplateau_kb])
    else:
        tuner.search(x=x_train, y=y_train, epochs=args.epochs,
                     verbose=2, validation_data=(x_val, y_val),
                     batch_size=args.batch_size, callbacks=[earlystopping_kb, terminateonnan_kb, reducelronplateau_kb])
    best_hps = tools.create_hypermodel.get_best_hyperparameters(tuner, num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    os.makedirs("./models/", exist_ok=True)
    best_model.save("../models/untrained/photozannv" + str(args.version) + "_untrained", include_optimizer=False)
    del tuner, best_model, best_hps


def parse_arguments(args):
    """Parse command line arguments.
    Args:
        args (list): Command line arguments.
    Returns:
        args (Namespace): Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_destination', type=str,
                        default="../../Tuner/",
                        help='Path where to save the models.')
    parser.add_argument('--data_path', type=str,
                        default="../data/simCatalog_three_pix_triout_v1.txt",
                        help='Path from where to load data.')
    parser.add_argument('-o', '--overwrite', type=bool,
                        default=False,
                        help='Overwrite previous data.')
    parser.add_argument('--epochs', type=int,
                        default=2,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=1024 * 8,
                        help='Batch size.')
    parser.add_argument('--version', type=str,
                        default="10p",
                        help='Version of the model.')
    parser.add_argument('--factor', type=int,
                        default=2,
                        help='Number of hyperband rounds.')
    parser.add_argument('--hyperband_iterations', type=int,
                        default=1,
                        help='Repetitions of each full hyperband algorithm.')
    parser.add_argument('--decay_rate', type=float,
                        default=0.9,
                        help='Decay rate.')
    parser.add_argument('--decay_epochs', type=int,
                        default=10,
                        help='Decay steps in epochs.')
    parser.add_argument('--seed', type=float,
                        default=None,
                        help='Seed for pseudorandom.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
