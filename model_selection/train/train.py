import tensorflow as tf
import argparse
import os
import math
import sys
sys.path.append("../")
import tools
import numpy as np
import time


def main(args):
    min_loss = 100000000
    start_time = time.time()
    model_name = args.model_path.split("/")[-1].split("_")[0]
    if args.cut:
        model_name += "_cut"
    model_name += "_trained"
    print(model_name)
    x_train, y_train, x_val, y_val, _, _ = tools.create_simulated_data.txt_file_split_to_npy(args.data_path,
                                                                                         seed=args.seed)["sims"]
    if args.version == "auto":
        args.version = args.model_path.split("_")[0].split("v")[-1]
    if str(args.version).isdigit():
        args.version = int(args.version)
        probbabilistic = False
    elif "p" in str(args.version):
        args.version = int(args.version.split("p")[0])
        probbabilistic = True
    else:
        raise Exception("Version must be a number.")
    if args.version % 10 != 0:
        y_train = (y_train[(args.version % 10)-1],)
        y_val = (y_val[(args.version % 10)-1],)
        # single output models
        loss_weights = np.array([1])
    else:
        # multi output models use Mahalanobis distance as a loss function weights
        loss_weights = np.array([1 / (0.1 ** 2), 1 / (0.02 ** 2), 1 / (0.1 ** 2)])
    if args.model_path == "None":
        arhitecture = tools.create_model.random_arhitecture(seed=args.seed)
    elif os.path.exists(args.model_path):
        model = tf.keras.models.load_model(args.model_path, compile=False, custom_objects={"Sampling": tools.create_model.Sampling})
        arhitecture = tools.create_model.extract_arhitecture(model)
    else:
        raise Exception("MODEL path "+args.model_path+" does not exist.")
    print("Version:", args.version, end="")
    if probbabilistic:
        loss_fn = tools.create_model.marginal_posterior_density_loss(1.0)
        output_units = 2
        print("p")
    else:
        loss_fn = "mse"
        output_units = 1
        print()
    # this is the place where data cut is made
    if args.cut:
        mask = ((x_train[0][:, 0] + x_train[0][:, 1] + x_train[0][:, 2]) < 25.7).flatten()
        x_train_cut = [x[mask] for x in x_train]
        y_train_cut = [y[mask] for y in y_train]

        mask = ((x_val[0][:, 0] + x_val[0][:, 1] + x_val[0][:, 2]) < 25.7).flatten()
        x_val_cut = [x[mask] for x in x_val]
        y_val_cut = [y[mask] for y in y_val]

        x_train = x_train_cut
        y_train = y_train_cut
        x_val = x_val_cut
        y_val = y_val_cut
    for i in range(args.iterations):
        # Check the version of the model
        if args.version // 10 == 1:
            model = tools.create_model.photozannv1(arhitecture, x_train[1].shape, output_units=output_units)
        elif args.version // 10 == 2:
            model = tools.create_model.photozannv2(arhitecture, x_train[1].shape, output_units=output_units)
        elif args.version // 10 == 3:
            model = tools.create_model.photozannv3(arhitecture, x_train[1].shape, output_units=output_units)
        else:
            raise Exception('Version of the model must integer between [1,3]')
        max_lr = 0.01
        model.compile(optimizer=tf.keras.optimizers.Adam(max_lr, clipvalue=10.0),
                      loss=loss_fn, metrics=[tools.create_model.coverage, tools.create_model.mse], loss_weights=loss_weights)
        earlystopping_kb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10*args.decay_epochs, verbose=1,
                                                            restore_best_weights=True, min_delta=1e-4)
        terminateonnan_kb = tf.keras.callbacks.TerminateOnNaN()
        reducelronplateau_kb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.decay_rate,
                                                                    patience=args.decay_epochs, verbose=1)
        model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_val, y_val),
                  callbacks=[earlystopping_kb, terminateonnan_kb, reducelronplateau_kb], verbose=2)
        print("Training time:", time.time()-start_time)
        model.train_time = time.time()-start_time
        eval = model.evaluate(x_val, y_val, verbose=0, batch_size=args.batch_size)[0]
        print ("loss = ", eval)
        if eval < min_loss:
            min_loss = eval
            model.save(os.path.join(args.model_output, model_name), include_optimizer=False)
            print("Model saved.")


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output', type=str,
                        default="../../models/trained/",
                        help='Path to save trained model.')
    parser.add_argument('--model_path', type=str,
                        default="../models/untrained/photozannv20p_untrained",
                        help='Path to untrained model.')
    parser.add_argument('--data_path', type=str,
                        default="../data/simCatalog_three_pix_triout_chiTest4.txt",
                        help='Path from where to load data.')
    parser.add_argument('--version', type=str,
                        default="auto",
                        help='Version of the model')
    parser.add_argument('--cut', type=argparse.BooleanOptionalAction,
                        default=False,
                        help='To cut the data for rmag between 25 and 26 or not.')
    parser.add_argument('--iterations', type=int,
                        default=5,
                        help='Number of iterations.')
    parser.add_argument('--epochs', type=int,
                        default=8192,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=16384,
                        help='Batch size.')
    parser.add_argument('--decay_rate', type=float,
                        default=0.9,
                        help='Decay rate.')
    parser.add_argument('--decay_epochs', type=int,
                        default=10,
                        help='Decay steps in epochs.')
    parser.add_argument('--seed', type=int,
                        default=None,
                        help='Seed for pseudorandom.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
