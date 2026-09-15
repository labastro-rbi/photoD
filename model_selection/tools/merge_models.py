import tensorflow as tf
import os
import tools
import argparse
import sys

def merge_models (path):
    paths1 = [i for i in os.listdir(path) if i.split("_trained")[0].split("v")[-1][1] == "1"]
    paths1.sort()
    paths2 = [i for i in os.listdir(path) if i.split("_trained")[0].split("v")[-1][1] == "2"]
    paths2.sort()
    paths3 = [i for i in os.listdir(path) if i.split("_trained")[0].split("v")[-1][1] == "3"]
    paths3.sort()
    assert (len(paths1) == len(paths2)) and (len(paths1) == len(paths3))
    for i in range(len(paths1)):
        model1 = tf.keras.models.load_model(os.path.join(path, paths1[i]),  compile=False,
                                            custom_objects={"correlation": None, "Sampling": tools.create_model.Sampling})
        model2 = tf.keras.models.load_model(os.path.join(path, paths2[i]),  compile=False,
                                            custom_objects={"correlation": None, "Sampling": tools.create_model.Sampling})
        model3 = tf.keras.models.load_model(os.path.join(path, paths3[i]),  compile=False,
                                            custom_objects={"correlation": None, "Sampling": tools.create_model.Sampling})
        # input_x = tf.keras.layers.Input(shape=model1.input_shape[0][1:], name='x_input_main')
        # input_xerr = tf.keras.layers.Input(shape=model1.input_shape[1][1:], name='xerr_input_main')
        model1._name = "Mr_prediction"
        model1.layers[-1]._name = "Mr_output"
        model2._name = "Ar_prediction"
        model3._name = "FeH_prediction"
        output2 = model2(model1.inputs)
        output3 = model3(model1.inputs)
        model = tf.keras.Model(inputs=model1.inputs, outputs=[model1.outputs, output2, output3], name="Merged_model")
        save_name = paths1[i][:11] + "4" + paths1[i][12:]
        print (save_name)
        model.save(os.path.join(path, save_name), include_optimizer=False)


def remove_optimizers(path):
    for i in os.listdir(path):
        model = tf.keras.models.load_model(os.path.join(path, i),  compile=False, custom_objects={"correlation": None, "Sampling": tools.create_model.Sampling})
        model.save(os.path.join(path, i), include_optimizer=False)


def main(args):
    remove_optimizers(args.path)
    merge_models(args.path)

def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default="../models/trained_chiTest4/",
                        help='Path to save trained model.')
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

