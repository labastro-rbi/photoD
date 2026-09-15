"""
Timing of the full pipeline: 30 runs of photod.pipeline with the same settings, written to
../outputs/training_time_<number of GPUs>.csv. Run it with different CUDA_VISIBLE_DEVICES (see training_time.sh).
"""
import sys

import pandas as pd
import tensorflow as tf

sys.path.append("../Environment")
import photod.pipeline as pipeline


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Train a photometric distance model")
    parser.add_argument("--input_training_path", type=str,
                        help="Path to the file with the input data for training",
                        default="../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt")
                        #default="../../data/BayesMethod3D_SEGUEpatch-l110-KarloTest-short1.txt")
    parser.add_argument("--input_prediction_path", type=str,
                        help="Path to the file with the input data for prediction",
                        default="")
    parser.add_argument("--model_path", type=str,
                        help="Path to the file with the trained model",
                        default="../PhotoD.keras")
    parser.add_argument("--output_prediction_path", type=str,
                        help="Path to the file to output for prediction",
                        default="")
    parser.add_argument("--train_size", type=int, help="Size of the training set", default=10000)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=1024)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=1024)
    parser.add_argument("--iterations", type=int, help="Number of iterations", default=2)
    parser.add_argument("--decay_epochs", type=int, help="Decay epochs", default=10)
    parser.add_argument("--decay_rate", type=float, help="Decay rate", default=0.7)
    parser.add_argument("--output_file_path", type=str, help="Path to the file with the output data",
                        default="")
    parser.add_argument("--do_plots", type=bool, help="Create the plots", default=False)
    parser.add_argument("--do_metrics", type=bool, help="Create the metrics", default=False)
    parser.add_argument("--plot_path", type=str, help="Path to the folder with the plots", default="")
    return parser


def main():
    args = arg_parser().parse_args()
    for i in range(30):
        result = pipeline.main(args)
        times_mesurements = result.time
        df = pd.DataFrame.from_dict(times_mesurements, orient='index').T
        df["index"] = i
        df["GPUS"] = len(tf.config.experimental.list_physical_devices('GPU'))
        df["Training_size"] = result.training_size
        df["Test_size"] = result.evaluating_size
        df = df.set_index("index")
        if i == 0:
            df.to_csv(
                "../outputs/training_time_" + str(len(tf.config.experimental.list_physical_devices('GPU'))) + ".csv",
                mode='w', header=True)
        else:
            df.to_csv(
                "../outputs/training_time_" + str(len(tf.config.experimental.list_physical_devices('GPU'))) + ".csv",
                mode='a', header=False)
    return None

if __name__ == "__main__":
    main()
