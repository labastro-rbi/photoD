import numpy as np
import pandas as pd
from astropy.table import Table
import os
import tensorflow as tf


def create_tf_dataset(x, y, batch_size, strategy=None):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=max(x[0].shape))
    if strategy is not None:
        dataset = dataset.batch(batch_size).repeat()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        dataset = strategy.experimental_distribute_dataset(dataset)
    else:
        dataset = dataset.batch(batch_size).prefetch(20)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

    return dataset


def read_trilegallsst(path='./TRILEGAL_three_pix_triout_V1.txt'):
    """
    Read TRILEGAL simulation augmented with LSST colors, see TRILEGAL_makeTestFile.ipynb.
    :param path: path to the file
    :return: simulation table
    """
    colnames = ['glon', "glat", "comp", "logg", "FeH", "Mr", "DM", "Ar", "rmagObs0", "ug0", "gr0", "ri0", "iz0",
     "rmag", "ugObs", "grObs", "riObs", "izObs", "uErr", "gErr", "rErr", "iErr", "zErr", "ugSL", "grSL", "riSL",
     "izSL", "ugErrSL", "grErrSL", "riErrSL", "izErrSL"]
    change_colnames = {"ugObs": "ug", "grObs": "gr", "riObs": "ri", "izObs": "iz"}
    # comp: Galactic component the star belongs to: 1→thin disk; 2→thick disk; 3→halo; 4→bulge; 5→Magellanic Clouds.
    # Mr = rmag - Ar - DM - 5
    # rmag0, ug0...iz0: intrinsic values without dust extinction (but include photometric noise)
    # rmag, ug...iz: dust extinction included
    # uErr...zErr: photometric noise
    sims = Table.read(path, format='ascii', names=colnames)
    #for col in change_colnames.keys():
    #    sims.rename_column(col, change_colnames[col])
    sims['gi0'] = sims['gr0'] + sims['ri0']
    sims['gi'] = sims['grObs'] + sims['riObs']
    return sims


def magnitudes_from_colors(colors):
    """
    Convert colors to magnitudes.
    :param colors: np.array of colors
    :return: np.array of magnitudes
    """
    r_mag, ug, gr, ri, iz = (colors[:, i] for i in range(colors.shape[1]))
    i_mag = r_mag - ri
    g_mag = r_mag + gr
    z_mag = i_mag - iz
    u_mag = g_mag + ug
    return np.array([r_mag, u_mag, g_mag, i_mag, z_mag]).T


def colors_from_maginitudes(magnitudes):
    """
    Convert magnitudes to colors.
    :param magnitudes: np.array of magnitudes
    :return: np.array of colors
    """
    r_mag, u_mag, g_mag, i_mag, z_mag = (magnitudes[:, i] for i in range(magnitudes.shape[1]))
    ug = u_mag - g_mag
    gr = g_mag - r_mag
    ri = r_mag - i_mag
    iz = i_mag - z_mag
    return np.array([r_mag, ug, gr, ri, iz]).T


def magnitude_from_colors_errror(colors_err):
    """
    Convert colors error to magnitudes error.
    :param colors_err: np.array of colors error
    :return: np.array of magnitudes error
    """
    r_mag_err, ug_err, gr_err, ri_err, iz_err = (colors_err[:, i] for i in range(colors_err.shape[1]))
    i_mag_err = np.sqrt(-r_mag_err**2 + ri_err**2)
    g_mag_err = np.sqrt(-r_mag_err**2 + gr_err**2)
    z_mag_err = np.sqrt(-i_mag_err**2 + iz_err**2)
    u_mag_err = np.sqrt(-g_mag_err**2 + ug_err**2)
    return np.array([r_mag_err, u_mag_err, g_mag_err, i_mag_err, z_mag_err]).T


def colors_from_maginitudes_errror(magnitudes_err):
    """
    Convert magnitudes error to colors error.
    :param magnitudes_err: np.array of magnitudes error
    :return: np.array of colors error
    """
    r_mag_err, u_mag_err, g_mag_err, i_mag_err, z_mag_err = (magnitudes_err[:, i] for i in range(magnitudes_err.shape[1]))
    ug_err = np.sqrt(u_mag_err**2 + g_mag_err**2)
    gr_err = np.sqrt(g_mag_err**2 + r_mag_err**2)
    ri_err = np.sqrt(r_mag_err**2 + i_mag_err**2)
    iz_err = np.sqrt(i_mag_err**2 + z_mag_err**2)
    return np.array([r_mag_err, ug_err, gr_err, ri_err, iz_err]).T


def split_by_magnitude_index(sims):
    """
    Split simulation by magnitude.
    :param sims: simulation table
    :return: tuple simulation tables split by magnitude
    """
    sims['umag'] = sims['rmag'] + sims['grObs'] + sims['ugObs']
    sims_actual = sims == sims
    sims1 = sims['umag'] < 25
    sims2 = (sims['umag'] > 25) & (sims['umag'] < 26)
    sims3 = (sims['umag'] > 26) & (sims['umag'] < 27)
    # try 0.2 < g-r < 0.6 as in Tomography II, where photometric FeH should work the best
    simsb = (sims['gr0'] > 0.2) & (sims['gr0'] < 0.6) & (sims['umag'] < 26)
    return sims_actual, sims1, sims2, sims3, simsb


def split_train_test_val_index(sims, split, seed=None):
    """
    Split simulation by train, test and validation. Split is the fraction of train data. The rest is split equally 
    between test and validation. This only returns the mask, not the actual data.
    :param sims: simulation table
    :param split: fraction of train data
    :param seed: random seed
    :return: tuple of boolean mask for train, test and validation to be further used in splitting the data
    """
    np.random.seed(seed)
    index_all = np.random.choice(a=["train", "val", "test"], size=len(sims), p=[split, (1 - split) / 2, (1 - split) / 2])
    index_train = index_all == "train"
    index_val = index_all == "val"
    index_test = index_all == "test"
    return index_train, index_val, index_test


def split_by_xy(d):
    """
    """
    x = d["rmag", "ugObs", "grObs", "riObs", "izObs"].to_pandas()
    y = d["Mr", "Ar", "FeH"].to_pandas()
    x_error = pd.DataFrame()
    x_error["rErr"] = d["rErr"].data
    for col_name in x.columns[1:]:
        error1 = d[col_name[0]+"Err"].data
        error2 = d[col_name[1]+"Err"].data
        x_error[col_name+"Err"] = np.sqrt(np.square(error1)+np.square(error2))
    return (np.array(x), np.array(x_error)), \
        (np.array(y["Mr"]).reshape(-1, 1), np.array(y["Ar"]).reshape(-1,1), np.array(y["FeH"]).reshape(-1,1))


def prepare_data(sims_data, split, seed=None):
    train_index, test_index, val_index = split_train_test_val_index(sims=sims_data, split=split, seed=seed)
    sims_index, sims1_index, sims2_index, sims3_index, simsb_index = split_by_magnitude_index(sims=sims_data)
    all_sims = {"sims": sims_index, "sims1": sims1_index, "sims2": sims2_index,
                "sims3": sims3_index, "simsB": simsb_index}
    for sim in all_sims.keys():
        train_sims_data = split_by_xy(sims_data[all_sims[sim] & train_index])
        test_sims_data = split_by_xy(sims_data[all_sims[sim] & test_index])
        val_sims_data = split_by_xy(sims_data[all_sims[sim] & val_index])
        all_sims[sim] = (train_sims_data[0], train_sims_data[1], val_sims_data[0],
                         val_sims_data[1], test_sims_data[0], test_sims_data[1])
    return all_sims


def txt_file_split_to_npy(input_path="../data/TRILEGAL_three_pix_triout_V1.txt",
                          output_path=None, seed=None, split=0.7):
    """
    Convert txt file to npy file.
    :param input_path:
    :param output_path:
    :param seed:
    :param split:
    :return:
    """
    if os.path.exists(input_path):
        if seed is not None:
            seed = int(seed)
        sims = read_trilegallsst(input_path)
        all_sims = prepare_data(sims, split=split, seed=seed)
    else:
        raise Exception("DATA path "+input_path+" does not exist.")
    for sim in all_sims.keys():
        x_train, y_train, x_val, y_val, x_test, y_test = all_sims[sim]
        if output_path is not None:
            with open(os.path.join(output_path, sim+".npy"), 'wb') as f:
                np.save(f, np.array(x_train))
                np.save(f, np.array(y_train))
                np.save(f, np.array(x_val))
                np.save(f, np.array(y_val))
                np.save(f, np.array(x_test))
                np.save(f, np.array(y_test))
    return all_sims


def read_npy(input_path, remove=True):
    with open(input_path, 'rb') as f:
        x_train = np.load(f)
        y_train = np.load(f)
        x_val = np.load(f)
        y_val = np.load(f)
        x_test = np.load(f)
        y_test = np.load(f)
    if remove:
        os.remove(input_path)

    return (x_train[0], x_train[1]), \
           tuple([y_train[:, i].reshape(-1, 1) for i in range(y_train.shape[-1])]), \
        (x_val[0], x_val[1]), \
             tuple([y_val[:, i].reshape(-1, 1) for i in range(y_val.shape[-1])]), \
           (x_test[0], x_test[1]), \
           tuple([y_test[:, i].reshape(-1, 1) for i in range(y_test.shape[-1])])

if __name__ == "__main__":
    data = txt_file_split_to_npy(output_path="../data/")