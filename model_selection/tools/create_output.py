import pandas as pd
import tools
import tensorflow as tf
import numpy as np
from astropy.table import Table


def prepare_data(sims_data, split, seed=None):
    train_index, test_index, val_index = tools.create_simulated_data.split_train_test_val_index(sims=sims_data,
                                                                                                split=split, seed=seed)
    sims_index, sims1_index, sims2_index, sims3_index, simsb_index = tools.create_simulated_data.split_by_magnitude_index(
        sims=sims_data)
    all_sims = {"sims": sims_index, "sims1": sims1_index, "sims2": sims2_index,
                "sims3": sims3_index, "simsB": simsb_index}
    for sim in all_sims.keys():
        train_sims_data = tools.create_simulated_data.split_by_xy(sims_data[all_sims[sim]])
        all_sims[sim] = (train_sims_data[0], train_sims_data[1])
    return all_sims, train_index, test_index, val_index


def txt_file_split_to_npy(input_path="../data/TRILEGAL_three_pix_triout_RealErrors.txt",
                          output_path=None, seed=None, split=0.7):
    """
    Convert txt file to npy file.
    :param input_path:
    :param output_path:
    :param seed:
    :param split:
    :return:
    """
    if seed is not None:
        seed = int(seed)
    sims = tools.create_simulated_data.read_trilegallsst(input_path)
    all_sims, train_index, test_index, val_index = prepare_data(sims, split=split, seed=seed)
    return all_sims, train_index, test_index, val_index


def create_outputs(models_path, datapath, output_path):
    outputs = ["Mr", "Ar", "FeH"]
    data, index_train, index_val, index_test = txt_file_split_to_npy(datapath, seed=31)
    sims = tools.create_simulated_data.read_trilegallsst(datapath).to_pandas()
    sims["test_set"] = index_test * 1 + index_val * (-1)
    x_train, y_train = data["sims"]
    x = x_train
    model = tf.keras.models.load_model(models_path, compile=False)
    prediction = model(x)
    sigma_p = ([np.abs(np.array(prediction[i])).reshape((-1, 2))[:, -1:] for i in
                range(len(prediction))])  # cleaning shapes of the sigma outputs
    p = tuple([np.array(prediction[i]).reshape((-1, 2))[:, :1] for i in
               range(len(prediction))])  # cleaning shapes of the outputs
    for k in range(len(p)):
        sims["Model_" + outputs[k]] = p[k]
        sims["Model_" + outputs[k] + "Err"] = sigma_p[k]
    sims_astropy = Table.from_pandas(sims)
    sims_astropy.write(output_path, format='ascii', overwrite=True)


def split_by_xy_specific(d):
    """
    """
    x = d["rmag", "ugObs", "grObs", "riObs", "izObs"].to_pandas()
    y = d["Mr", "Ar", "FeH"].to_pandas()
    x_error = pd.DataFrame()
    x_error["rErr"] = d["rErr"].data
    for col_name in x.columns[1:]:
        error1 = d[col_name[0] + "Err"].data
        error2 = d[col_name[1] + "Err"].data
        x_error[col_name + "Err"] = np.sqrt(np.square(error1) + np.square(error2))
    return (np.array(x), np.array(x_error)), \
        (np.array(y["Mr"]).reshape(-1, 1), np.array(y["Ar"]).reshape(-1, 1), np.array(y["FeH"]).reshape(-1, 1))


def read_trilegallsst_specific(path='./TRILEGAL_three_pix_triout_V1.txt'):
    """
    Read TRILEGAL simulation augmented with LSST colors, see TRILEGAL_makeTestFile.ipynb.
    :param path: path to the file
    :return: simulation table
    """
    colnames = ["glon", "glat", "comp", "logg", "FeH", "Mr", "DM", "Ar", "rmagObs0", "ug0", "gr0", "ri0", "iz0", "rmag",
                "ugObs", "grObs", "riObs", "izObs", "uErr", "gErr", "rErr", "iErr", "zErr", "ugSL", "grSL", "riSL",
                "izSL", "ugErrSL", "grErrSL", "riErrSL", "izErrSL"]
    # comp: Galactic component the star belongs to: 1→thin disk; 2→thick disk; 3→halo; 4→bulge; 5→Magellanic Clouds.
    # Mr = rmag - Ar - DM - 5
    # rmag0, ug0...iz0: intrinsic values without dust extinction (but include photometric noise)
    # rmag, ug...iz: dust extinction included
    # uErr...zErr: photometric noise
    sims = Table.read(path, format='ascii', names=colnames)
    sims['gi0'] = sims['gr0'] + sims['ri0']
    sims['giObs'] = sims['grObs'] + sims['riObs']
    return sims


def recreate_csv(input_path, output_path):
    input_colnames = ['glon', "glat", "comp", "logg", "FeH", "Mr", "DM", "Ar", "rmagObs0", "ug0", "gr0", "ri0", "iz0",
                      "rmag", "ugObs", "grObs", "riObs", "izObs", "uErr", "gErr", "rErr", "iErr", "zErr", "ugSL", "grSL", "riSL",
                      "izSL", "ugErrSL", "grErrSL", "riErrSL", "izErrSL"]
    output_colnames = ["glon", "glat", "comp", "logg", "FeH", "Mr", "DM", "Ar", "rmagObs0", "ug0", "gr0", "ri0", "iz0",
                       "rmag", "ug", "gr", "ri", "iz", "uErr", "gErr", "rErr", "iErr", "zErr"]
    change_colnames = {"ugObs": "ug", "grObs": "gr", "riObs": "ri", "izObs": "iz"}
    sims = Table.read(input_path, format='ascii', names=input_colnames)
    for col in change_colnames.keys():
        sims.rename_column(col, change_colnames[col])

    sims = sims[output_colnames]
    sims.write(output_path, format='ascii', overwrite=True)
    x_train, y_train, x_val, y_val, _, _ = tools.create_simulated_data.txt_file_split_to_npy(output_path,
                                                                                             seed=31)["sims"]
    #print (x_train[0].shape, y_train[0].shape, x_val[0].shape, y_val[0].shape)



if __name__ == "__main__":
    models = ["../models/trained_chiTest4/photozannv14p_trained", "../models/trained_chiTest4/photozannv10p_trained",
              "../models/trained_chiTest4/photozannv24p_trained", "../models/trained_chiTest4/photozannv20p_trained"]
    outputs = ["../data/simCatalog_three_pix_triout_chiTest4_SimpleSingle.txt",
               "../data/simCatalog_three_pix_triout_chiTest4_SimpleMulti.txt",
               "../data/simCatalog_three_pix_triout_chiTest4_NaiveSingle.txt",
               "../data/simCatalog_three_pix_triout_chiTest4_NaiveMulti.txt"]
    for i in range(len(models)):
        print(i)
        create_outputs(models[i],
                       "../data/simCatalog_three_pix_triout_chiTest4.txt",
                       outputs[i])
    """create_outputs_specific("../models/trained/photozannv20p_trained",
                            "../data/simCatalog_three_pix_triout_chiTest4.txt",
                            "../data/simCatalog_three_pix_triout_chiTest4_NaiveMulti.txt")"""
    #recreate_csv("../data/simCatalog_three_pix_triout_chiTest4.txt", "../data/simplified_chiTest4.txt")
    #x_train, y_train, x_val, y_val, _, _ = tools.create_simulated_data.txt_file_split_to_npy("../data/simCatalog_three_pix_triout_chiTest4.txt")["sims"]
    #print (x_train[0].shape, y_train[0].shape, x_val[0].shape, y_val[0].shape)