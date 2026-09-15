# PhotoD NN: Stellar Parameters Using Photometry and a Neural Network

## About

This branch holds the neural network version of PhotoD, used for Mrakovčić, Ivezić & Palaversa (2025), AJ 170, 72.
A small network is trained on the Bayesian PhotoD estimates of Mr, Ar and [Fe/H] for a few thousand stars and then
predicts the three parameters and their uncertainties for the rest of the catalog, orders of magnitude faster.

Layout:
- `Environment/` - the `photod` package (network, training, QA plots and the pipeline used for the paper)
- `PhotoD.keras` - trained network
- `scripts/` - training set size and timing tests, plots of the pipeline outputs
- `notebooks/` - comparison of the network and Bayesian uncertainties
- `figures/` - QA figures for the simulated (SDSS patch RA 340-350) and real (SEGUE l=110) catalogs
- `data/` - input catalogs (gzipped)
- `model_selection/` - the earlier study that selected the network type and architecture

## Installation
### 1. Clone the repository

```bash
git clone -b PhotoD_NN https://github.com/labastro-rbi/photoD.git
cd photoD/Environment
```

### 2. Create conda environment
It is recommended to install PhotoD in a conda environment. To install conda, please follow the instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). If you don't want to use conda, you can
skip this step.

```bash
conda create -n photod python=3.10 -y
conda activate photod
```

### 3. Install photod
```bash
pip install .
cd ..
```

### 4. Unpack the data
```bash
gunzip -k data/*.gz
```

## Data

- `BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt` - simulated LSST catalog (TRILEGAL, SDSS Stripe 82 patch
  RA 340-350) with the Bayesian estimates (`MrEst`, `ArEst`, `FeHEst`), their uncertainties (`MrUnc`, `ArUnc`,
  `FeHUnc`) and the true values (`MrTrue`, `ArTrue`, `FeHTrue`)
- `BayesMethod3D_SEGUEpatch-l110-KarloTest-short1.txt` - SDSS stars in the SEGUE l=110 field with their Bayesian
  estimates

Any catalog with the same columns can be used: magnitudes `umag`...`zmag` (or `rmag` and the colors `ug`, `gr`,
`ri`, `iz`), their errors `uerr`...`zerr` (or `uErr`...`zErr`), and the estimates with their uncertainties.

## Usage
### Train and test a model
```python
import photod

photod_model = photod.PhotoD(batch_size=1024)
photod_model.import_csv("./data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt", reduce=10000, test_split=0.2)
photod_model.create_model()
photod_model.train_model(epochs=1024, iterations=2, decay_epochs=10, decay_rate=0.7)
photod_model.train_error_model(epochs=256, iterations=1)

x, y, y_error, p, sigma_p, bayes_sigma = photod_model.test_model()
print(photod_model.metrics)
```
`example.py` does the same, with the QA plots and saving of the models.

### Load a trained model
```python
photod_model.load_model("./PhotoD.keras")
photod_model.model.summary()
```

### Predict stellar parameters
The inputs are rmag, u-g, g-r, r-i, i-z and their errors, both with shape (n, 5):
```python
p_predicted, sigma_p_predicted, bayes_sigma_predicted = photod_model.predict((x, x_error))
```
`p_predicted` holds Mr, Ar and [Fe/H]. `sigma_p_predicted` is the uncertainty of the network and
`bayes_sigma_predicted` the uncertainty of the Bayesian estimates predicted by the error network; the total
uncertainty is their sum in quadrature.

### Run the pipeline
The pipeline trains on `--train_size` random stars of a catalog, predicts the rest, writes the catalog with the
`MrNN`, `ArNN`, `FeHNN` columns and their uncertainties, and makes the QA plots:
```bash
cd Environment/photod
python pipeline.py --input_training_path ../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt \
                   --input_prediction_path ../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest.txt \
                   --output_prediction_path ../../data/BayesMethod3D_SDSSpatchRA340-350_KarloTest_NN.txt \
                   --train_size 10000 --plot_path ../../outputs/simulation/
```

### Save the model
```python
photod_model.save_model("./PhotoD.keras")
```
