# Model selection

The study that selected the network used in the pipeline. Networks are trained on the TRILEGAL simulation of three
HEALPix pixels (`data/simCatalog_three_pix_triout_chiTest4.txt.gz`, unpack it with `gunzip -k` first) to predict
Mr, Ar and [Fe/H] from rmag, u-g, g-r, r-i, i-z and their errors.

Four types of networks are compared, simple or naive and single or multi output:
- simple: only the colors as inputs
- naive: the colors and their errors as inputs
- single output: a separate network for each parameter
- multi output: one network for all three parameters

Each of them predicts the value and its uncertainty, trained with the marginal posterior density loss.
The pipeline in `Environment/` uses a naive multi-output network.

- `tools/` - models, data splitting, hyperparameter search and plotting
- `tune/` - Hyperband search of the architecture for each model version (e.g. `tune/tune 10p 20p` submits the
  SLURM jobs); the best architectures are saved in `models/untrained/`
- `train/` - training of the selected architectures; the trained models are in `models/trained_chiTest4/`
- `Results/` - notebooks with the data, the single and multi output models, and their figures
- `USER/` - the four final models in the `.keras` format and a tutorial on how to use them

Model versions: `v1x` simple, `v2x` naive; `x=0` multi output, `x=1,2,3` single output for Mr, Ar, [Fe/H] and
`x=4` the three single output models merged into one; the `p` suffix marks models that predict the uncertainty.

Besides TensorFlow this part needs `tensorflow-probability`, `keras-tuner`, `astropy`, `astroML`, `corner` and `pydot`.
