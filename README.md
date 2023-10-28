# Canonical Manifold Flows README

This is the code we used for our paper, [Canonical normalizing flows for manifold learning]
(https://doi.org/10.48550/arXiv.2310.12743) (NeurIPS 2023).  Our code builds directly off   from the [RNF codebase][Caterini et al.](https://doi.org/10.48550/arXiv.2310.12743) (NeurIPS 2021), which we cite in our manuscript. Caterini et al.'s code did not have a license, and given its public availability, we used it freely, dependencies do have licenses, all of which do allow us to use the respective code.

The up-to-date code can be found at: https://github.com/k-flouris/cmf

## Setup

First, install submodules:

    $ git submodule init
    $ git submodule update

Next, install dependencies. If you use `conda`, the following will create an environment called `canonical`:

    conda env create -f environment-lock.yml

Activate this with

    conda activate canonical

before running any code or tests.

If you don't use `conda`, then please see `environment.yml` for a list of required packages, which will need to be installed manually e.g. via `pip`.

### Obtaining datasets

Our code runs on several types of datasets, including synthetic 2-D data, tabular data, and image data. The 2-D datasets are automatically generated, and the image datasets are downloaded automatically. However the tabular datasets will need to be manually downloaded from [this location](https://zenodo.org/record/1161203). The following should do the trick:

    mkdir -p data/ && wget -O - https://zenodo.org/record/1161203/files/data.tar.gz | tar --strip-components=1 -C data/ -xvzf - data/{gas,hepmass,miniboone,power}

This will download the data to `data/`. If `data/` is in the same directory as `main.py`, then everything will work out-of-the-box by default. If not, the location to the data directory will need to be specified as an argument to `main.py` (see `--help`).



To train our model on the sphere dataset, run:

    ./main.py --model non-square --g_ij_loss True --dataset 3d-circle 
 
Note that this will actually launch a grid of runs over various values for the `regularization_param`, `likelihood_warmup`, and `lr` config values as described in Appendix F.1. To overrirde the grid search and just launch a specific configuration, please edit the `non-square` section of the file `config/two_d.py` by removing the `GridParams` specification, `g_ij_loss` includes the regularization which activates the canonical manifold learning flow. 

If you are on a GPU device, please specify `CUDA_VISIBLE_DEVICES=`, i.e. to the empty string, as this experiment is not currently supported to run on the GPU.

To launch a baseline two-step procedure run, add the flag `--baseline` to the command above.

To visualize the result of an experiment, either use tensorboard (described below), or locate the directory in which information about the run is stored (e.g. `<run>=runs/MthDD_HH-MM-SS`) and use the command

    ./main.py --resume <run> --test

This will produce the plots:
1. `density.png` showing the density on the manifold
2. `distances.png` showing the "speed" at which the manifold is parametrized
3. `pullback.png` showing the pullback density required to be learned.

## Tabular Data Experiments

To train a tabular model, run:

    CUDA_VISIBLE_DEVICES=0 ./main.py --model non-square --g_ij_loss True --dataset <dataset-name>

where `<dataset-name>` is one of `power`, `gas`, `hepmass`, or `miniboone`.
To evaluate a single completed run, locate its run directory -- say it is `<run>` -- and run the command

    CUDA_VISIBLE_DEVICES=0 ./main.py --resume <run> --test



## Image Experiments

To train an image model, run:

    CUDA_VISIBLE_DEVICES=<device(s)> ./main.py --model non-square --g_ij_loss True --dataset <dataset-name>

where `<devices(s)>` is a string specifying one or more `CUDA` devices, e.g. `<devices(s)>=2,3,4`, and `<dataset-name>` is either `mnist`, `fashion-mnist`, `svhn`, or `cifar10` (although we only include results from MNIST and Fashion-MNIST in the manuscript).

Again, an RNFs-TS method can be launched by appending the flag `--baseline`.

A variety of parameters were modified as noted in Appendix F.3; to launch a particular run matching what is noted in the paper, you can either check the list of runs in `image_runs.sh`, or modify the hyperparameters directly in the `config/images.py` file under the heading `non-square`. Switching between RNFs-ML (exact) and RNFs-ML ($K=$`<k>`) is the same as the previous section, although increasing `K` too much will greatly increase memory requirements.

## Miscellaneous - Run Directory and Tensorboard Logging

By default, running `./main.py` will create a directory inside `runs/` that contains

- Configuration info for the run
- Version control info for the point at which the run was started
- Checkpoints created by the run

This allows easily resuming a previous run via the `--resume` flag, which takes as argument the directory of the run that should be resumed.
To avoid creating this directory, use the `--nosave` flag.

The `runs/` directory also contain Tensorboard logs giving various information about the training run, including 2-D density plots in this case. To inspect this run the following command in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.
For 2D datasets, the "Images" tab shows the learned density, and for image datasets, the "Images" tab shows samples from the model over training.
The "Text" tab also shows the config used to produce each run.

# Major Differences Versus Non-Square Codebase

Besides the differences listed in the first section, there are some major code changes which allow canonical manifold flows to function in this codebase.

## CMF Density

The main workhorse of the codebase is the `NonSquareHeadDensity` class (in `cif/models/components/densities/non_square.py`), which allows for specification of canonical manifold flows. This class acts as a `Density` object, and specifies the head of the canonical manifold learning flows

# Bibtex

