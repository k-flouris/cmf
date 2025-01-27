# Canonical Manifold Flows README

This is the code we used for our paper, [Canonical normalizing flows for manifold learning](https://doi.org/10.48550/arXiv.2310.12743) (NeurIPS 2023). Our code builds directly off from the [RNF codebase][Caterini et al.](https://doi.org/10.48550/arXiv.2310.12743) (NeurIPS 2021), which we cite in our manuscript. Caterini et al.'s code did not have a license, and given its public availability, we used it freely. Dependencies do have licenses, all of which allow us to use the respective code.

The up-to-date code can be found at: https://github.com/k-flouris/cmf

## Setup

First, install submodules:

    $ git submodule init
    $ git submodule update

Next, install dependencies. If you use `conda`, the following will create an environment called `canonical`:

    conda env create -f environment-lock.yml

Activate this with:

    conda activate canonical

before running any code or tests.

If you don't use `conda`, then please see `environment.yml` for a list of required packages, which will need to be installed manually e.g. via `pip`.

## Obtaining Datasets

Our code runs on several types of datasets, including synthetic 2-D data, tabular data, and image data. The 2-D datasets are automatically generated, and the image datasets are downloaded automatically. However, the tabular datasets will need to be manually downloaded from [this location](https://zenodo.org/record/1161203). The following should do the trick:

    mkdir -p data/ && wget -O - https://zenodo.org/record/1161203/files/data.tar.gz | tar --strip-components=1 -C data/ -xvzf - data/{gas,hepmass,miniboone,power}

This will download the data to `data/`. If `data/` is in the same directory as `main.py`, then everything will work out-of-the-box by default. Otherwise, specify the data directory location as an argument to `main.py` (see `--help`).

## Simulated Datasets

Our code supports a variety of simulated datasets, including spheres and fuzzy line manifolds embedded in higher dimensions.

To train our model on the sphere dataset, run:

    ./main.py --model non-square --dataset sphere --config g_ij_loss=True --config lr=0.001 --config latent_dimension=3 --config log_jacobian_method=cholesky

To train our model on the fuzzy line dataset, run:

    ./main.py --model non-square --dataset fuzzy-line --config g_ij_loss=True --config lr=0.0005 --config latent_dimension=4 --config log_jacobian_method=cholesky
    
Note that this will actually launch a grid of runs over various values for the `regularization_param`, `likelihood_warmup`, and `lr` config values as described in Appendix F.1. To override the grid search and just launch a specific configuration, please edit the `non-square` section of the file `config/two_d.py` by removing the `GridParams` specification. `--config g_ij_loss=True` is used to overide the config parameters includes and 'g_ij_loss' includes the regularization which activates the canonical manifold learning flow.


To launch a baseline two-step procedure run, add the flag `--baseline` to the command above.

### 6D Sinusoid 1D embedded ion 6D ataset Example

To train our model on a randomized 6D sphere dataset, run:

    CCUDA_VISIBLE_DEVICES=<device(s)> ./main.py --model non-square --dataset sinusoid-1-6 \
    --config max_epochs=1000 --config log_jacobian_method=cholesky --config hutchinson_samples=1 \
    --config g_ij_loss=True --config centering_loss=False --config g_ij_global_loss=True \
    --config centering_regularization_param=0 --config metric_regularization_param=1 \
    --config elbo_regularization_param=1 --config regularization_param=1 \
    --config likelihood_warmup=False --config latent_dimension=6 --config lr=0.0007 \
    --config use_fid=True --config epochs_per_test=15 --config early_stopping=False \
    --num-seeds 5 --logdir-root runs/simulated/global_topo_6d_multi_embedings_properplot

    
Where `<devices(s)>` is a string specifying one or more `CUDA` devices, e.g., `2,3,4`.

These configurations provide fine-grained control over the training process, ensuring optimal learning performance. 

## Tabular Data Experiments

To train a tabular model, run:

    CUDA_VISIBLE_DEVICES=<device(s)> ./main.py --model non-square --config g_ij_loss=True --dataset <dataset-name>

where `<dataset-name>` is one of `power`, `gas`, `hepmass`, or `miniboone`.

Suggested parameter example for `power` dataset:

    ./main.py --model non-square --dataset power --config g_ij_loss=True --config lr=0.001 --config latent_dimension=10

## Image Experiments

To train an image model, run:

    CUDA_VISIBLE_DEVICES=<device(s)> ./main.py --model non-square  --dataset <dataset-name> --config g_ij_loss=True

where `<devices(s)>` is a string specifying one or more `CUDA` devices, e.g., `2,3,4`, and `<dataset-name>` is either `mnist`, `fashion-mnist`, `svhn`, or `cifar10`.

Suggested parameter example for `mnist` dataset:

    ./main.py --model non-square --dataset mnist --config g_ij_loss=True --config lr=0.0003 --config latent_dimension=64

## Testing and Experimentation

To visualize the result of an experiment, either use tensorboard (described below), or locate the directory in which information about the run is stored (e.g. `<run>=runs/MthDD_HH-MM-SS`) and use the command:

    ./main.py --resume <run> --test

This will produce the plots:

1. `density.png` showing the density on the manifold
2. `distances.png` showing the "speed" at which the manifold is parametrized
3. `pullback.png` showing the pullback density required to be learned.

To further experiment with simulated data, use `visualizer.py` (remember to update `experiment.py` accordingly). Additionally, the `test_metric` option in the config can be used for experimentation on the trained model. See `visualizer.py` for details.

More geometric and topological experimentation can be implemented easily in the `non-square.py` density module, as the metric and other important geometric objects are already calculated there.
    
    
    
## Miscellaneous - Run Directory and Tensorboard Logging

By default, running `./main.py` will create a directory inside `runs/` that contains:

- Configuration info for the run
- Version control info for the point at which the run was started
- Checkpoints created by the run

To inspect results using Tensorboard, run:

    tensorboard --logdir runs/ --port=8008

Then navigate to http://localhost:8008 to view logs.

## Major Differences Versus Non-Square Codebase

Besides the differences listed in the first section, there are major code changes allowing canonical manifold flows to function in this codebase.

## CMF Density

The main workhorse of the codebase is the `NonSquareHeadDensity` class (in `cif/models/components/densities/non_square.py`), which allows for specification of canonical manifold flows. This class acts as a `Density` object, and specifies the head of the canonical manifold learning flows.

## BibTeX Citation

If you use this code in your research, please cite our work:

    @article{flouris2023canonical,
      title={Canonical Normalizing Flows for Manifold Learning},
      author={Kyriakos Flouris and others},
      journal={NeurIPS},
      year={2023},
      doi={10.48550/arXiv.2310.12743}
    }

# Canonical Manifold Learning Flow (CMF)

    
## Summary
Canonical Manifold Flows (CMF) are a class of generative modeling techniques that assume a low-dimensional manifold description of the data. The embedding of such a manifold into the high-dimensional space of the data is achieved via learnable invertible transformations. Once the manifold is properly aligned via a reconstruction loss, the probability density is tractable on the manifold, and maximum likelihood can be used to optimize the network parameters. Naturally, the lower-dimensional representation of the data requires an injective mapping. Recent approaches have enforced that the density aligns with the modeled manifold while efficiently calculating the density volume-change term when embedding into the higher-dimensional space. However, unless the injective mapping is analytically predefined, the learned manifold is not necessarily an efficient representation of the data. The latent dimensions of such models frequently learn an entangled intrinsic basis, with degenerate information being stored in each dimension. Alternatively, if a locally orthogonal and/or sparse basis is learned, termed a canonical intrinsic basis, it can serve in learning a more compact latent space representation. Toward this end, CMF proposes a method where a novel optimization objective enforces the transformation matrix to have few prominent and non-degenerate basis functions. By minimizing the off-diagonal manifold metric elements' l1-norm, such a basis is achieved, which is simultaneously sparse and/or orthogonal. Canonical manifold flow yields a more efficient use of the latent space, automatically generating fewer prominent and distinct dimensions to represent data, and consequently a better approximation of target distributions than other manifold flow methods in most experiments conducted, resulting in lower FID scores.
    

## Method

### Canonical Intrinsic Basis
We demonstrate the performance of CMF using synthetic data generated on a fuzzy line with noise in the perpendicular and parallel directions. A comparison of density plots for fuzzy lines learned using RNF and CMF is shown below:

![Density plot for a fuzzy line with RNF](figures/linernf.pdf)
**(a) Density plot for a fuzzy line learned with RNF**

![Density plot for a fuzzy line with CMF](figures/linecan.pdf)
**(b) Density plot for a fuzzy line learned with CMF**

CMF results in non-degenerate latent variables, which better capture the structure of the underlying manifold.

### Canonical Manifold Learning

A canonical manifold, \(\mathfrak{M}\), is defined as having an orthogonal and/or sparse basis \(\mathbf{e}_i\) such that \(\mathbf{e}_i \cdot \mathbf{e}_j = 0\) for all \(\mathbf{y} \in \mathfrak{M}\) whenever \(i \neq j\).

To enforce this property, the off-diagonal elements of the metric tensor are minimized:

\[
    \| G_{i\neq j} \|^1_1 \triangleq \sum_{i}\sum_{j\neq i}\|G_{ij}\|^1_1
\]

This objective promotes sparsity and orthogonality in the learned representations, leading to improved manifold learning.

## Experiments

### Simulated Data

CMF was evaluated on two simulated datasets:
- **2D Sphere in \(\mathbb{R}^3\)**
- **Moebius Band in \(\mathbb{R}^3\)**

Comparison results for RNF and CMF methods:

![Hollow sphere learned with RNF](figures/spherernf.pdf)
**(a) Hollow sphere learned with RNF**

![Hollow sphere learned with CMF](figures/spherecan.pdf)
**(b) Hollow sphere learned with CMF**

![Moebius band learned with RNF](figures/moebiusrnf.pdf)
**(c) Moebius band learned with RNF**

![Moebius band learned with CMF](figures/moebiuscan.pdf)
**(d) Moebius band learned with CMF**

Results indicate that CMF achieves a better separation of latent components and improved learning of complex topological structures.

### Image Data

CMF was also evaluated on popular image datasets: MNIST, Fashion-MNIST, Omniglot, CIFAR-10, SVHN, and CelebA. The results demonstrate superior performance in terms of Frechet Inception Distance (FID) compared to RNF and MFlow, especially for lower-dimensional latent spaces.

#### Sparsity and Orthogonality

The sparsity and orthogonalization encouraged by CMF are demonstrated by analyzing the metric tensor:

![Fashion-MNIST RNF](figures/fm10_rnf_plot_g_combined.pdf)
**(a) Fashion-MNIST trained with RNF (MACS=0.03)**

![Fashion-MNIST CMF](figures/fm10_cmf_plot_g_combined.pdf)
**(b) Fashion-MNIST trained with CMF (MACS=0.02)**

![Omniglot RNF](figures/om10_rnf_plot_g_combined.pdf)
**(c) Omniglot trained with RNF (MACS=0.04)**

![Omniglot CMF](figures/om10_cmf_plot_g_combined.pdf)
**(d) Omniglot trained with CMF (MACS=0.03)**

CMF achieves lower Mean Absolute Cosine Similarity (MACS) values, indicating better orthogonalization and improved representation.
