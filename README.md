# GPSig
A Gaussian process library for Bayesian learning from sequential data, such as time series, using signature kernels as covariance functions based on GPflow and TensorFlow. This repository contains supplementary code to the paper https://arxiv.org/abs/1906.08215.
***
## Installing
To get started, you should first clone the repository using git, e.g. with the command
```
git clone https://github.com/tgcsaba/GPSig.git
```
and then create and activate a virtual environment with Python 3.12
```
conda create -n env_name python=3.12
conda activate env_name
```
Then, install the requirements using pip by
```
pip install -r requirements.txt
```
The code is based on TensorFlow 2 and GPflow 2. On Linux (including WSL2), `tensorflow[and-cuda]` installs the CUDA 12 libraries needed to run computations on an NVIDIA GPU (which we heavily recommend, if you have one available), so no separate CUDA installation is required. You can check that the GPU is found by
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
To check the installation, run the tests from the root of the repository by
```
python -m pytest tests
```
The regression tests compare the outputs of this version against reference outputs recorded with the original TensorFlow 1.15 / GPflow 1.5 implementation (see `tests/reference/make_reference.py`).
***
## Getting started
To get started, we suggest to first look at the notebook `signature_kernel.ipynb`, which gives a simple worked out example of how to use the signature kernel as a standalone object. In this notebook, we validate the implementation of the signature kernel by comparing our results to an alternative way of computing signature features using the `esig` package. Note that this notebook is not part of the requirements: current versions of `esig` require `numpy>=2`, which is not compatible with GPflow 2.
The difference between the two ways of computing the signature kernel is a 'kernel trick', which makes it possible to compute the signature kernel using only inner product evaluation on the underlying state-space.

In the other notebook, `ts_classification.ipynb`, a worked out example is given on how to use signature kernels for time series classification using inter-domain sparse variational inference with inducing tensors to make computations tractable and efficient. To make the most of these examples, we also recommend to look into the [GPflow](https://github.com/GPflow/GPflow) syntax of defining kernels and GP models, a Gaussian process library that we build on.
***

## Changes in the TensorFlow 2 version
The code was ported from TensorFlow 1.15 / GPflow 1.5 to TensorFlow 2 / GPflow 2, keeping the interface of the original version (e.g. `kern.compute_K_symm`, `gpsig.models.SVGP(X, Y, kern=..., feat=...)`, `gpsig.training.optimize`). Apart from the port, the following changes were made to the original code:
- Fixed: with `InducingSequences` and `normalization=True` (the default), the cross-covariance between inducing sequences and inputs was normalized twice in `SignatureKernel.K_seq_n_seq_covs`.
- Disabled: the low-rank mode (`low_rank=True`) raises a `NotImplementedError`, since its implementation is known to be incorrect (the low-rank features of levels >= 2 are computed from the wrong factor, and the lengthscales/lags are ignored by `K`).
- Disabled: `SignatureSpectral` raises a `NotImplementedError`, since its base kernel only supports unbatched inputs, so most kernel computations fail.
- The `MultiClass` likelihood of GPflow 2 squashes the probabilities in `RobustMax` slightly differently than GPflow 1, which changes the corresponding ELBO and predictions by about 1e-4 (relative).

## Download datasets
The benchmarks directory contains the appropriate scripts used to run the benchmarking experiments in the paper. The datasets can be downloaded from our dropbox folder using the `download_data.sh` script in the `./benchmarks/datasets` folder by running
```
cd benchmarks
bash ./datasets/download_data.sh
```
or manually by copy-pasting the dropbox url containd within the aforementioned script.

## Support
We encourage the use of this code for applications, and we aim to provide support in as many cases as possible. For further assistance or to tell us about your project, please send an email to

`csaba.toth@maths.ox.ac.uk` or `harald.oberhauser@maths.ox.ac.uk`.
