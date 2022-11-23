# Case study `deeptrafo`

This repository contains code for reproducing the results in
[arXiv:...](https://arxiv.org/abs/...) using the R package `deeptrafo` with
stable version available on [CRAN](https://CRAN.R-project.org/package=deeptrafo)
and development version at
[GitHub](https://github.com/neural-structured-additive-learning/deeptrafo).

# Installation

To install `deeptrafo` run
```r
remotes::install_github("https://github.com/neural-structured-additive-learning/deeptrafo")
```
for the development version or
```r
install.packages("deeptrafo")
```
for the stable version from CRAN.

# Requirements

The requirements are given in the `DESCRIPTION`. If the package is loaded
manually using `devtools::load_all()`, make sure the following packages are
availabe:

  - `Matrix`
  - `dplyr`
  - `keras`
  - `mgcv`
  - `reticulate`
  - `tensorflow`
  - `tfprobability`
  - `deepregression`

If you set up a Python environment for the first time, install `reticulate` and
run the `check_and_install()` function from the `deepregression` package. This
tries to install miniconda, TF 2.10.0, TFP 0.16 and keras 2.8.0.

# Troubleshooting

See
[here](https://github.com/neural-structured-additive-learning/deepregression/blob/main/README.md#troubleshooting)
for troubleshooting your Python/R installation.

# How to cite deeptrafo

For methodology, please cite

```

```

When using the software, please cite

```

```

