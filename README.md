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

The requirements are given in the `dependencies.R` file and can be installed by
sourcing the R script or `make dependencies`. If you set up a Python environment
for the first time, install `reticulate` and run the `check_and_install()`
function from the `deepregression` package. This tries to install miniconda, TF
2.10.0, TFP 0.16 and keras 2.8.0.

# Troubleshooting

See
[here](https://github.com/neural-structured-additive-learning/deepregression/blob/main/README.md#troubleshooting)
for troubleshooting your Python/R installation.

# Movies data

The data `movies.csv` is licensed under CC0 and available in raw form from
[kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

# How to cite deeptrafo

For methodology, please cite

```

```

When using the software, please cite

```

```

