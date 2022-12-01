# Case study `deeptrafo`

This repository contains code for reproducing the results in
[arXiv:...](https://arxiv.org/abs/...) using the R package `deeptrafo` with
stable version available on [CRAN](https://CRAN.R-project.org/package=deeptrafo)
and development version on
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

# Reproducing the results

1. Install dependencies via `make dependencies` or sourcing `dependencies.R`

2. Preprocess the `movies` data with `make data` or sourcing `movies.R`

3. Reproduce the results via `make repro` or sourcing `code.R`

Simply run `make all` for all of the above.

# Troubleshooting

See
[here](https://github.com/neural-structured-additive-learning/deepregression/blob/main/README.md#troubleshooting)
for troubleshooting your Python/R installation.

# Movies data

The data `movies.csv` is licensed under CC0 and available in raw form from
[kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

# How to cite `deeptrafo`

For methodology please cite an appropriate selection of:

1. Deep conditional transformation models

```
@inproceedings{sick2020deep,
doi = {10.1109/icpr48806.2021.9413177},
year = {2021},
publisher = {{IEEE}},
author = {Beate Sick and Torsten Hothorn and Oliver D\"urr},
title = {Deep transformation models: Tackling complex regression problems with
neural network based transformation models},
booktitle = {2020 25th International Conference on Pattern Recognition ({ICPR})}
}
```

```
@InProceedings{baumann2020deep,
  doi = {10.1007/978-3-030-86523-8\_1},
  year = {2021},
  publisher = {Springer International Publishing},
  pages = {3--18},
  author = {Philipp F. M. Baumann and Torsten Hothorn and David R\"{u}gamer},
  title = {Deep Conditional Transformation Models},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track}
}
```

2. Probabilistic time series forecasts with autoregressive transformation models

```
@article{rugamer2021timeseries,
  doi = {10.48550/arXiv.2110.08248},
  year = {2021},
  journal = {arXiv preprint},
  note = {To appear in \emph{Statistics \& Computing}},
  author = {David R\"ugamer and Philipp FM Baumann and Thomas Kneib and Torsten Hothorn},
  title = {Probabilistic Time Series Forecasts with Autoregressive Transformation Models}
}
```

3. Ordinal neural network transformation models

```
@article{kook2020ordinal,
  doi = {10.1016/j.patcog.2021.108263},
  year = {2022},
  publisher = {Elsevier {BV}},
  volume = {122},
  pages = {108263},
  author = {Lucas Kook and Lisa Herzog and Torsten Hothorn and Oliver D\"{u}rr and Beate Sick},
  title = {Deep and interpretable regression models for ordinal outcomes},
  journal = {Pattern Recognition}
}
```

4. Transformation ensembles

```
@article{kook2022deep,
  title={Deep interpretable ensembles},
  author={Kook, Lucas and G{\"o}tschi, Andrea and Baumann, Philipp FM and Hothorn, Torsten and Sick, Beate},
  journal={arXiv preprint arXiv:2205.12729},
  year={{2022}},
  doi={10.48550/arXiv.2205.12729}
}
```

When using the software, please cite

```
@article{kook2022estimating,
  title={Estimating Conditional Distributions with Neural Networks using R package deeptrafo},
  author={Kook, Lucas and Baumann, Philipp FM and D{\"u}rr, Oliver and Sick, Beate and R{\"u}gamer, David},
  journal={arXiv preprint arXiv:2211.13665},
  year={2022},
  doi={10.48550/arXiv.2211.13665}
}
```

