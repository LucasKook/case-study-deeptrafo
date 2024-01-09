# Install dependencies

packages <- c("Matrix", "boot", "data.table", "data.table", "dplyr", "ggridges",
              "ggplot2", "ggrepel", "ggsci", "jsonlite", "keras", "knitr",
              "lubridate", "mgcv", "moments", "patchwork", "reticulate",
              "tensorflow", "tfprobability", "tidytext", "tidyverse", "tm",
              "tram", "remotes", "ordinal", "ggpubr")

new_pkg <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_pkg)) install.packages(new_pkg)

if (!require(tsdl))
  remotes::install_github("FinYang/tsdl")

remotes::install_github("neural-structured-additive-learning/safareg")
remotes::install_github("neural-structured-additive-learning/deepregression@92166e218f54e3ab30c2fee3a8cab1909140d30e", upgrade = "never")
remotes::install_github("neural-structured-additive-learning/deeptrafo@93afb9117592ef3d8ceb4afe807ff7207aa9f2f4", upgrade = "never")

library("deeptrafo")
reticulate::py_install(packages = "gensim")
