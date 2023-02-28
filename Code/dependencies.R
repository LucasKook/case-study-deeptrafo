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
remotes::install_github("neural-structured-additive-learning/deepregression@78ea251c2d859d2c5cb5f0f2b22109aa7882d2ba")
remotes::install_github("neural-structured-additive-learning/deeptrafo@28cf0cfc725f6d7a0ba1c8ad237941b0e48c679d")