# Install dependencies
# TODO: Update GitHub commit numbers

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
remotes::install_github("neural-structured-additive-learning/deepregression@a8de92a7374a88fc5ed859238ce45f56813743e8")
remotes::install_github("neural-structured-additive-learning/deeptrafo@463145a67f467f4975907695fa0905374a0f282d")
