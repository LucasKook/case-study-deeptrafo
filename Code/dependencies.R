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
remotes::install_github("neural-structured-additive-learning/deepregression@e3218777f94c7fda05b0e54ad90dae3a08daf184")
remotes::install_github("neural-structured-additive-learning/deeptrafo@83c4d8bbb5a7543f8a9ecc82e4c283883aa944f1")
