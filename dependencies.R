# Install dependencies

packages <- c("Matrix", "boot", "data.table", "data.table", "deepregression",
              "deeptrafo", "dplyr", "ggjoy", "ggplot2", "ggrepel", "ggsci",
              "jsonlite", "keras", "knitr", "lubridate", "mgcv", "moments",
              "patchwork", "reticulate", "safareg", "tensorflow",
              "tfprobability", "tidytext", "tidyverse", "tm", "tram", "remotes")

new_pkg <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_pkg)) install.packages(new_pkg)

if (!require(tsdl))
  remotes::install_github("FinYang/tsdl")

