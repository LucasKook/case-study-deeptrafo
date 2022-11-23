# Install dependencies

packages <- c("Matrix", "boot", "data.table", "data.table", "deepregression",
              "deeptrafo", "dplyr", "ggjoy", "ggplot2", "ggrepel", "ggsci",
              "jsonlite", "keras", "knitr", "lubridate", "mgcv", "moments",
              "patchwork", "reticulate", "safareg", "tensorflow",
              "tfprobability", "tidytext", "tidyverse", "tm", "tram", "tsdl")

new_pkg <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_pkg)) install.packages(new_pkg)

