## ----include=FALSE------------------------------------------------------------
library("knitr")
opts_chunk$set(engine = "R", tidy = FALSE, prompt = TRUE, cache = FALSE,
               fig.width = 6.5 * 0.8, fig.height = 4.5 * 0.8, fig.fullwidth = TRUE,
               fig.path = "Figures/", fig.ext = c("pdf", "eps", "jpg", "tiff"),
               dpi = 600)
knitr::render_sweave()  # use Sweave environments
knitr::set_header(highlight = "")  # do not \usepackage{Sweave}
## R settings
options(prompt = "R> ", continue = "+  ", width = 76, useFancyQuotes = FALSE,
        digits = 3L)


## ----preliminaries, echo = FALSE, results = "hide", message=FALSE-------------
library("deeptrafo")
library("ggplot2")
# devtools::install_github("FinYang/tsdl") # for tsdl data set
library("tsdl")
library("reticulate")
library("safareg")
library("data.table")
library("ggplot2")
library("patchwork")
library("ggjoy")
library("moments")
library("lubridate")
library("boot")
library("ggrepel")
library("tram")

theme_set(theme_bw() + theme(legend.position = "top"))

# Params ------------------------------------------------------------------

bpath <- "."
nr_words <- 10000
embedding_size <- 100
maxlen <- 100
order_bsp <- 25
repetitions <- 2

# Loading the data --------------------------------------------------------
if (!file.exists("data_splitted.RDS"))
  source("movies.R")

ATMonly <- FALSE


## ----data, eval=!ATMonly------------------------------------------------------
data_list <- readRDS(file.path(bpath, "data_splitted.RDS"))[[1]]
train <- data_list[[1]]
test <- data_list[[2]]

tokenizer <- text_tokenizer(num_words = nr_words)
mov <- readRDS(file.path(bpath, "mov_ov.RDS"))
tokenizer |> fit_text_tokenizer(mov)
words <- readRDS(file.path(bpath, "words.RDS"))


## ----formula-interface, eval=!ATMonly-----------------------------------------
fm <- vote_count | genreAction ~ 0 + s(budget, df = 6) + popularity


## ----setting-up-dctms, eval=!ATMonly------------------------------------------
opt <- optimizer_adam(learning_rate = 0.1, decay = 4e-4)
(m_fm <- cotramNN(formula = fm, data = train, optimizer = opt))


## ----fitting-dctms, eval=!ATMonly---------------------------------------------
hist_m_fm <- fit(m_fm, epochs = 1e3, validation_split = 0.1, batch_size = 64,
  verbose = FALSE)
unlist(coef(m_fm, which = "shifting"))


## ----plot-hist, fig.width=8.1, fig.height=4.05, eval=!ATMonly-----------------
p1 <- data.frame(x = 1:hist_m_fm$params$epochs, 
                 training = hist_m_fm$metrics$loss,
                 validation = hist_m_fm$metrics$val_loss) |> 
  tidyr::gather("set", "loss", training, validation) |> 
  ggplot(aes(x = x, y = loss, color = set)) +
  geom_line() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "epochs", color = "")
  
nd <- list(genreAction = c(0, 1), budget = rep(mean(train$budget), 2), 
           popularity = rep(mean(train$popularity), 2))
preds <- predict(m_fm, newdata = nd, q = seq(0, 2000, 25), type = "trafo")
pdat <- data.frame(y = as.numeric(names(preds)),
                   ga0 = unlist(lapply(preds, \(x) x[1, 1])),
                   ga1 = unlist(lapply(preds, \(x) x[2, 1])))
p2 <- pdat |> tidyr::gather("action", "trafo", ga0, ga1) |> 
  ggplot(aes(x = y, y = trafo, color = action)) +
  geom_step() +
  labs(x = "vote count", y = "transformation", color = "genreAction") +
  scale_color_manual(values = colorspace::diverging_hcl(2),
                     labels = c("ga0" = 0, "ga1" = 1))

ggpubr::ggarrange(
  p1 + labs(tag = "A"), p2 + labs(tag = "B"), nrow = 1,
  common.legend = FALSE
)


## ----working-with-neural-networks, eval=!ATMonly------------------------------
embd_mod <- function(x) x |>
  layer_embedding(input_dim = nr_words, output_dim = embedding_size) |>
  layer_lstm(units = 50, return_sequences = TRUE) |>
  layer_lstm(units = 50, return_sequences = FALSE) |>
  layer_dropout(rate = 0.1) |>
  layer_dense(25) |> 
  layer_dropout(rate = 0.2) |> 
  layer_dense(5) |> 
  layer_dropout(rate = 0.3) |> 
  layer_dense(1)

fm_deep <- update(fm, . ~ . + deep(texts))
m_deep <- deeptrafo(fm_deep, data = train,
  list_of_deep_models = list(deep = embd_mod))

dhist <- fit(m_deep, epochs = 50, validation_split = 0.1, batch_size = 32,
  callbacks = list(callback_early_stopping(patience = 5)), verbose = FALSE)


## ----ensembling-dctms, eval=!ATMonly------------------------------------------
ens_deep <- ensemble(m_deep, n_ensemble = 3, epochs = 50, batch_size = 64,
  verbose = FALSE)


## ----ensembling-dctms-methods, eval=!ATMonly----------------------------------
logLik(ens_deep, newdata = test, convert_fun = mean)


## ----ensembling-dctms-plot-prereq, eval=!ATMonly------------------------------
d <- deeptrafo:::.call_for_all_members(
  ens_deep, deeptrafo:::plot.deeptrafo, only_data = TRUE)
pd <- data.frame(cbind(
  value = d[[1]][[1]]$value, do.call("cbind", lapply(
    d, \(x) x[[1]]$partial_effect))))
ord <- order(pd$value)
pdat <- pd[ord,]
nd <- data.frame(
  value = pdat$value, ens = apply(pdat[, -1], 1, mean),
  sd = apply(pdat[, -1], 1, sd))


## ----ensembling-dctms-plot, eval=!ATMonly-------------------------------------
plot(pdat$value, pdat$V2, type = "l", ylim = range(pdat[, -1]),
     xlab = "log(1 + bugdet)", ylab = "partial effect", col = "gray80", lty = 2)
matlines(pdat$value, pdat[, -1:-2], col = "gray80", lty = 2)
rug(train$budget, col = rgb(.1, .1, .1, .3))

lines(nd$value, nd$ens)
polygon(c(nd$value, rev(nd$value)),
        c(nd$ens + 2 * nd$sd, rev(nd$ens - 2 * nd$sd)),
        col = rgb(.1, .1, .1, .1), border = FALSE)

legend("topright", legend = c("ensemble", "individual"), lty = c(1, 2),
       col = c(1, "gray80"), lwd = 1, bty = "n")


## ----cross-validating-dctms, fig.width=9, fig.height=4.05, eval=!ATMonly------
cv_deep <- cv(m_deep, epochs = 50, cv_folds = 5, batch_size = 64)
plot_cv(cv_deep)


## ----preproc-ontram-----------------------------------------------------------
train$action <- ordered(train$genreAction)
test$action <- ordered(test$genreAction, levels = levels(train$action))


## ----embd_mod-----------------------------------------------------------------
make_keras_model <- function() {
  return(
    keras_model_sequential(name = "embd")  |> 
      layer_embedding(input_dim = nr_words, output_dim = embedding_size) |>
      layer_lstm(units = 50, return_sequences = TRUE)  |> 
      layer_lstm(units = 50, return_sequences = FALSE)  |> 
      layer_dropout(rate = 0.1) |>
      layer_dense(25)  |>  
      layer_dropout(rate = 0.2)  |> 
      layer_dense(5, name = "penultimate")  |> 
      layer_dropout(rate = 0.3) |> 
      layer_dense(1)
  )
}


## ----formulas-ontram----------------------------------------------------------
fm_0 <- action ~ 1
fm_tab <- action ~ 0 + popularity
fm_text <- action ~ 0 + deep(texts)
fm_semi <- action ~ 0 + popularity + deep(texts)


## ----mods-ontram--------------------------------------------------------------
### unconditional ####
m_0 <- PolrNN(fm_0, data = train, optimizer = optimizer_adam(
  learning_rate = 1e-2, decay = 1e-4), weight_options = weight_control(
  general_weight_options = list(trainable = FALSE, use_bias = FALSE),
  warmstart_weights = list(list(), list(), list("1" = 0))))
fit(m_0, epochs = 3e3, validation_split = 0, batch_size = length(
  train$action), verbose = FALSE)

### only tabular ####
m_tab <- PolrNN(fm_tab, data = train, optimizer = optimizer_adam(
  learning_rate = 0.1, decay = 1e-4))
fit(m_tab, epochs = 1e3, batch_size = length(train$action),
  validation_split = 0, verbose = FALSE)
exp(-unlist(coef(m_tab, which = "shifting")))

### only text ####
embd <- make_keras_model()
m_text <- PolrNN(fm_text, data = train, list_of_deep_models = list(
  deep = embd), optimizer = optimizer_adam(learning_rate = 1e-4))
fit(m_text, epochs = 10, callbacks = list(callback_early_stopping(
  patience = 2, restore_best_weights = TRUE)), verbose = FALSE)

### semi-structured ####
embd_semi <- make_keras_model()

optimizer <- function(model) {
  optimizers_and_layers <- list(
    tuple(optimizer_adam(learning_rate = 1e-2),
    get_layer(model, "ia_1__2")),
    tuple(optimizer_adam(learning_rate = 1e-2),
    get_layer(model, "popularity_3")), 
    tuple(optimizer_adam(learning_rate = 1e-4), 
    get_layer(model, "embd")))
  multioptimizer(optimizers_and_layers)
}
m_semi <- PolrNN(fm_semi, data = train, list_of_deep_models = list(
  deep = embd_semi), optimizer = optimizer)
fit(m_semi, epochs = 10, callbacks = list(callback_early_stopping(
  patience = 2, restore_best_weights = TRUE)), verbose = FALSE)


## ----mods-comparison-ontram---------------------------------------------------
all.equal(unname(unlist(coef(m_0, which = "interacting"))),
  qlogis(mean(train$action == 0)), tol = 1e-6)

# compare test prediction performance
bci <- function(mod) {
   lli <- logLik(mod, newdata = test, convert_fun = identity)
   bt <- boot(lli, statistic = \(x, d) mean(x[d]), R = 1e4)
   btci <- boot.ci(bt, conf = 0.95, type = "perc")$percent[1, 4:5]
   c("nll" = mean(lli), "lwr" = btci[1], "upr" = btci[2])
}

mods <- list("unconditional" = m_0, "tabular only" = m_tab, 
  "text only" = m_text, "semi-structured" = m_semi)
do.call("cbind", lapply(mods, bci))

c("tabular only" = unlist(unname(coef(m_tab))), 
  "semi-structured" = unlist(unname(coef(m_semi))))


## ----embedding-pca-plot-------------------------------------------------------
d <- table(test$texts)
d_sorted <- d[order(d, decreasing = TRUE)]
nw <- as.integer(names(d_sorted[1:1e3]))

mod <- keras_model(embd$layers[[1]]$input, get_layer(
  embd, "embedding")$output)
number_words_per_movie <- 100
inp <- texts_to_sequences(tokenizer, words$word[nw]) |> 
  pad_sequences(maxlen = number_words_per_movie, truncating = "post")
dat <- mod(inp)$numpy()

dat <- cbind(words[nw, ], dat)
res <- prcomp(dat[,-c(1:2)])

### ggplot word embedding ###
df <- data.frame(
  PC1 = res$x[,1],
  PC2 = res$x[,2],
  words = dat$word
)

topN <- 100

gp1 <- ggplot(df, aes(x = PC1, y = PC2)) + geom_point(size = 0.2, col = "gray") +
  geom_point(data = df[1:topN, ], aes(x = PC1, y = PC2), size = 0.2, col = "black") +
  geom_text_repel(data = df[1:topN, ], aes(x = PC1, y = PC2, label = words),
                           size = rel(3.5), max.overlaps = 20) +
  theme_bw() + labs(subtitle = paste0(topN, " most frequent words"))

##### PCA of embedding from m_text ####
mod <- keras_model(embd$layers[[1]]$input, get_layer(
  embd, "penultimate")$output)
mod_movie_dat <- mod(test$texts)$numpy()
res <- prcomp(mod_movie_dat)

genres <- rep(0, length(test$genreAction)) # All movies
genres[test$genreAction == 1] <- 1 # Action Movis
genres[test$genreRomance == 1] <- 2 # Action Romance

### ggplot text embedding ###
df2 <- data.frame(
  PC1 = res$x[,1],
  PC2 = res$x[,2],
  genres = factor(genres, levels = 0:2, labels = c(
    "Neither", "Action", "Romance"))
)

gp2 <- ggplot(df2, aes(x = PC1, y = PC2, col = genres, size = genres)) + 
  geom_point() +
  scale_color_manual(values = c("gray", "red", "blue")) +
  scale_size_manual(values = 1.3 * c(0.3, 1, 1)) +
  theme_bw() + theme(legend.position = "top") +
  labs(subtitle = "movie reviews in the test data")

ggpubr::ggarrange(gp1, gp2, common.legend = TRUE)
ggsave(file = "embedding-pca.pdf", height = 4, width = 8)


## ----preliminaries_ATM, echo = FALSE, results = "hide", message=FALSE---------
d <- subset(tsdl, "Meteorology")
nm <- "Mean maximum temperature in Melbourne: degrees C. Jan 71 – Dec 90."
temp_idx <- sapply(d, attr, "description") == nm
y <- d_ts <- d[temp_idx][[1]]

n_gr <- 300
min_supp <- 10
max_supp <- 30
M <- 6L # order bsp
ep <- 1e4
p <- 3
gr <- seq(min_supp ,max_supp , length.out = n_gr)
time_var <- seq(as.Date("1971-01-01"), as.Date("1990-12-01"), by = "month")
d_ts <- data.table(time = time_var, y = as.numeric(d_ts))
d_ts[, month := factor(month(time))]
d_ts[, paste0("y_lag_", 1:p) := shift(y, n = 1:p, type = "lag", fill = NA)]
d_ts <- na.omit(d_ts)
len_y <- nrow(d_ts)


## ----formula-interface_ATM----------------------------------------------------
lags <- c(paste0("y_lag_", 1:p, collapse = "+"))
atplags <- c(paste0("atplag(y_lag_", 1:p, ")", collapse = "+"))
(fm_atm <- as.formula(paste0("y |", lags, "~ 0 + month +", atplags)))
(fm_atp <- as.formula(paste0("y ~ 0 + month +", atplags)))
(fm_colr <- as.formula(paste0("y ~ 0 + month + ", lags)))


## ----fitting-ATMs-------------------------------------------------------------
mod_fun <- \(fm) ColrNN(fm, data = d_ts, trafo_options = trafo_control(
  order_bsp = M, support = c(min_supp, max_supp)), tf_seed = 1,
  optimizer = optimizer_adam(learning_rate = 0.01))

mods <- lapply(list(fm_atm, fm_atp, fm_colr), mod_fun)

fit_fun <- \(m) m |> fit(epochs = ep, callbacks = list(
  callback_early_stopping(patience = 50, monitor = "loss"),
  callback_reduce_lr_on_plateau(patience = 20, factor = 0.9, monitor = "loss")), 
  batch_size = nrow(d_ts), validation_split = 0, verbose = FALSE)

lapply(mods, \(m) {
  mhist <- fit_fun(m)
  plot(mhist)
})


## ----in-sample-logLiks-ATM----------------------------------------------------
t_idx <- seq(as.Date("1977-06-01"), as.Date("1978-05-01"), by = "month")
ndl <- d_ts[d_ts$time %in% t_idx]
structure(unlist(lapply(mods, logLik, newdata = ndl)), names = c(
  "ATM", paste0("AT(", p, ")"), "Colr"))


## ----in_sample_cond_densities_ATMs--------------------------------------------
# In-sample densities
m_atm <- mods[[1]]
m_atp <- mods[[2]]
m_colr <- mods[[3]]

nd <- ndt <-  d_ts |> dplyr::mutate(y_true = y, y = list(gr)) |> 
  tidyr::unnest(y)
nd$d_atp <- c(predict(m_atp, newdata = nd, type = "pdf"))
nd$d_atm <- c(predict(m_atm, newdata = nd, type = "pdf"))
nd$d_colr <- c(predict(m_colr, newdata = nd, type = "pdf"))
d_density <- nd |> 
  tidyr::gather("method", "y_density", d_atm, d_atp, d_colr) |> 
  dplyr::mutate(y_grid = y) |> as.data.table()

# In-sample trafos
ndt$t_atp <- c(predict(m_atp, newdata = ndt, type = "trafo"))
ndt$t_atm <- c(predict(m_atm, newdata = ndt, type = "trafo"))
ndt$t_colr <- c(predict(m_colr, newdata = ndt, type = "trafo"))


## ----plot-ATMs, fig.width=10.8, fig.height=8.1--------------------------------
d_sub_dens <- d_density[time %in% t_idx]

Sys.setlocale("LC_ALL", "en_GB.UTF-8")
g_dens <- ggplot() + 
  geom_path(data = d_sub_dens, aes(x = y_true, y = time, group = method), 
            colour="red", size=1.5, alpha = 0.2) +
  geom_point(data = d_sub_dens, aes(x = y_true, y = time, group = method), 
             colour="red", size=1, shape=4) +
  geom_joy(data = d_sub_dens, 
           aes(height = y_density, x = y_grid, y = time, 
               group = time, fill = factor(time)), 
           stat="identity", alpha = 0.7, colour = rgb(0,0,0,0.5)) +
  scale_y_date(date_breaks = "4 months", date_labels = "%b %Y") + 
  scale_fill_viridis_d() +
  guides(fill = guide_legend(nrow = 2)) +
  facet_grid(~ method, labeller = as_labeller(c("d_atp" = paste0("AT(", p, ")"), 
                                              "d_atm" = "ATM",
                                              "d_colr" = "Colr"))) + 
  theme_bw() + 
  labs(color = "month") +
  xlab("") +
  ylab("") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        strip.background = element_blank(),
        panel.border = element_blank(),
        text = element_text(size=12),
        legend.position = "none",
        rect = element_rect(fill = "transparent"))

trafos <- ndt |> 
  tidyr::gather("method", "h", t_atm, t_atp, t_colr) |> 
  dplyr::filter(time %in% t_idx) |> 
  dplyr::mutate(month = ordered(
    format(as.Date(time), format = "%b %Y"),
    levels = format(sort(unique(as.Date(time))), format = "%b %Y")))

g_trafo <- ggplot(trafos) + 
    geom_line(aes(x = y, y = h, color = month)) +
    theme_set(theme_bw() + theme(legend.position = "bottom")) + 
    facet_grid(~ method, labeller = as_labeller(
      c("t_atm" = "ATM", "t_atp" = paste0("AT(", p, ")"), "t_colr" = "Colr"))) +
    ylab(expression(hat(h)(y[t]*"|"*Y[p] *"="* y[p]))) + 
    xlab(expression(y[t])) +
    scale_x_continuous(expand = c(0,0), breaks = c(15, 20, 25)) + 
    guides(color = guide_legend(nrow = 2)) +
    theme(text = element_text(size=12), legend.position = "bottom") + 
    xlab("Mean maximum temperature (°C) in Melbourne") +
    theme(strip.background = element_blank(), strip.text.x = element_blank()) +
    scale_colour_viridis_d()

(p1 <- g_dens / g_trafo)


## ----handling-censored-responses, eval=!ATMonly-------------------------------
deeptrafo:::response(y = c(0L, 1L))


## ----warmstart-and-fix-weights-shift, eval=!ATMonly---------------------------
nn <- keras_model_sequential() |>
  layer_dense(input_shape = 1L, units = 3L, activation = "relu",
  use_bias = FALSE, kernel_initializer = initializer_constant(value = 1))
unlist(get_weights(nn))


## ----weight_control, eval=!ATMonly--------------------------------------------
args(weight_control)


## ----warmstart-and-fix-weights-shift-2, eval=!ATMonly-------------------------
data("wine", package = "ordinal")
mw <- deeptrafo(rating ~ 0 + temp, data = wine,
  weight_options = weight_control(warmstart_weights = list(list(), list(), 
  list("temp" = 0))))
unlist(coef(mw))


## ----custom-basis, eval=!ATMonly----------------------------------------------
linear_basis <- function(y) {
  ret <- cbind(1, y)
  if (NROW(ret) == 1)
    return(as.vector(ret))
  ret
}

linear_basis_prime <- function(y) {
  ret <- cbind(0, rep(1, length(y)))
  if (NROW(ret) == 1)
    return(as.vector(ret))
  ret
}

constraint <- function(w, bsp_dim) {
  w_res <- tf$reshape(w, shape = list(bsp_dim, as.integer(nrow(w) / bsp_dim)))
  w1 <- tf$slice(w_res, c(0L, 0L), size = c(1L, ncol(w_res)))
  wrest <- tf$math$softplus(tf$slice(w_res, c(1L, 0L), size = c(as.integer(nrow(
    w_res) - 1), ncol(w_res))))
  w_w_cons <- k_concatenate(list(w1, wrest), axis = 1L)
  tf$reshape(w_w_cons, shape = list(nrow(w), 1L))
}

tfc <- trafo_control(
  order_bsp = 1L,
  y_basis_fun = linear_basis,
  y_basis_fun_prime = linear_basis_prime,
  basis = constraint
)

set.seed(1)
n <- 1e3
d <- data.frame(y = 1 + rnorm(1e3), x = rnorm(1e3))
m <- deeptrafo(y ~ 0 + x, data = d, trafo_options = tfc,
               optimizer = optimizer_adam(learning_rate = 1e-2),
               latent_distr = "normal")
fit(m, batch_size = 1e3, epochs = 1e3, validation_split = NULL,
    callbacks = list(callback_reduce_lr_on_plateau(monitor = "loss")),
    verbose = FALSE)
abs(unlist(coef(m)) - coef(Lm(y ~ x, data = d)))


## ----alternative-interface, eval=!ATMonly-------------------------------------
dord <- data.frame(Y = ordered(sample.int(6, 100, TRUE)), 
  X = rnorm(100), Z = rnorm(100))
ontram(response = ~ Y, intercept = ~ X, shift = ~ 0 + s(Z, df = 3),
  data = dord)


## ----large-factor-models------------------------------------------------------
set.seed(0)
n <- 1e6
nlevs <- 1e3
X <- factor(sample.int(nlevs, n, TRUE))
Y <- (X == 2) - (X == 3) + rnorm(n)
d <- data.frame(Y = Y, X = X)
m <- LmNN(Y ~ 0 + fac(X), data = d, additional_processor = list(
  fac = fac_processor), optimizer = optimizer_adam(learning_rate = 1e-2))
fit(m, batch_size = 1e4, epochs = 20, validation_split = 0, callbacks = list(
  callback_early_stopping("loss", patience = 3), 
  callback_reduce_lr_on_plateau("loss", 0.9, 2)), verbose = FALSE)
bl <- unlist(coef(m, which = "interacting"))
- (unlist(coef(m))[1:5] + bl[1]) / bl[2]

