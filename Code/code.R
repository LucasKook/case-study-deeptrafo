################################################################################
################################################################################
# Replication material for "Estimating conditional distributions with Neural
# Networks using R package deeptrafo" by Lucas Kook, Philipp FM Baumann, Oliver
# Duerr, Beate Sick, and David Ruegamer.
################################################################################
################################################################################

# Dependencies can be installed via the separately provided `dependencies.R`
# file

## ----include=FALSE------------------------------------------------------------
library("knitr")
opts_chunk$set(engine = "R", tidy = FALSE, prompt = TRUE, cache = FALSE,
               fig.width = 6.5 * 0.8, fig.height = 4.5 * 0.8,
               fig.fullwidth = TRUE, fig.path = "Figures/",
               fig.ext = c("pdf", "eps", "jpg", "tiff"), dpi = 600)
knitr::render_sweave()  # use Sweave environments
knitr::set_header(highlight = "")  # do not \usepackage{Sweave}
## R settings
options(prompt = "R> ", continue = "+  ", width = 76, useFancyQuotes = FALSE,
        digits = 3L)

## ----preliminaries, echo = FALSE, results = "hide", message=FALSE-------------
library("deeptrafo")
library("ggplot2")
library("tsdl") # available from GitHub (FinYang/tsdl)
library("reticulate")
library("safareg")
library("data.table")
library("patchwork")
library("ggridges")
library("moments")
library("lubridate")
library("boot")
library("ggrepel")
library("tram")

theme_set(theme_bw() + theme(legend.position = "top"))

outdir <- "../Figures"
if (!dir.exists(outdir)) dir.create(outdir)

# Params ------------------------------------------------------------------

bpath <- "../Data"
nr_words <- 1e4
embedding_size <- 1e2
maxlen <- 1e2
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

ggplot(data.frame(vote_count = train$vote_count), aes(x = vote_count)) +
  stat_ecdf() +
  geom_vline(xintercept = test$vote_count[1], color = "darkblue", linetype = 2) +
  labs(y = "ECDF (training data)") +
  theme(text = element_text(size = 13.5))

ggsave(file.path(outdir, "vote-count.pdf"), width = 4, height = 3)

## ----formula-interface, eval=!ATMonly-----------------------------------------
fm <- vote_count | genreAction ~ 0 + s(budget, df = 6) + popularity


## ----setting-up-dctms, eval=!ATMonly------------------------------------------
opt <- optimizer_adam(learning_rate = 0.1, decay = 4e-4)
(m_fm <- cotramNN(formula = fm, data = train, optimizer = opt))


## ----fitting-dctms, eval=!ATMonly---------------------------------------------
m_fm_hist <- fit(m_fm, epochs = 1e3, validation_split = 0.1, batch_size = 64,
                 verbose = FALSE)
unlist(coef(m_fm, which = "shifting"))


## ----plot-hist, fig.width=8.1, fig.height=4.05, eval=!ATMonly-----------------
p1 <- data.frame(x = 1:m_fm_hist$params$epochs,
                 training = m_fm_hist$metrics$loss,
                 validation = m_fm_hist$metrics$val_loss) |>
  tidyr::gather("set", "loss", training, validation) |>
  ggplot(aes(x = x, y = loss, color = set)) +
  geom_line() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "epochs", color = "") +
  theme(text = element_text(size = 13))

nd <- list(genreAction = c(0, 1), budget = rep(mean(train$budget), 2),
           popularity = rep(mean(train$popularity), 2))
preds <- predict(m_fm, newdata = nd, q = seq(0, 2000, 25), type = "trafo")
pdat <- data.frame(y = as.numeric(names(preds)),
                   ga0 = unlist(lapply(preds, \(x) x[1, 1])),
                   ga1 = unlist(lapply(preds, \(x) x[2, 1])))
p2 <- pdat |> tidyr::gather("action", "trafo", ga0, ga1) |>
  ggplot(aes(x = y, y = trafo, color = action)) +
  geom_step() +
  labs(x = "vote count", y = "transformation function", color = "genreAction") +
  scale_color_manual(values = colorspace::diverging_hcl(2),
                     labels = c("ga0" = 0, "ga1" = 1)) +
  theme(text = element_text(size = 13))

ggpubr::ggarrange(
  p1 + labs(tag = "A"), p2 + labs(tag = "B"), nrow = 1,
  common.legend = FALSE
)

ggsave(file.path(outdir, "plot-hist.pdf"), width = 8.1, height = 4.05)

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

fit(m_deep, epochs = 50, validation_split = 0.1, batch_size = 32,
    callbacks = list(callback_early_stopping(patience = 5)), verbose = FALSE)


## ----ensembling-dctms, eval=!ATMonly------------------------------------------
ens_deep <- ensemble(m_deep, n_ensemble = 3, epochs = 50, batch_size = 64,
                     verbose = FALSE)


## ----ensembling-dctms-methods, eval=!ATMonly----------------------------------
unlist(logLik(ens_deep, newdata = test, convert_fun = \(x) - mean(x)))

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
pdf(file.path(outdir, "ensembling-dctms-plot.pdf"), height = 4.5 * 0.8, width = 6.5 * 0.8)
plot(pdat$value, pdat$V2, type = "l", ylim = range(pdat[, -1]),
     xlab = "log(1 + bugdet)", ylab = "partial effect", col = "gray80", lty = 2,
     las = 1)
matlines(pdat$value, pdat[, -1:-2], col = "gray80", lty = 2)
rug(train$budget, col = rgb(.1, .1, .1, .3))

lines(nd$value, nd$ens)
polygon(c(nd$value, rev(nd$value)),
        c(nd$ens + 2 * nd$sd, rev(nd$ens - 2 * nd$sd)),
        col = rgb(.1, .1, .1, .1), border = FALSE)

legend("topright", legend = c("ensemble", "individual"), lty = c(1, 2),
       col = c(1, "gray80"), lwd = 1, bty = "n")
dev.off()


## ----cross-validating-dctms, fig.width=9, fig.height=4.05, eval=!ATMonly------
cv_deep <- cv(m_deep, epochs = 50, cv_folds = 5, batch_size = 64)
pdf(file.path(outdir, "cross-validating-dctms.pdf"), height = 4.5, width = 9)
plot_cv(cv_deep)
dev.off()


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

all.equal(unlist(unname(coef(m_0, which = "interacting"))),
  qlogis(mean(train$action == 0)), tol = 1e-6)

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
  embd, "embedding_1")$output)
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
  theme_bw() + labs(subtitle = paste0(topN, " most frequent words")) +
  theme(text = element_text(size = 13))

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
  labs(subtitle = "movie reviews in the test data") +
  theme(text = element_text(size = 13))

ggpubr::ggarrange(gp1, gp2, common.legend = TRUE)
ggsave(file = file.path(outdir, "embedding-pca.pdf"), height = 4, width = 8)


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
d_ts_lag <- d_ts[, paste0("y_lag_", 1:p) := shift(y, n = 1:p, type = "lag", fill = NA)]
d_ts_lag <- na.omit(d_ts_lag)

## ----formula-interface_ATM----------------------------------------------------
lags <- c(paste0("y_lag_", 1:p, collapse = "+"))
(fm_atm <- as.formula(paste0("y |", lags, "~ 0 + month + atplag(1:p)")))
(fm_atp <- y ~ 0 + month + atplag(1:p))
(fm_colr <- as.formula(paste0("y ~ 0 + month + ", lags)))


## ----fitting-ATMs-------------------------------------------------------------
mod_fun <- function(fm, d) ColrNN(fm, data = d, trafo_options = trafo_control(
  order_bsp = M, support = c(min_supp, max_supp)), tf_seed = 1,
  optimizer = optimizer_adam(learning_rate = 0.01))

mods <- c(lapply(list(fm_atm, fm_atp), mod_fun, d = d_ts),
          lapply(list(fm_colr), mod_fun, d = d_ts_lag))

fit_fun <- \(m) m |> fit(epochs = ep, callbacks = list(
  callback_early_stopping(patience = 50, monitor = "loss"),
  callback_reduce_lr_on_plateau(patience = 20, factor = 0.9, monitor = "loss")),
  batch_size = nrow(d_ts_lag), validation_split = 0, verbose = FALSE)

lapply(mods, \(m) {
  mhist <- fit_fun(m)
  # plot(mhist)
})


## ----in-sample-logLiks-ATM----------------------------------------------------
t_span_one <- seq(as.Date("1977-03-01"), as.Date("1978-05-01"), by = "month")
ndl <- d_ts[d_ts$time %in% t_span_one]
t_span_two <- seq(as.Date("1977-06-01"), as.Date("1978-05-01"), by = "month")
ndl_lag <- d_ts_lag[d_ts_lag$time %in% t_span_two]
structure(unlist(c(lapply(mods[1:2], logLik, newdata = ndl),
                   lapply(mods[3], logLik, newdata = ndl_lag))), names = c(
  "ATM", paste0("AT(", p, ")"), "Colr"))


## ----in_sample_cond_densities_ATMs--------------------------------------------
# In-sample densities
m_atm <- mods[[1]]
m_atp <- mods[[2]]
m_colr <- mods[[3]]

# colr
nd <- ndt <- d_ts_lag |> dplyr::mutate(y_true = y, y = list(gr)) |>
  tidyr::unnest(y)
nd$d_colr <- c(predict(m_colr, newdata = nd, type = "pdf"))

pg <- TRUE
attr(pg, "rname") <- "y_true"
attr(pg, "y") <- "y"

# atp and atm
nd_atp <- ndt_atp <- d_ts |> dplyr::mutate(y_true = y, y = list(gr)) |>
  tidyr::unnest(y)
nd_atp_lag <- subset(nd_atp, time >= "1971-04-01")
nd_atp <- nd_atp[order(nd_atp$y, nd_atp$time),]
nd_atp_lag <- nd_atp_lag[order(nd_atp_lag$y, nd_atp_lag$time),]
nd_atp_lag$d_atp <- c(predict(m_atp, newdata = nd_atp, type = "pdf",
                              pred_grid = pg))
nd_atp_lag$d_atm <- c(predict(m_atm, newdata = nd_atp, type = "pdf",
                              pred_grid = pg))

nd <- merge(nd, nd_atp_lag)

d_density <- nd |>
  tidyr::gather("method", "y_density", d_atm, d_atp, d_colr) |>
  dplyr::mutate(y_grid = y) |> as.data.table()

# In-sample trafos

# colr
ndt$t_colr <- c(predict(m_colr, newdata = ndt, type = "trafo"))

# atp and atm
ndt_atp_lag <- subset(ndt_atp, time >= "1971-04-01")
ndt_atp <- ndt_atp[order(ndt_atp$y, ndt_atp$time),]
ndt_atp_lag <- nd_atp_lag[order(nd_atp_lag$y, nd_atp_lag$time),]
ndt_atp_lag$t_atp <- c(predict(m_atp, newdata = ndt_atp, type = "trafo",
                               pred_grid = pg))
ndt_atp_lag$t_atm <- c(predict(m_atm, newdata = ndt_atp, type = "trafo",
                               pred_grid = pg))

ndt <- merge(ndt, ndt_atp_lag)

## ----plot-ATMs, fig.width=10.8, fig.height=8.1--------------------------------
plot_period <- seq(as.Date("1983-06-01"), as.Date("1984-05-01"), by = "month")
d_sub_dens <- d_density[time %in% plot_period]

Sys.setlocale("LC_ALL", "en_GB.UTF-8")
g_dens <- ggplot() +
  geom_path(data = d_sub_dens, aes(x = y_true, y = time, group = method),
            colour="red", linewidth = 1.5, alpha = 0.2) +
  geom_point(data = d_sub_dens, aes(x = y_true, y = time, group = method),
             colour="red", size = 1, shape = 4) +
  geom_density_ridges(data = d_sub_dens,
           aes(height = y_density, x = y_grid, y = time,
               group = time, fill = factor(time)),
           stat = "identity", alpha = 0.7, colour = rgb(0, 0, 0, 0.5)) +
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
        text = element_text(size = 13),
        legend.position = "none",
        rect = element_rect(fill = "transparent"))

trafos <- ndt |>
  tidyr::gather("method", "h", t_atm, t_atp, t_colr) |>
  dplyr::filter(time %in% plot_period) |>
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
  theme(text = element_text(size = 13), legend.position = "bottom") +
  xlab("Mean maximum temperature (°C) in Melbourne") +
  theme(strip.background = element_blank(), strip.text.x = element_blank()) +
  scale_colour_viridis_d()

(p1 <- g_dens / g_trafo)

ggsave(file.path(outdir, "plot-ATMs.pdf"), p1, width = 10.8, height = 8.1)

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
mw <- deeptrafo(rating ~ 0 + temp, data = wine, weight_options = weight_control(
  warmstart_weights = list(list(), list(), list("temp" = 0))))
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
d <- data.frame(y = 1 + rnorm(n), x = rnorm(n))
m <- deeptrafo(y ~ 0 + x, data = d, trafo_options = tfc,
               optimizer = optimizer_adam(learning_rate = 1e-2),
               latent_distr = "normal")
fit(m, batch_size = n, epochs = 5e3, validation_split = NULL,
    callbacks = list(callback_reduce_lr_on_plateau(monitor = "loss")),
    verbose = FALSE)
abs(unlist(coef(m)) - coef(Lm(y ~ x, data = d)))


## ----alternative-interface, eval=!ATMonly-------------------------------------
dord <- data.frame(Y = ordered(sample.int(6, 100, TRUE)),
                   X = rnorm(100), Z = rnorm(100))
ontram(response = ~ Y, intercept = ~ X, shift = ~ 0 + s(Z, df = 3),
       data = dord)

## ----word2vec-----------------------------------------------------------------

### Load embedding
embedding_dim <- 300

if (file.exists("word2vec_embd_matrix.RDS")) {

  embedding_matrix <- readRDS("word2vec_embd_matrix.RDS")
  vocab_size <- nrow(embedding_matrix)

} else {

  # py_install(packages = "gensim")
  gensim <- import("gensim")
  # download embedding
  # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
  model <- gensim$models$KeyedVectors$load_word2vec_format(
    "../Data/GoogleNews-vectors-negative300.bin", binary = TRUE)

  vocab_size <- length(words$word)
  embedding_matrix <- matrix(0, nrow = vocab_size, ncol = embedding_dim)

  names_model <- names(model$key_to_index)
  for (i in 1:vocab_size) {
    word <- words$word[i]
    if (word %in% names_model) {
      embedding_matrix[i, ] <- model[[word]]
    }
  }
  saveRDS(embedding_matrix, file = "word2vec_embd_matrix.RDS")

}

### Shallow
w2v_mod <- function(x) x |>
  layer_embedding(input_dim = vocab_size, output_dim = embedding_dim,
                  weights = list(embedding_matrix), trainable = FALSE) |>
  layer_flatten() |>
  layer_dense(units = 1)

fm_w2v <- action ~ 0 + shallow(texts)
m_w2v <- deeptrafo(fm_w2v, data = train,
                   list_of_deep_models = list(shallow = w2v_mod),
                   optimizer = optimizer_adam(learning_rate = 1e-5))

dhist <- fit(m_w2v, epochs = 200, validation_split = 0.1, batch_size = 32,
             callbacks = list(callback_early_stopping(patience = 5)),
             verbose = FALSE)

bci(m_w2v)

### Deep
w2v2_mod <- function(x) x |>
  layer_embedding(input_dim = vocab_size, output_dim = embedding_dim,
                  weights = list(embedding_matrix), trainable = FALSE) |>
  layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") |>
  layer_max_pooling_1d(pool_size = 5) |>
  layer_conv_1d(filters = 128, kernel_size = 5, activation = "relu") |>
  layer_global_max_pooling_1d() |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 1, activation = "sigmoid")

fm_w2v2 <- action ~ 0 + deep(texts)
m_w2v2 <- deeptrafo(fm_w2v2, data = train,
                    list_of_deep_models = list(deep = w2v2_mod),
                    optimizer = optimizer_adam(learning_rate = 1e-5))

dhist <- fit(m_w2v2, epochs = 200, validation_split = 0.1, batch_size = 32,
             callbacks = list(callback_early_stopping(patience = 5)),
             verbose = FALSE)

bci(m_w2v2)

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
logLik(m, batch_size = 1e4, convert_fun = \(x) - mean(x))
