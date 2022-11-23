# Preprocess movies
# deeptrafo case study
# LK, DR 2022

bpath <- "."
nr_words <- 1e4
embedding_size <- 1e2
maxlen <- 1e2
order_bsp <- 25
repetitions <- 2

if (!file.exists(file.path(bpath, "data_splitted.RDS"))) {

  library("keras")
  library("tidyverse")
  library("jsonlite")
  library("lubridate")
  library("ggplot2")
  library("ggsci")
  library("tidytext")
  library("tm")
  library("data.table")
  theme_set(theme_bw())

  # read in data
  movies <- read_csv(file.path(bpath, "movies.csv"))

  # get genres
  movies <- movies %>%
    filter(original_language == "en" & status == "Released") %>%
    # filter(nchar(genres)>2) %>%
    mutate(genres = lapply(genres, function(x)
      fromJSON(x)$name)) %>%
    select(
      genres,
      budget,
      overview,
      popularity,
      production_countries,
      release_date,
      revenue,
      runtime,
      vote_average,
      vote_count
    ) %>%
    filter(vote_count > 0) %>%
    mutate(release_date = as.numeric(as.Date("2020-01-01") - as.Date(release_date)))

  genres <-
    movies %>% unnest(c(genres), names_repair = "unique") %>% select(genres)
  table(genres)

  movies <- movies[unlist(sapply(movies$genres, function(x)
    length(x) > 1 | (!("TV Movie" %in% x) & !("Foreign" %in% x)))), ]

  # vote_average
  ggplot(
    movies %>%  unnest(c(genres), names_repair = "unique") %>%
      filter(!genres %in% c(NA, "TV Movie", "Foreign")),
    aes(x = log(revenue), fill = genres)
  ) +
    geom_density(alpha = 0.3) + xlab("log-revenue") +
    theme_bw() +
    theme(
      axis.line = element_line(colour = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      legend.title = element_blank(),
      text = element_text(size = 14),
      legend.position = "bottom"
    ) +
    scale_color_jco() +
    guides(fill = guide_legend(nrow = 3, byrow = TRUE))

  all_genres <- sort(unique(unlist(movies$genres)))
  genres_wide <- t(sapply(movies$genres,
                          function(x)
                            colSums(
                              model.matrix( ~ -1 + genre,
                                            data = data.frame(genre =
                                                                factor(x, levels = all_genres)))
                            )))

  colnames(genres_wide) <- gsub(" ", "_", colnames(genres_wide))

  movies <- cbind(movies %>% select(-genres), genres_wide)


  # init tokenizer
  tokenizer <- text_tokenizer(num_words = nr_words)

  # remove stopwords
  data("stop_words")
  stopwords_regex <- paste(c(stopwords('en'), stop_words$word),
                           collapse = '\\b|\\b')
  stopwords_regex <- paste0('\\b', stopwords_regex, '\\b')
  movies$overview <- tolower(movies$overview)
  movies$overview <-
    stringr::str_replace_all(movies$overview, stopwords_regex, '')
  movies$overview <- gsub('[[:punct:] ]+', ' ', movies$overview)

  saveRDS(movies$overview, file = file.path(bpath, "mov_ov.RDS"))

  tokenizer %>% fit_text_tokenizer(movies$overview)

  # text to sequence
  text_seqs <- texts_to_sequences(tokenizer, movies$overview)

  # pad text sequences
  text_padded <- text_seqs %>%
    pad_sequences(maxlen = maxlen, truncating = "post")

  # save words for later
  words <- tibble(word = names(tokenizer$word_index),
                  id = as.integer(unlist(tokenizer$word_index)))

  words <- words %>%
    filter(id <= tokenizer$num_words) %>%
    arrange(id)

  saveRDS(words, file = file.path(bpath, "words.RDS"))
  rm(words)
  gc()

  # text sequences as list of one array
  text_embd <-
    list(texts = array(text_padded, dim = c(NROW(movies), maxlen)))

  # create input list
  data <- append(movies, text_embd)

  # transformations
  data$vote_count <- as.integer(data$vote_count)
  data$budget <- log(data$budget + 1)
  data$revenue <- log(data$revenue + 1)
  data$popularity <- log(data$popularity + 1)

  rm(movies, text_embd)
  gc()

  data_list <- list()

  # repeat analysis 10 times
  for (repl in 1:repetitions) {
    set.seed(41 + repl)

    train_ind <-
      sample(1:NROW(data$runtime), round(0.8 * NROW(data$runtime)))
    test_ind <- setdiff(1:NROW(data$runtime), train_ind)

    train <-
      lapply(data, function(x)
        if (length(dim(x)) == 2)
          x[train_ind, ]
        else
          x[train_ind])
    test <-
      lapply(data, function(x)
        if (length(dim(x)) == 2)
          x[test_ind, ]
        else
          x[test_ind])

    data_list[[repl]] <- list(train = train, test = test)
  }

  saveRDS(data_list, file.path(bpath, "data_splitted.RDS"))
} else {
  data_list <- readRDS(file.path(bpath, "data_splitted.RDS"))
}
