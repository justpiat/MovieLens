# Loading the needed libraries (please note that this process could take a couple of minutes):
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dlookr)) install.packages("dlookr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dlookr)
library(lubridate)
library(recosystem)

# Loading the file into R (for R 4.0 or upper version):
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")


# Creating edx set and validation set (final hold-out test set). Validation set will be 10% of MovieLens data:
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Making sure userId and movieId in validation set are also present in edx set:
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Adding rows removed from validation set back into edx set:
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Checking properties of the dataset:
head(edx)
class(edx)
dim(edx)
diagnose(edx)

# Creating tibble with the number of observations of each score:
edx %>% group_by(rating) %>% summarise(n = n()) %>% arrange(desc(n))

# Creating tibble of 10 genres with the highest number of ratings:
genres = c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime", 
           "Documentary", "Drama", "Fantasy", "Film-Noir" ,"Horror", "Musical",
           "Mystery", "Romance","Sci-Fi", "Thriller", "War", "Western")
G <- sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})
G <- as.data.frame(G, optional = TRUE) %>% setNames(., c("Number of ratings"))
G %>% arrange(desc(`Number of ratings`)) %>% top_n(10)

# Histograms showing that some movies are rated more often than others and that some users are far more active than others in giving ratings:
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Tibble showing 10 movies with the greatest number of ratings with their average, median and standard deviation of their ratings:
edx %>% group_by(movieId, title) %>%
  summarize(count = n(), average = mean(rating), median = median(rating), 
            sd = sd(rating)) %>%
  arrange(desc(count))

# Rating variability plot:
T <- edx %>% group_by(movieId, title) %>%
  summarize(count = n(), average = mean(rating), median = median(rating), sd = sd(rating)) %>% 
  filter(count > 1000)
plot(T$average, T$sd, xlab="average rating", ylab="standard deviation")

# Calculating the overall average for all the ratings in the dataset:
mu <- mean(edx$rating)
print(mu)

# Histogram which demonstrates the number of movies that are on both sides of the overall average mu (movie effect):
edx %>% group_by(movieId) %>% summarize(b_m = mean(rating - mu)) %>% 
  qplot(b_m, geom ="histogram", bins = 12, data = ., color = I("black"))

# Plot showing variability among users (user effect):
edx %>% group_by(userId) %>% summarize(b_u = mean(rating - mu)) %>% 
  qplot(b_u, geom ="histogram", bins = 12, data = ., color = I("black"))

# Plot showing variability among genres (genre effect):
edx %>% group_by(genres) %>% 
  summarize(b_g = mean(rating - mu)) %>% 
  qplot(b_g, geom ="histogram", bins = 12, data = ., color = I("black"))

# Plot showing the number effect - the trend that movies with more ratings per year tend to have a slightly higher average rating:
  mutate(count = n()) %>%
  filter(count > 1000) %>% 
  summarize(b_n = n(), years = 2008 - first(year),
            title = title[1],
            rating = mean(rating)) %>%
  mutate(rate = b_n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth() +
  xlim(0, 5000)

# Creating train and test sets of ratings from edx:
set.seed(1, sample.kind = "Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Performing semi_join to make sure that all movies and users in the test set are present also in the training set:
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% 
  semi_join(train_set, by = "userId")

# Constructing RMSE function:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculating the average rating in the train set:
mu <- mean(train_set$rating)
print(mu)

# MOVIE BIAS MODEL:
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_m
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_1_rmse)

# MOVIE + USER BIAS MODEL:
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_m + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_2_rmse)

# MOVIE + USER + GENRE BIAS MODEL:
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% 
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_m - b_u))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred2 = mu + b_m + b_u + b_g) %>%
  .$pred2
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_3_rmse)

# MOVIE + USER + GENRE + NUMBER BIAS MODEL:
number_avgs <- train_set %>% 
  mutate(year = year(as_datetime(timestamp))) %>% group_by(movieId) %>% 
  mutate(count = n()) %>%
  mutate(n = count, years = 2008 - first(year),
         title = title[1],
         rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ungroup() %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% 
  left_join(genre_avgs, by='genres') %>% 
  group_by(rate) %>%
  summarize(movieId = movieId, b_n = mean(rating - mu - b_m - b_u - b_g))

# Removing the first column and keeping only unique values:
number_avgs <- number_avgs[,-1]
number_avgs <- number_avgs %>% distinct(movieId, .keep_all = TRUE)

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>% 
  left_join(number_avgs, by='movieId') %>% 
  mutate(pred3 = mu + b_m + b_u + b_g + b_n) %>%
  .$pred3
model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
print(model_4_rmse)

# Calculating predictions with regularized movie, user, genre and number biases for the sequence of lambdas between 0 and 5:
# Please note that the following model takes a few minutes to execute.
lambdas <- seq(0, 5, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_m <- train_set %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  b_g <- train_set %>% 
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_m - b_u- mu)/(n()+l))
  b_n <- train_set %>% 
    mutate(year = year(as_datetime(timestamp))) %>% group_by(movieId) %>% 
    mutate(count = n()) %>%
    mutate(n = count, years = 2008 - first(year),
           title = title[1],
           rating = mean(rating)) %>%
    mutate(rate = n/years) %>%
    ungroup() %>%
    left_join(b_m, by='movieId') %>%
    left_join(b_u, by='userId') %>% 
    left_join(b_g, by='genres') %>% 
    group_by(rate) %>%
    summarize(movieId = movieId, b_n = sum(rating - b_m - b_u - b_g - mu)/(n()+l))
  b_n <- b_n[,-1]
  b_n <- b_n %>% distinct(movieId, .keep_all = TRUE)
  predicted_ratings <- 
    test_set %>% 
    left_join(b_m, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_n, by = 'movieId') %>%
    mutate(pred = mu + b_m + b_u + b_g + b_n) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
lambda <- lambdas[which.min(rmses)]
model_5_rmse <- min(rmses)
print(model_5_rmse)

# Converting the data to recosystem format:
train_data <- data_memory(user_index = train_set$userId, 
                          item_index = train_set$movieId,
                          rating = train_set$rating)

test_data <- data_memory(user_index = test_set$userId, 
                         item_index = test_set$movieId,
                         rating = test_set$rating)

# Training the data using recosystem package:
recommender <- Reco()
recommender$train(train_data)

# Returning predicted values in memory using test set and calculating RMSE:
predictions <- recommender$predict(test_data, out_memory())
model_6_rmse <- RMSE(predictions, test_set$rating)
print(model_6_rmse)

# Converting the validation set into recosystem format:
validation_data <- data_memory(user_index = validation$userId, 
                               item_index = validation$movieId,
                               rating = validation$rating)

# Calculating predictions and the final RMSE:
predictions_final <- recommender$predict(validation_data, out_memory())
final_model_RMSE <- RMSE(predictions_final, validation$rating)
print(final_model_RMSE)

# Creating a data frame storing the results of all the models:
rmse_results <- data_frame(method = c("MOVIE BIAS MODEL", "MOVIE + USER BIAS MODEL", 
                                      "MOVIE + USER + GENRE BIAS MODEL", 
                                      "MOVIE + USER + GENRE + NUMBER BIAS MODEL", 
                                      "REGULARIZED MODEL", "MF MODEL", "FINAL MODEL"),
                           RMSE = c(model_1_rmse, model_2_rmse, model_3_rmse, model_4_rmse, 
                                    model_5_rmse, model_6_rmse, final_model_RMSE))
rmse_results$RMSE <- format(round(rmse_results$RMSE,8),nsmall=8)
print(rmse_results)