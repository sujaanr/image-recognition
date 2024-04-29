# Load necessary libraries
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(kknn)
library(gbm)
library(kernlab)
library(nnet)

# Read data
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)

# Convert to matrix for image operations
train_matrix <- as.matrix(train)
test_matrix <- as.matrix(test)

# Define custom color palette for plotting
colors <- c('white', 'black')
color_palette <- colorRampPalette(colors = colors)

# Plot the average image of each digit
par(mfrow = c(4, 3), pty = 's', mar = c(1, 1, 1, 1), xaxt = 'n', yaxt = 'n')
average_digits <- array(dim = c(10, 784))
for (digit in 0:9) {
  digit_data <- train_matrix[train_matrix[, 1] == digit, -1]
  average_image <- colSums(digit_data) / max(colSums(digit_data)) * 255
  average_digits[digit + 1, ] <- average_image
  image_matrix <- array(average_image, dim = c(28, 28))[ , 28:1]  # Rotate to display correctly
  image(1:28, 1:28, image_matrix, main = as.character(digit), col = color_palette(256))
}

# Preprocess the data
train <- train[, -nearZeroVar(train)]
train[-1] <- sapply(train[-1], scale)
test <- sapply(test, scale)

# Split data into training and test sets
set.seed(123)  # for reproducibility
indexes <- sample(1:nrow(train), size = 0.3 * nrow(train))
train_digits <- train[-indexes,]
test_digits <- train[indexes,]

# Model fitting and evaluation
# Decision Tree
fit_decision_tree <- function(data) {
  model <- rpart(label ~ ., data = data, method = "class")
  model_pruned <- prune(model, cp = model$cptable[which.min(model$cptable[, "xerror"]), "CP"])
  list(model = model_pruned, predict_train = predict(model_pruned, data, type = "class"))
}

# Random Forest
fit_random_forest <- function(data) {
  model <- randomForest(label ~ ., data = data, ntree = 100)
  list(model = model, predict_train = predict(model, data))
}

# Naive Bayes
fit_naive_bayes <- function(data) {
  model <- naiveBayes(x = data[,-1], y = data[,1])
  list(model = model, predict_train = predict(model, data[-1]))
}

# K-Nearest Neighbors
fit_knn <- function(train_data, test_data) {
  train_control <- trainControl(method = "cv", number = 10)
  model <- train(label ~ ., data = train_data, method = "knn", trControl = train_control)
  list(model = model, predict_train = predict(model, train_data), predict_test = predict(model, test_data))
}

# Gradient Boosting Machine
fit_gbm <- function(data) {
  gbm_grid <- expand.grid(interaction.depth = c(1, 5, 9), n.trees = seq(50, 300, by = 50), shrinkage = 0.1)
  train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, search = "grid")
  model <- train(label ~ ., data = data, method = "gbm", tuneGrid = gbm_grid, trControl = train_control)
  list(model = model, predict_train = predict(model, data))
}

# Support Vector Machine
fit_svm <- function(data) {
  train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, search = "grid")
  model <- train(label ~ ., data = data, method = "svmRadial", trControl = train_control)
  list(model = model, predict_train = predict(model, data))
}

# Neural Network
fit_nn <- function(data) {
  size_grid <- expand.grid(size = c(1, 5, 10), decay = c(0.001, 0.01))
  train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, search = "grid")
  model <- train(label ~ ., data = data, method = "nnet", tuneGrid = size_grid, trControl = train_control)
  list(model = model, predict_train = predict(model, data))
}

dt_results <- fit_decision_tree(train_digits)
rf_results <- fit_random_forest(train_digits)
nb_results <- fit_naive_bayes(train_digits)
knn_results <- fit_knn(train_digits, test_digits)
gbm_results <- fit_gbm(train_digits)
svm_results <- fit_svm(train_digits)
nn_results <- fit_nn(train_digits)

# Display results and calculate accuracy
calculate_accuracy <- function(predictions, actual) {
  mean(predictions == actual$label)
}

# Print accuracies
cat("Decision Tree Accuracy:", calculate_accuracy(dt_results$predict_train, train_digits), "\n")
cat("Random Forest Accuracy:", calculate_accuracy(rf_results$predict_train, train_digits), "\n")
cat("Naive Bayes Accuracy:", calculate_accuracy(nb_results$predict_train, train_digits), "\n")
cat("KNN Accuracy:", calculate_accuracy(knn_results$predict_test, test_digits), "\n")
cat("GBM Accuracy:", calculate_accuracy(gbm_results$predict_train, train_digits), "\n")
cat("SVM Accuracy:", calculate_accuracy(svm_results$predict_train, train_digits), "\n")
cat("NN Accuracy:", calculate_accuracy(nn_results$predict_train, train_digits), "\n")
