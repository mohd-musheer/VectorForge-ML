library(VectorForgeML)

source("R/LinearRegression.R")
source("R/metrics.R")
source("R/split.R")
source("R/scalers.R")

cat("Loading dataset...\n")

# LOAD DATASET
df <- read.csv("dataset/students.csv")
print(head(df))

# TARGET
y <- df$End_Sem_Marks

# REMOVE TARGET + USELESS COLS
df$End_Sem_Marks <- NULL
df$Student_ID <- NULL

# FEATURES MATRIX
X <- as.matrix(df)

# REMOVE CONSTANT COLUMNS
if (exists("cpp_drop_constant_cols", mode = "function")) {
  X <- cpp_drop_constant_cols(X)$X
} else {
  X <- X[, apply(X, 2, var) != 0]
}

cat("\nSamples:", nrow(X))
cat("\nFeatures:", ncol(X), "\n")

# TRAIN TEST SPLIT
data <- train_test_split(X, y, seed = 42)

# SCALING
scaler <- StandardScaler$new()
X_train <- scaler$fit_transform(data$X_train)
X_test <- scaler$transform(data$X_test)

# TRAIN MODEL
cat("\nTraining model...\n")
model <- LinearRegression$new()
model$fit(X_train, data$y_train)

# PREDICT
pred <- model$predict(X_test)

# METRICS
cat("\nResults\n")
cat("RMSE:", rmse(data$y_test, pred), "\n")
cat("MSE:", mse(data$y_test, pred), "\n")
cat("R2:", r2_score(data$y_test, pred), "\n")