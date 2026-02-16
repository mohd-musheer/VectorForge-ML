# library(VectorForgeML)
# source("R/LinearRegression.R")
# source("R/split.R")
# source("R/scalers.R")
# source("R/encoders.R")
# ls("package:VectorForgeML")

# cat("Loading dataset...\n")


# df <- read.csv("inst/dataset/cars.csv", stringsAsFactors = FALSE)
# df <- df[rep(1:nrow(df), 10), ]
# cat("Rows:", nrow(df), "\n")

# y <- df$msrp
# df$msrp <- NULL

# cat_cols <- sapply(df, is.character)
# cat_df <- df[, cat_cols, drop = FALSE]
# num_df <- df[, !cat_cols, drop = FALSE]

# start <- Sys.time()

# encoder <- OneHotEncoder$new()
# cat_encoded <- encoder$fit_transform(cat_df)

# X <- cbind(as.matrix(num_df), cat_encoded)

# if (exists("cpp_drop_constant_cols", mode = "function")) {
#   X <- cpp_drop_constant_cols(X)$X
# } else {
#   X <- X[, apply(X, 2, var) != 0]
# }

# data <- train_test_split(X, y, seed = 42)
# scaler <- StandardScaler$new()
# X_train <- scaler$fit_transform(data$X_train)

# model <- LinearRegression$new()
# model$fit(X_train, data$y_train)

# end <- Sys.time()
# cat("\nYour Framework Time:", as.numeric(end - start), "seconds\n")

library(VectorForgeML)
# source("R/pipeline.R")
# source("R/column_transformer.R")


cat("Loading dataset...\n")

df <- read.csv("inst/dataset/cars.csv", stringsAsFactors=FALSE)
df <- df[rep(1:nrow(df),10), ]
cat("Rows:", nrow(df), "\n")

y <- df$msrp
df$msrp <- NULL

# detect column types
cat_cols <- names(df)[sapply(df,is.character)]
num_cols <- names(df)[!sapply(df,is.character)]

# =========================
# BUILD PIPELINE
# =========================

preprocessor <- ColumnTransformer$new(
  num_cols=num_cols,
  cat_cols=cat_cols,
  num_pipeline=StandardScaler$new(),
  cat_pipeline=OneHotEncoder$new()
)

pipe <- Pipeline$new(list(
  preprocessor,
  LinearRegression$new()
))

# =========================
# TRAIN
# =========================

start <- Sys.time()

pipe$fit(df,y)

end <- Sys.time()

cat("\nTrain Time:", as.numeric(end-start),"seconds\n")

# =========================
# PREDICT
# =========================

pred <- pipe$predict(df)

cat("RMSE:", rmse(y,pred),"\n")
cat("R2:", r2_score(y,pred),"\n")
