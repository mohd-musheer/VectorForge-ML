# =====================================================cls
# VectorForgeML Ridge Regression Full Pipeline Example
# =====================================================

library(VectorForgeML)
ls("package:VectorForgeML")
cat("Loading dataset...\n")

# ---- LOAD DATA ----
df <- read.csv("inst/dataset/cars.csv")

cat("Rows:", nrow(df), "\n\n")

# ---- TARGET + FEATURES ----
y <- df$msrp
X <- df
X$msrp <- NULL


# ---- TRAIN TEST SPLIT ----
split <- train_test_split(X, y, test_size = 0.2, seed = 42)


# ---- DETECT COLUMN TYPES ----
cat_cols <- names(X)[sapply(X, is.character)]
num_cols <- names(X)[!sapply(X, is.character)]


# ---- PREPROCESSOR ----
preprocessor <- ColumnTransformer$new(
  num_cols = num_cols,
  cat_cols = cat_cols,
  num_pipeline = StandardScaler$new(),
  cat_pipeline = OneHotEncoder$new()
)


# ---- PIPELINE ----
pipe <- Pipeline$new(list(
  preprocessor,
  RidgeRegression$new()
))


# ---- TRAIN MODEL ----
cat("Training model...\n")

start <- Sys.time()

pipe$fit(split$X_train, split$y_train)

end <- Sys.time()

cat("Train time:", round(as.numeric(end-start),4),"sec\n\n")


# ---- PREDICTION ----
pred <- pipe$predict(split$X_test)


# ---- METRICS ----
cat("Model Evaluation\n")
cat("-----------------\n")

cat("RMSE:", round(rmse(split$y_test, pred),3),"\n")
cat("R2 :", round(r2_score(split$y_test, pred),3),"\n\n")


# ---- VISUALIZATION ----
png("actual_vs_pred.png",700,600)

plot(split$y_test, pred,
     col="blue",
     pch=19,
     xlab="Actual MSRP",
     ylab="Predicted MSRP",
     main="Actual vs Predicted")

abline(0,1,col="red",lwd=2)

dev.off()

cat("Saved plot → actual_vs_pred.png\n\n")


# ---- CUSTOM PREDICTION ----
cat("Predicting custom car...\n")

new_car <- data.frame(
  model_year = 2020,
  brand = "Toyota",
  type = "SUV",
  miles_per_gallon = 22,
  premium_version = 1,
  collection_car = 0
)

price <- pipe$predict(new_car)

cat("\nPredicted MSRP:", round(price,2),"\n")
