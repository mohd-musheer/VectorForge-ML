library(VectorForgeML)

cat("Loading dataset...\n")
df <- read.csv("inst/dataset/heart_disease_b.csv")

cat("Rows:", nrow(df), "\n")


y <- df$target
X <- df
X$target <- NULL

split <- train_test_split(X, y, test_size=0.2, seed=42)

cat_cols <- names(X)[sapply(X, is.character)]
num_cols <- names(X)[!sapply(X, is.character)]

preprocessor <- ColumnTransformer$new(
  num_cols = num_cols,
  cat_cols = cat_cols,
  num_pipeline = StandardScaler$new(),
  cat_pipeline = OneHotEncoder$new()
)

pipe <- Pipeline$new(list(
  preprocessor,
  LogisticRegression$new()
))

start <- Sys.time()

pipe$fit(split$X_train, split$y_train)

end <- Sys.time()

pred <- pipe$predict(split$X_test)

cat("\nTrain Time:", as.numeric(end - start), "sec\n")

acc <- accuracy_score(split$y_test, pred)
cat("Accuracy:", acc, "\n")


new_patient <- data.frame(
  age=58, sex=0, cp=0, trestbps=100, chol=248, 
  fbs=0, restecg=0, thalach=122, exang=0, 
  oldpeak=1.0, slope=1, ca=0, thal=2
)

result <- pipe$predict(new_patient)
cat("\n Probability of Heart Disease:", result, "\n")
