library(VectorForgeML)
cat("Loading dataset...\n")
ls("package:VectorForgeML")
df <- read.csv("inst/dataset/stud_logis.csv")
cat("Rows:", nrow(df), "\n")

# remove id column
df$Student_ID <- NULL

# target
y <- df$End_Sem_Pass
df$End_Sem_Pass <- NULL

# matrix
X <- as.matrix(df)

# split
data <- train_test_split(X, y, seed=42)

# scale
scaler <- StandardScaler$new()
X_train <- scaler$fit_transform(data$X_train)
X_test  <- scaler$transform(data$X_test)

# train
model <- LogisticRegression$new()

start <- Sys.time()

model$fit(X_train, data$y_train)

end <- Sys.time()

# predict
pred <- model$predict(X_test)

cat("\nTrain Time:", as.numeric(end-start),"sec\n")

# accuracy
acc <- accuracy_score(pred,data$y_test)

cat("Accuracy:", acc, "\n")
