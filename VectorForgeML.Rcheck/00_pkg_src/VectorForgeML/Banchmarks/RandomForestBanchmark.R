library(VectorForgeML)

df <- read.csv("inst/dataset/winequality.csv",sep=";")

y <- df$quality
X <- df; X$quality<-NULL

split <- train_test_split(X,y,0.2,42)

cat_cols <- names(X)[sapply(X,is.character)]
num_cols <- names(X)[!sapply(X,is.character)]

pre <- ColumnTransformer$new(
  num_cols,cat_cols,
  StandardScaler$new(),
  OneHotEncoder$new()
)

pipe <- Pipeline$new(list(
  pre,
  RandomForest$new(ntrees=100,max_depth=7,4,mode="classification")
))

pipe$fit(split$X_train,split$y_train)
pred <- pipe$predict(split$X_test)

cat("Accuracy:",accuracy_score(split$y_test,round(pred)),"\n")
