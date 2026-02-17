library(VectorForgeML)
ls("package:VectorForgeML")
print('Regression')

df <- read.csv("inst/dataset/winequality.csv", sep=";")

y <- df$quality
X <- df
X$quality <- NULL

split <- train_test_split(X,y,0.2,42)

cat_cols <- names(X)[sapply(X,is.character)]
num_cols <- names(X)[!sapply(X,is.character)]

pre <- ColumnTransformer$new(
  num_cols=num_cols,
  cat_cols=cat_cols,
  num_pipeline=StandardScaler$new(),
  cat_pipeline=OneHotEncoder$new()
)

pipe <- Pipeline$new(list(
  pre,
  KNN$new(k=29, mode="classification")
))

pipe$fit(split$X_train,split$y_train)

pred <- pipe$predict(split$X_test)

cat("Accuracy:",accuracy_score(split$y_test,pred),"\n")