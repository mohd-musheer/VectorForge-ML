library(VectorForgeML)
# ls("package:VectorForgeML")
print('classification : winequality')

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
k=find_best_k(X,y)
cat('best K : ',k,'\n')
cat('Type of K',class(k),'\n')
pipe <- Pipeline$new(list(
  pre,
  KNN$new(k, mode="classification")
))

pipe$fit(split$X_train,split$y_train)

pred <- pipe$predict(split$X_test)

cat("Accuracy:",accuracy_score(split$y_test,pred),"\n")
#-------------------------------------------------------------------------------------------------------------------------

print('Classification : titanic\n')
library(VectorForgeML)

df <- read.csv("inst/dataset/titanic.csv")

df$Name <- NULL
df$Ticket <- NULL
df$Cabin <- NULL

y <- df$Survived
X <- df
X$Survived <- NULL

split <- train_test_split(X,y,0.2,42)

cat_cols <- names(X)[sapply(X,is.character)]
num_cols <- names(X)[!sapply(X,is.character)]

pre <- ColumnTransformer$new(
  num_cols=num_cols,
  cat_cols=cat_cols,
  num_pipeline=StandardScaler$new(),
  cat_pipeline=OneHotEncoder$new()
)
X_proc <- pre$fit_transform(X)
k <- find_best_k(X_proc,y)
cat('best K : ',k,'\n')
pipe <- Pipeline$new(list(
  pre,
  KNN$new(k, mode="classification")
))

pipe$fit(split$X_train,split$y_train)

pred <- pipe$predict(split$X_test)

cat("Accuracy:",accuracy_score(split$y_test,pred),"\n")
#---------------------------------------------------------------------------------------------------------------------------------
print('regression : cars\n')

library(VectorForgeML)

df <- read.csv("inst/dataset/cars.csv")

y <- df$msrp
X <- df
X$msrp <- NULL

split <- train_test_split(X,y,0.2,42)

cat_cols <- names(X)[sapply(X,is.character)]
num_cols <- names(X)[!sapply(X,is.character)]

pre <- ColumnTransformer$new(
  num_cols=num_cols,
  cat_cols=cat_cols,
  num_pipeline=StandardScaler$new(),
  cat_pipeline=OneHotEncoder$new()
)
X_proc <- pre$fit_transform(X)
k <- find_best_k(X_proc,y)
cat('best K : ',k,'\n')
pipe <- Pipeline$new(list(
  pre,
  KNN$new(k, mode="regression")
))

pipe$fit(split$X_train,split$y_train)

pred <- pipe$predict(split$X_test)

cat("RMSE:",rmse(split$y_test,pred),"\n")
cat("R2:",r2_score(split$y_test,pred),"\n")
