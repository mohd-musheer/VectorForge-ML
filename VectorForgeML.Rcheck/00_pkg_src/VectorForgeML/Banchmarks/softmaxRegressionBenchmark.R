library(VectorForgeML)

df <- read.csv("inst/dataset/winequality.csv", sep=";")

y <- df$quality
X <- df
X$quality <- NULL

split <- train_test_split(X,y,0.2,42)

pipe <- Pipeline$new(list(
  StandardScaler$new(),
  SoftmaxRegression$new()
))

start <- Sys.time()

pipe$fit(split$X_train, split$y_train)

end <- Sys.time()

pred <- pipe$predict(split$X_test)

cat("Train time:", end-start,"\n")
cat("Accuracy:", accuracy_score(split$y_test,pred),"\n")

cm <- confusion_matrix(split$y_test, pred)
png("cm.png",700,600)
plot_confusion_matrix(cm)
dev.off()

new_sample <- data.frame(
  "fixed acidity" = 7.4,
  "volatile acidity" = 0.7,
  "citric acid" = 0,
  "residual sugar" = 1.9,
  "chlorides" = 0.076,
  "free sulfur dioxide" = 11,
  "total sulfur dioxide" = 34,
  "density" = 0.9978,
  "pH" = 3.51,
  "sulphates" = 0.56,
  "alcohol" = 9.4
)

output=pipe$predict(new_sample)
cat('output : ',output)
probablity=pipe$predict_proba(new_sample)
cat('\nprobablity : ',probablity)













# mullti_data <- data.frame(
#   "fixed acidity" = c(7.4, 7.8),
#   "volatile acidity" = c(0.7, 0.88),
#   "citric acid" = c(0,0),
#   "residual sugar" = c(1.9,2.6),
#   "chlorides" = c(0.076,0.098),
#   "free sulfur dioxide" = c(11,25),
#   "total sulfur dioxide" = c(34,67),
#   "density" = c(0.9978,0.9968),
#   "pH" = c(3.51,3.2),
#   "sulphates" = c(0.56,0.68),
#   "alcohol" = c(9.4,9.8)
# )

# output=pipe$predict(mullti_data)
# cat('output : ',output)