# library(VectorForgeML)
# source("R/LinearRegression.R")
# source("R/split.R")
# source("R/scalers.R")
# source("R/encoders.R")
# ls("package:VectorForgeML")
# source("R/pipeline.R")
# source("R/column_transformer.R")

library(VectorForgeML)
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
