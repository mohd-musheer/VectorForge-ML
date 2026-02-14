library(Rcpp)

Sys.setenv(
  PKG_CXXFLAGS="-O3 -march=native -funroll-loops -ffast-math -flto",
  PKG_LIBS="-lRblas -lRlapack"
)


sourceCpp("src/LinearRegression.cpp")

source("R/LinearRegression.R")
source("R/split.R")
source("R/scalers.R")
source("R/encoders.R")

cat("Loading dataset...\n")

df <- read.csv("dataset/cars.csv", stringsAsFactors=FALSE)

# enlarge dataset (simulate big data)
df <- df[rep(1:nrow(df), 10), ]

cat("Rows:", nrow(df), "\n")


# target
y <- df$msrp
df$msrp <- NULL


# split types
cat_cols <- sapply(df, is.character)

cat_df <- df[, cat_cols, drop=FALSE]
num_df <- df[, !cat_cols, drop=FALSE]


# start timing
start <- Sys.time()


# encoding
encoder <- OneHotEncoder$new()
cat_encoded <- encoder$fit_transform(cat_df)

# combine
X <- cbind(as.matrix(num_df), cat_encoded)

# remove constant columns
if (exists("cpp_drop_constant_cols", mode="function")) {
  X <- cpp_drop_constant_cols(X)$X
} else {
  X <- X[, apply(X,2,var)!=0]
}

# split
data <- train_test_split(X,y, seed=42)

# scale
scaler <- StandardScaler$new()
X_train <- scaler$fit_transform(data$X_train)

# train
model <- LinearRegression$new()
model$fit(X_train, data$y_train)

end <- Sys.time()

cat("\nYour Framework Time:",
    as.numeric(end-start),"seconds\n")
