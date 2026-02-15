drop_constant_columns <- function(X, eps = 1e-12) {
  X <- as.matrix(X)
  cpp_drop_constant_cols(X, eps = eps)
}
