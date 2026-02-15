fit_linear_model <- function(X, y) {
  model <- LinearRegression$new()
  model$fit(X, y)
  model
}

predict_linear_model <- function(model, X) {
  if (!methods::is(model, "LinearRegression")) {
    stop("model must be a LinearRegression object")
  }
  model$predict(X)
}
