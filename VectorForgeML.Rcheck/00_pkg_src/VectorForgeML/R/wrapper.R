#' Fit Linear Model (Fast C++ backend)
#'
#' Internal helper for linear regression training.
#'
#' @param X numeric matrix
#' @param y numeric vector
#'
#' @return model object
#' @export
fit_linear_model <- function(X, y) {
  model <- LinearRegression$new()
  model$fit(X, y)
  model
}
#' Predict Linear Model
#'
#' Predict values using trained linear model.
#'
#' @param model trained model
#' @param X matrix
#'
#' @return numeric vector
#' @export
predict_linear_model <- function(model, X) {
  if (!methods::is(model, "LinearRegression")) {
    stop("model must be a LinearRegression object")
  }
  model$predict(X)
}
