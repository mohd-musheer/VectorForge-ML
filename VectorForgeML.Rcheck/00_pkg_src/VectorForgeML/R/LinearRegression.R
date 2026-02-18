#' Linear Regression Model
#'
#' Fast linear regression implemented in C++ backend.
#'
#' @return LinearRegression object
#'
#' @examples
#' model <- LinearRegression$new()
#' X <- matrix(rnorm(100),50,2)
#' y <- rnorm(50)
#' model$fit(X,y)
#' model$predict(X)
#'
#' @export
LinearRegression <- setRefClass(
  "LinearRegression",
  fields = list(ptr = "externalptr"),
  methods = list(
    initialize = function() {
      ptr <<- lr_create()
    },
    fit = function(X, y) {
      X <- as.matrix(X)
      y <- as.numeric(y)
      lr_fit(ptr, X, y)
      invisible(NULL)
    },
    predict = function(X) {
      X <- as.matrix(X)
      lr_predict(ptr, X)
    }
  )
)