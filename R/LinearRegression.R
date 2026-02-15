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