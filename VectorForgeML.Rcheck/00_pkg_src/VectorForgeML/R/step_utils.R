has_method <- function(obj, method) {
  if (length(method) != 1L || !nzchar(method)) {
    return(FALSE)
  }

  ref_methods <- tryCatch({
    ref <- methods::getRefClass(class(obj)[1L])
    as.character(ref$methods())
  }, error = function(e) character(0))
  if (method %in% ref_methods) {
    return(TRUE)
  }

  fn <- tryCatch({
    if (is.environment(obj)) {
      get0(method, envir = obj, inherits = FALSE)
    } else if (is.list(obj)) {
      obj[[method]]
    } else {
      NULL
    }
  }, error = function(e) NULL)

  is.function(fn)
}

is_transformer <- function(obj) {
  has_method(obj, "fit_transform") || (has_method(obj, "fit") && has_method(obj, "transform"))
}

is_estimator <- function(obj) {
  has_method(obj, "fit") && has_method(obj, "predict")
}
