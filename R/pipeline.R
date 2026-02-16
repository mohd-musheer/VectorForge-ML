pipeline_has_method <- function(obj, method) {
  if (exists("has_method", mode = "function")) {
    return(has_method(obj, method))
  }

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

pipeline_is_transformer <- function(obj) {
  pipeline_has_method(obj, "fit_transform") ||
    (pipeline_has_method(obj, "fit") && pipeline_has_method(obj, "transform"))
}

pipeline_is_estimator <- function(obj) {
  pipeline_has_method(obj, "fit") && pipeline_has_method(obj, "predict")
}

Pipeline <- setRefClass(
  "Pipeline",

  fields=list(
    steps="list",
    trained="logical"
  ),

  methods=list(

    initialize=function(steps){
      if (!is.list(steps) || length(steps) == 0L) {
        stop("steps must be a non-empty list")
      }
      steps <<- steps
      trained <<- FALSE
    },

    fit=function(X,y){

      if (missing(y)) {
        stop("y is required to fit a pipeline with an estimator")
      }

      data <- X
      n_steps <- length(steps)
      found_estimator <- FALSE

      for(i in seq_len(n_steps)){
        step <- steps[[i]]

        if(pipeline_is_transformer(step)){
          if(pipeline_has_method(step,"fit_transform")){
            data <- step$fit_transform(data)
          } else {
            step$fit(data)
            data <- step$transform(data)
          }
          next
        }

        if(pipeline_is_estimator(step)){
          if (i != n_steps) {
            stop("Estimator step must be the last pipeline step")
          }
          step$fit(data,y)
          found_estimator <- TRUE
          break
        }

        stop(
          paste0(
            "Invalid pipeline step at index ",
            i,
            ": expected transformer (fit/transform) or estimator (fit/predict)"
          )
        )
      }

      if (!found_estimator) {
        stop("Pipeline requires a final estimator step with fit() and predict()")
      }

      trained <<- TRUE
      invisible(NULL)
    },

    predict=function(X){

      if(!trained)
        stop("Pipeline not trained")

      data <- X
      n_steps <- length(steps)

      for(i in seq_len(n_steps)){
        step <- steps[[i]]

        if(pipeline_is_transformer(step)){
          data <- step$transform(data)
          next
        }

        if(pipeline_is_estimator(step)){
          return(step$predict(data))
        }

        stop(
          paste0(
            "Invalid pipeline step at index ",
            i,
            ": expected transformer (fit/transform) or estimator (fit/predict)"
          )
        )
      }

      stop("No prediction step found")
    }
  )
)
