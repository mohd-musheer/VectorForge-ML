column_transformer_has_method <- function(obj, method) {
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

as_feature_matrix <- function(x) {
  if (is.matrix(x)) {
    return(x)
  }
  if (is.data.frame(x)) {
    return(as.matrix(x))
  }
  if (is.vector(x)) {
    return(matrix(x, ncol = 1L))
  }
  as.matrix(x)
}

validate_df_columns <- function(df, columns, label) {
  if (!is.data.frame(df)) {
    stop("ColumnTransformer requires a data.frame input")
  }

  if (length(columns) == 0L) {
    return(invisible(NULL))
  }

  missing_cols <- setdiff(columns, names(df))
  if (length(missing_cols) > 0L) {
    stop(paste0(label, " columns not found: ", paste(missing_cols, collapse = ", ")))
  }
}

ColumnTransformer <- setRefClass(
  "ColumnTransformer",

  fields=list(
    num_cols="character",
    cat_cols="character",
    num_pipeline="ANY",
    cat_pipeline="ANY",
    trained="logical"
  ),

  methods=list(

    initialize=function(num_cols, cat_cols,
                        num_pipeline=NULL,
                        cat_pipeline=NULL){

      num_cols <<- num_cols
      cat_cols <<- cat_cols
      num_pipeline <<- num_pipeline
      cat_pipeline <<- cat_pipeline
      trained <<- FALSE

      if (length(intersect(num_cols, cat_cols)) > 0L) {
        stop("num_cols and cat_cols must be disjoint")
      }
    },

    fit=function(df, y = NULL){
      validate_df_columns(df, num_cols, "Numeric")
      validate_df_columns(df, cat_cols, "Categorical")

      if(!is.null(num_pipeline) && length(num_cols) > 0L)
        num_pipeline$fit(df[,num_cols,drop=FALSE])

      if(!is.null(cat_pipeline) && length(cat_cols) > 0L)
        cat_pipeline$fit(df[,cat_cols,drop=FALSE])

      trained <<- TRUE
      invisible(NULL)
    },

    transform=function(df){
      if (!trained) {
        stop("ColumnTransformer not trained")
      }

      validate_df_columns(df, num_cols, "Numeric")
      validate_df_columns(df, cat_cols, "Categorical")

      parts <- list()

      if(!is.null(num_pipeline) && length(num_cols) > 0L){
        if (!column_transformer_has_method(num_pipeline, "transform")) {
          stop("num_pipeline must implement transform()")
        }
        parts[[length(parts)+1]] <-
          as_feature_matrix(num_pipeline$transform(df[,num_cols,drop=FALSE]))
      }

      if(!is.null(cat_pipeline) && length(cat_cols) > 0L){
        if (!column_transformer_has_method(cat_pipeline, "transform")) {
          stop("cat_pipeline must implement transform()")
        }
        parts[[length(parts)+1]] <-
          as_feature_matrix(cat_pipeline$transform(df[,cat_cols,drop=FALSE]))
      }

      if (length(parts) == 0L) {
        return(matrix(numeric(0), nrow = nrow(df), ncol = 0L))
      }

      out <- do.call(cbind, parts)
      storage.mode(out) <- "double"
      out
    },

    fit_transform=function(df){
      fit(df)
      transform(df)
    }
  )
)
