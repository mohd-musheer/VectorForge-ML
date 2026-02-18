# =========================
# LABEL ENCODER
# =========================

#' Label Encoder
#'
#' Converts categorical labels into numeric values.
#'
#' @return LabelEncoder object
#' @export
LabelEncoder <- setRefClass(
  "LabelEncoder",
  fields=list(map="list"),

  methods=list(

    fit=function(x){
      vals <- unique(x)
      map <<- setNames(seq_along(vals)-1, vals)
    },

    transform=function(x){
      as.numeric(map[x])
    },

    fit_transform=function(x){
      fit(x)
      transform(x)
    }

  )
)


# =========================
# ONE HOT ENCODER
# =========================

#' One Hot Encoder
#'
#' Converts categorical variables into binary vectors.
#'
#' @return OneHotEncoder object
#' @export
OneHotEncoder <- setRefClass(
  "OneHotEncoder",
  fields=list(categories="list"),

  methods=list(

    fit=function(df){
      if (!is.data.frame(df)) {
        stop("OneHotEncoder$fit expects a data.frame")
      }

      categories <<- lapply(df, function(col)
        unique(as.character(col))
      )

      invisible(NULL)
    },

    transform=function(df){
      if (!is.data.frame(df)) {
        stop("OneHotEncoder$transform expects a data.frame")
      }

      n <- nrow(df)
      if (n == 0) {
        return(matrix(numeric(0), nrow=0, ncol=0))
      }

      widths <- vapply(categories, length, integer(1))
      total_cols <- sum(widths)
      out <- matrix(0L, nrow=n, ncol=total_cols)
      out_names <- character(total_cols)

      col_start <- 1L
      for(colname in names(df)){
        col <- as.character(df[[colname]])
        cats <- categories[[colname]]
        k <- length(cats)
        if (k == 0) {
          next
        }

        idx <- match(col, cats)
        valid <- !is.na(idx)
        row_ids <- which(valid)
        col_ids <- col_start + idx[valid] - 1L
        out[cbind(row_ids, col_ids)] <- 1L

        out_names[col_start:(col_start+k-1L)] <- paste(colname, cats, sep="_")
        col_start <- col_start + k
      }

      colnames(out) <- out_names
      storage.mode(out) <- "double"
      out
    },

    fit_transform=function(df){
      fit(df)
      transform(df)
    }

  )
)
