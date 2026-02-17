#' K Nearest Neighbors
#'
#' Fast KNN model
#'
#' @export
KNN <- setRefClass(
  "KNN",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(k=5, mode="classification"){

      m <- if(mode=="classification") 0 else 1
      ptr <<- knn_create(k,m)
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.numeric(y)
      knn_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      knn_predict(ptr,X)
    }
  )
)
