#' Softmax Regression Model
#'
#' Multiclass logistic regression.
#'
#' @return SoftmaxRegression object
#' @export
SoftmaxRegression <- setRefClass(
  "SoftmaxRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- softmax_create()
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.integer(y)
      softmax_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      softmax_predict(ptr,X)
    },

    predict_proba=function(X){
      X <- as.matrix(X)
      softmax_predict_proba(ptr,X)
    }
  )
)
