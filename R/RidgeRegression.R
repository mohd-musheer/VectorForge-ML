RidgeRegression <- setRefClass(
  "RidgeRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- ridge_create()
    },

    fit=function(X,y,lambda=1.0){
      X <- as.matrix(X)
      y <- as.numeric(y)
      ridge_fit(ptr,X,y,lambda)
    },

    predict=function(X){
      X <- as.matrix(X)
      ridge_predict(ptr,X)
    }
  )
)
