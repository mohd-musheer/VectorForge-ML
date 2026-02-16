LogisticRegression <- setRefClass(
  "LogisticRegression",
  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- logreg_create()
    },

    fit=function(X,y){
      X <- as.matrix(X)
      y <- as.numeric(y)
      logreg_fit(ptr,X,y)
    },

    predict=function(X){
      X <- as.matrix(X)
      logreg_predict(ptr,X)
    },

    predict_proba=function(X){
      X <- as.matrix(X)
      logreg_predict_proba(ptr,X)
    }
  )
)
