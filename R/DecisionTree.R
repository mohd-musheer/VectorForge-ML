DecisionTree <- setRefClass(
  "DecisionTree",
  fields=list(ptr="externalptr"),
  methods=list(
    initialize=function(max_depth=5){
      ptr <<- dt_create(max_depth)
    },
    fit=function(X,y){
      dt_fit(ptr,as.matrix(X),as.numeric(y))
    },
    predict=function(X){
      dt_predict(ptr,as.matrix(X))
    }
  )
)
