LinearRegression <- setRefClass(
  "LinearRegression",

  fields=list(ptr="externalptr"),

  methods=list(

    initialize=function(){
      ptr <<- lr_create()
    },

    fit=function(X,y){
      lr_fit(ptr,X,y)
    },

    predict=function(X){
      lr_predict(ptr,X)
    }

  )
)
