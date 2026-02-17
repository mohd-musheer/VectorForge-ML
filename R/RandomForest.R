RandomForest <- setRefClass(
  "RandomForest",
  fields=list(ptr="externalptr"),
  methods=list(

    initialize=function(ntrees=50,max_depth=6,mtry=NULL,mode="classification"){
      cls <- mode=="classification"
      ptr <<- rf_create(ntrees,max_depth,
                        if(is.null(mtry)) 3 else mtry,
                        cls)
    },

    fit=function(X,y){
      rf_fit(ptr,as.matrix(X),as.numeric(y))
    },

    predict=function(X){
      rf_predict(ptr,as.matrix(X))
    }
  )
)
