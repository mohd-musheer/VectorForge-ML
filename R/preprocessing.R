LinearRegression <- R6::R6Class("LinearRegression",

public = list(

coef=NULL,

fit=function(X,y){
    self$coef <- cpp_fit_predict(as.matrix(X),y)
},

predict=function(X){

    X <- as.matrix(X)
    X <- cbind(1,X)

    return(X %*% self$coef)
}

))
