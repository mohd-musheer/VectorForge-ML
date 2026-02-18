predict_model <- function(model, newdata, prob = FALSE){

  if(!methods::is(model,"Pipeline"))
    stop("model must be Pipeline")

  X <- newdata

  for(step in model$steps){

    if(has_method(step,"transform")){
      X <- step$transform(X)
      next
    }

    if(has_method(step,"predict")){
      if(prob && has_method(step,"predict_proba"))
        return(step$predict_proba(X))

      return(step$predict(X))
    }
  }
}
