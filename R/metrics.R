mse <- function(y_true,y_pred){
  mean((y_true-y_pred)^2, na.rm=TRUE)
}

rmse <- function(y_true,y_pred){
  sqrt(mse(y_true,y_pred))
}


r2_score <- function(y_true,y_pred){

  y_true <- y_true[!is.na(y_pred)]
  y_pred <- y_pred[!is.na(y_pred)]

  ss_res <- sum((y_true-y_pred)^2)
  ss_tot <- sum((y_true-mean(y_true))^2)

  if(is.na(ss_tot) || ss_tot==0)
    return(1)

  1 - ss_res/ss_tot
}

