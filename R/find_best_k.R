find_best_k <- function(X,y,k_values=seq(1,15,2)){

  split <- train_test_split(X,y,0.2,42)

  best_k <- k_values[1]
  best_score <- -Inf

  for(k in k_values){

    model <- KNN$new(k=k,mode="classification")

    model$fit(split$X_train,split$y_train)

    pred <- model$predict(split$X_test)

    score <- accuracy_score(split$y_test,pred)

    if(score > best_score){
      best_score <- score
      best_k <- k
    }
  }

  best_k
}
