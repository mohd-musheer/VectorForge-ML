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

# =========================
# CLASSIFICATION METRICS
# =========================

accuracy_score <- function(y_true, y_pred){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)
  mean(y_true == y_pred, na.rm = TRUE)
}


precision_score <- function(y_true, y_pred, positive = NULL){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)

  if(is.null(positive))
    positive <- unique(y_true)[1]

  tp <- sum(y_true == positive & y_pred == positive)
  fp <- sum(y_true != positive & y_pred == positive)

  if(tp + fp == 0) return(0)
  tp/(tp+fp)
}


recall_score <- function(y_true, y_pred, positive = NULL){
  y_true <- as.vector(y_true)
  y_pred <- as.vector(y_pred)

  if(is.null(positive))
    positive <- unique(y_true)[1]

  tp <- sum(y_true == positive & y_pred == positive)
  fn <- sum(y_true == positive & y_pred != positive)

  if(tp + fn == 0) return(0)
  tp/(tp+fn)
}


f1_score <- function(y_true, y_pred, positive = NULL){
  p <- precision_score(y_true, y_pred, positive)
  r <- recall_score(y_true, y_pred, positive)

  if(p+r == 0) return(0)
  2*p*r/(p+r)
}


# =========================
# MULTICLASS MACRO METRICS
# =========================

macro_precision <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    precision_score(y_true, y_pred, cls)))
}

macro_recall <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    recall_score(y_true, y_pred, cls)))
}

macro_f1 <- function(y_true, y_pred){
  classes <- unique(y_true)
  mean(sapply(classes, function(cls)
    f1_score(y_true, y_pred, cls)))
}
