# =========================
# STANDARD SCALER
# =========================

StandardScaler <- setRefClass(
  "StandardScaler",
  fields=list(
    mean="numeric",
    sd="numeric"
  ),

  methods=list(

    fit=function(X){
      X <- as.matrix(X)
      if (exists("cpp_scale_fit_transform", mode="function")) {
        out <- cpp_scale_fit_transform(X)
        mean <<- out$mean
        sd <<- out$sd
      } else {
        mean <<- colMeans(X)
        sd <<- apply(X,2,sd)
        sd[sd==0] <<- 1
      }
    },

    transform=function(X){
      X <- as.matrix(X)
      if (exists("cpp_scale_transform", mode="function")) {
        cpp_scale_transform(X, mean, sd)
      } else {
        scale(X, center=mean, scale=sd)
      }
    },

    fit_transform=function(X){
      X <- as.matrix(X)
      if (exists("cpp_scale_fit_transform", mode="function")) {
        out <- cpp_scale_fit_transform(X)
        mean <<- out$mean
        sd <<- out$sd
        out$X
      } else {
        fit(X)
        transform(X)
      }
    }

  )
)


# =========================
# MINMAX SCALER
# =========================

MinMaxScaler <- setRefClass(
  "MinMaxScaler",
  fields=list(
    minv="numeric",
    maxv="numeric"
  ),

  methods=list(

    fit=function(X){
      X <- as.matrix(X)
      minv <<- apply(X,2,min)
      maxv <<- apply(X,2,max)
    },

    transform=function(X){
      X <- as.matrix(X)
      (X - minv)/(maxv - minv + 1e-8)
    },

    fit_transform=function(X){
      fit(X)
      transform(X)
    }

  )
)
