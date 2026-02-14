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
      mean <<- colMeans(X)
      sd <<- apply(X,2,sd)
      sd[sd==0] <<- 1
    },

    transform=function(X){
      X <- as.matrix(X)
      scale(X, center=mean, scale=sd)
    },

    fit_transform=function(X){
      fit(X)
      transform(X)
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
