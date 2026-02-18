#' Principal Component Analysis
#'
#' Dimensionality reduction technique.
#'
#' @return PCA object
#' @export
PCA <- setRefClass(
  "PCA",
  fields=list(ptr="externalptr", ncomp="numeric"),
  methods=list(
    initialize=function(n_components=2){
      ncomp <<- n_components
      ptr <<- pca_create(n_components)
    },
    fit=function(X){
      pca_fit(ptr,as.matrix(X))
    },
    transform=function(X){
      pca_transform(ptr,as.matrix(X))
    },
    fit_transform=function(X){
      fit(X)
      transform(X)
    }
  )
)
