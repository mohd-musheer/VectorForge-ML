.onLoad <- function(libname, pkgname) {
  if (exists("cpp_set_blas_threads", mode = "function", inherits = TRUE)) {
    try(cpp_set_blas_threads(), silent = TRUE)
  }
}