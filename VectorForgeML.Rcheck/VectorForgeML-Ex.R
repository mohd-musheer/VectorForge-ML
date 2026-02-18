pkgname <- "VectorForgeML"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
library('VectorForgeML')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("LinearRegression-class")
### * LinearRegression-class

flush(stderr()); flush(stdout())

### Name: LinearRegression-class
### Title: Linear Regression Model
### Aliases: LinearRegression-class LinearRegression

### ** Examples

model <- LinearRegression$new()
X <- matrix(rnorm(100),50,2)
y <- rnorm(50)
model$fit(X,y)
model$predict(X)




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
