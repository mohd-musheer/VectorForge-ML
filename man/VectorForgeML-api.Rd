\name{VectorForgeML-api}
\alias{VectorForgeML-api}
\alias{LinearRegression}
\alias{StandardScaler}
\alias{MinMaxScaler}
\alias{LabelEncoder}
\alias{OneHotEncoder}
\alias{train_test_split}
\alias{mse}
\alias{rmse}
\alias{r2_score}
\alias{square_vec}
\alias{sum2}
\alias{cpp_sum_squares}
\alias{dot_product}
\alias{fastLm}
\title{VectorForgeML Main API}
\description{Core modeling, preprocessing, and utility functions provided by VectorForgeML.}
\details{
The package provides a reference-class linear regression model, feature preprocessing helpers,
and small numeric utility functions implemented with an Rcpp backend.
}
\section{Model class}{
\code{LinearRegression} supports \code{$fit(X, y)} and \code{$predict(X)}.
}
\section{Preprocessing classes}{
\code{StandardScaler}, \code{MinMaxScaler}, \code{LabelEncoder}, and \code{OneHotEncoder}
provide \code{fit}, \code{transform}, and \code{fit_transform} style workflows.
}
\section{Utilities}{
\code{train_test_split}, \code{mse}, \code{rmse}, \code{r2_score}, \code{square_vec},
\code{sum2}, \code{cpp_sum_squares}, \code{dot_product}, and \code{fastLm}.
}
\keyword{models}
\keyword{utilities}