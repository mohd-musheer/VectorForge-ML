\name{VectorForgeML-api}
\alias{VectorForgeML-api}
\alias{LinearRegression}
\alias{StandardScaler}
\alias{MinMaxScaler}
\alias{LabelEncoder}
\alias{OneHotEncoder}
\alias{fit_linear_model}
\alias{predict_linear_model}
\alias{drop_constant_columns}
\alias{train_test_split}
\alias{mse}
\alias{rmse}
\alias{r2_score}
\alias{square_vec}
\alias{sum2}
\alias{cpp_sum_squares}
\alias{dot_product}
\alias{fastLm}
\alias{cpp_drop_constant_cols}
\title{VectorForgeML Main API}
\description{Core modeling, preprocessing, and utility functions provided by VectorForgeML.}
\details{
The package provides a reference-class linear regression model, feature preprocessing helpers,
and numeric utility functions implemented with an Rcpp backend.
}
\section{Model API}{
\code{LinearRegression} supports \code{$fit(X, y)} and \code{$predict(X)}.
\code{fit_linear_model()} and \code{predict_linear_model()} are convenience wrappers.
}
\section{Preprocessing API}{
\code{StandardScaler}, \code{MinMaxScaler}, \code{LabelEncoder}, \code{OneHotEncoder},
and \code{drop_constant_columns()} (C++-backed).
}
\section{Utilities}{
\code{train_test_split}, \code{mse}, \code{rmse}, \code{r2_score}, \code{square_vec},
\code{sum2}, \code{cpp_sum_squares}, \code{dot_product}, \code{fastLm},
and \code{cpp_drop_constant_cols}.
}
\keyword{models}
\keyword{utilities}
