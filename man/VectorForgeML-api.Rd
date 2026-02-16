\name{VectorForgeML-api}
\alias{VectorForgeML-api}
\alias{LinearRegression}
\alias{StandardScaler}
\alias{MinMaxScaler}
\alias{LabelEncoder}
\alias{OneHotEncoder}
\alias{Pipeline}
\alias{ColumnTransformer}
\alias{fit_linear_model}
\alias{predict_linear_model}
\alias{drop_constant_columns}
\alias{train_test_split}
\alias{mse}
\alias{rmse}
\alias{r2_score}
\title{VectorForgeML Main API}
\description{Core modeling, preprocessing, and utility functions provided by VectorForgeML.}
\details{
The package provides a reference-class linear regression model, feature preprocessing helpers,
and pipeline composition utilities with an Rcpp backend.
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
\code{Pipeline}, \code{ColumnTransformer}, \code{train_test_split},
\code{mse}, \code{rmse}, and \code{r2_score}.
}
\keyword{models}
\keyword{utilities}
