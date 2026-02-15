#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector square_vec(NumericVector x) {
  return x * x;
}

// [[Rcpp::export]]
int sum2(int x, int y) {
  return x + y;
}

// [[Rcpp::export]]
double cpp_sum_squares(int n) {
  if (n <= 0) {
    return 0.0;
  }
  double s = 0.0;
  for (int i = 1; i <= n; ++i) {
    s += static_cast<double>(i) * static_cast<double>(i);
  }
  return s;
}

// [[Rcpp::export]]
double dot_product(NumericVector a, NumericVector b) {
  if (a.size() != b.size()) {
    stop("Vectors must have the same length");
  }

  const int n = a.size();
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}