#include <Rcpp.h>
#include <R_ext/BLAS.h>

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
  const double dn = static_cast<double>(n);
  return dn * (dn + 1.0) * (2.0 * dn + 1.0) / 6.0;
}

// [[Rcpp::export]]
double dot_product(NumericVector a, NumericVector b) {
  if (a.size() != b.size()) {
    stop("Vectors must have the same length");
  }

  const int n = a.size();
  const int inc = 1;
  return F77_CALL(ddot)(&n, a.begin(), &inc, b.begin(), &inc);
}
