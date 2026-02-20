#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <R_ext/BLAS.h>

using namespace Rcpp;
using std::vector;

struct LogRegModel {
  vector<double> coef;
  double intercept = 0.0;
  int p = 0;
};

// sigmoid
inline double sigmoid(double z){
  return 1.0 / (1.0 + std::exp(-z));
}

// [[Rcpp::export]]
SEXP logreg_create(){
  XPtr<LogRegModel> model(new LogRegModel(), true);
  return model;
}

// [[Rcpp::export]]
void logreg_fit(SEXP ptr,
                NumericMatrix X,
                NumericVector y,
                double lr = 0.1,
                int epochs = 100){

  XPtr<LogRegModel> m(ptr);

  int n = X.nrow();
  int p = X.ncol();

  m->coef.assign(p,0.0);
  m->p = p;
  m->intercept = 0.0;

  vector<double> pred(n);
  vector<double> grad(p);
  vector<double> z(n);
  vector<double> err(n);

  const char* transN = "N";
  const char* transT = "T";
  double alpha = 1.0;
  int inc = 1;

  for(int e=0;e<epochs;e++){

    // predictions
    std::fill(z.begin(), z.end(), m->intercept);
    double beta1 = 1.0;
    F77_CALL(dgemv)(transN, &n, &p, &alpha, X.begin(), &n, m->coef.data(), &inc, &beta1, z.data(), &inc FCONE);

    double grad_b = 0.0;
    for(int i=0;i<n;i++){
      pred[i] = sigmoid(z[i]);
      err[i] = pred[i] - y[i];
      grad_b += err[i];
    }

    // gradients
    double beta0 = 0.0;
    F77_CALL(dgemv)(transT, &n, &p, &alpha, X.begin(), &n, err.data(), &inc, &beta0, grad.data(), &inc FCONE);

    // update
    for(int j=0;j<p;j++)
      m->coef[j] -= lr * grad[j] / n;

    m->intercept -= lr * grad_b / n;
  }
}

// [[Rcpp::export]]
NumericVector logreg_predict(SEXP ptr, NumericMatrix X){

  XPtr<LogRegModel> m(ptr);
  int n = X.nrow();
  int p = X.ncol();

  NumericVector out(n);
  std::fill(out.begin(), out.end(), m->intercept);

  const char* transN = "N";
  double alpha = 1.0;
  double beta = 1.0;
  int inc = 1;

  F77_CALL(dgemv)(transN, &n, &p, &alpha, X.begin(), &n, m->coef.data(), &inc, &beta, out.begin(), &inc FCONE);

  for(int i=0;i<n;i++){
    out[i] = sigmoid(out[i]) > 0.5 ? 1 : 0;
  }

  return out;
}

// [[Rcpp::export]]
NumericVector logreg_predict_proba(SEXP ptr, NumericMatrix X){

  XPtr<LogRegModel> m(ptr);
  int n = X.nrow();
  int p = X.ncol();

  NumericVector out(n);
  std::fill(out.begin(), out.end(), m->intercept);

  const char* transN = "N";
  double alpha = 1.0;
  double beta = 1.0;
  int inc = 1;

  F77_CALL(dgemv)(transN, &n, &p, &alpha, X.begin(), &n, m->coef.data(), &inc, &beta, out.begin(), &inc FCONE);

  for(int i=0;i<n;i++){
    out[i] = sigmoid(out[i]);
  }

  return out;
}
