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

  for(int e=0;e<epochs;e++){

    // predictions
    for(int i=0;i<n;i++){
      double z = m->intercept;
      for(int j=0;j<p;j++)
        z += X(i,j)*m->coef[j];

      pred[i] = sigmoid(z);
    }

    // gradients
    std::fill(grad.begin(), grad.end(), 0.0);
    double grad_b = 0.0;

    for(int i=0;i<n;i++){
      double err = pred[i] - y[i];
      grad_b += err;

      for(int j=0;j<p;j++)
        grad[j] += err * X(i,j);
    }

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

  for(int i=0;i<n;i++){
    double z = m->intercept;
    for(int j=0;j<p;j++)
      z += X(i,j)*m->coef[j];

    out[i] = sigmoid(z) > 0.5 ? 1 : 0;
  }

  return out;
}

// [[Rcpp::export]]
NumericVector logreg_predict_proba(SEXP ptr, NumericMatrix X){

  XPtr<LogRegModel> m(ptr);
  int n = X.nrow();
  int p = X.ncol();

  NumericVector out(n);

  for(int i=0;i<n;i++){
    double z = m->intercept;
    for(int j=0;j<p;j++)
      z += X(i,j)*m->coef[j];

    out[i] = sigmoid(z);
  }

  return out;
}
