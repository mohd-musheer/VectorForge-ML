#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <R_ext/BLAS.h>

using namespace Rcpp;
using std::vector;

struct SoftmaxModel {
  vector<double> W;
  vector<double> b;
  int p = 0;
  int k = 0;
};

inline double log_sum_exp(const vector<double>& z){
  double m = *std::max_element(z.begin(), z.end());
  double s = 0.0;
  for(double v : z) s += std::exp(v - m);
  return m + std::log(s);
}

// [[Rcpp::export]]
SEXP softmax_create(){
  XPtr<SoftmaxModel> model(new SoftmaxModel(), true);
  return model;
}

// [[Rcpp::export]]
void softmax_fit(SEXP ptr,
                 NumericMatrix X,
                 IntegerVector y,
                 double lr = 0.1,
                 int epochs = 200){

  XPtr<SoftmaxModel> m(ptr);

  int n = X.nrow();
  int p = X.ncol();

  int k = max(y) + 1;

  m->p = p;
  m->k = k;

  m->W.assign(p*k, 0.0);
  m->b.assign(k, 0.0);

  vector<double> Z(n * k);
  vector<double> Err(n * k);
  const char* transN = "N";
  const char* transT = "T";
  double alpha = 1.0;
  double beta0 = 0.0;

  for(int e=0; e<epochs; e++){

    // Forward pass: Z = X * W
    F77_CALL(dgemm)(transN, transN, &n, &k, &p, &alpha, X.begin(), &n, m->W.data(), &p, &beta0, Z.data(), &n FCONE FCONE);

    std::fill(gradb.begin(), gradb.end(), 0.0);

    for(int i=0;i<n;i++){
      for(int c=0;c<k;c++){
        logits[c] = Z[i + c*n] + m->b[c];
      }

      double lse = log_sum_exp(logits);

      for(int c=0;c<k;c++){
        double prob = std::exp(logits[c] - lse);
        double err = prob - (y[i]==c);
        
        Err[i + c*n] = err;
        gradb[c] += err;
      }
    }

    // Backward pass for W: gradW = X^T * Err
    F77_CALL(dgemm)(transT, transN, &p, &k, &n, &alpha, X.begin(), &n, Err.data(), &n, &beta0, gradW.data(), &p FCONE FCONE);

    // update
    double scale = lr / n;

    for(size_t i=0;i<gradW.size();i++)
      m->W[i] -= scale * gradW[i];

    for(int c=0;c<k;c++)
      m->b[c] -= scale * gradb[c];
  }
}

// [[Rcpp::export]]
IntegerVector softmax_predict(SEXP ptr, NumericMatrix X){

  XPtr<SoftmaxModel> m(ptr);

  int n = X.nrow();
  int p = X.ncol();
  int k = m->k;

  IntegerVector out(n);
  vector<double> Z(n * k);

  const char* transN = "N";
  double alpha = 1.0;
  double beta = 0.0;

  F77_CALL(dgemm)(transN, transN, &n, &k, &p, &alpha, X.begin(), &n, m->W.data(), &p, &beta, Z.data(), &n FCONE FCONE);

  for(int i=0;i<n;i++){

    int best = 0;
    double bestv = -1e300;

    for(int c=0;c<k;c++){
      double z = Z[i + c*n] + m->b[c];
      
      if(z > bestv){
        bestv = z;
        best = c;
      }
    }

    out[i] = best;
  }

  return out;
}

// [[Rcpp::export]]
NumericMatrix softmax_predict_proba(SEXP ptr, NumericMatrix X){

  XPtr<SoftmaxModel> m(ptr);

  int n = X.nrow();
  int p = X.ncol();
  int k = m->k;

  NumericMatrix out(n,k);
  vector<double> Z(n * k);

  const char* transN = "N";
  double alpha = 1.0;
  double beta = 0.0;

  F77_CALL(dgemm)(transN, transN, &n, &k, &p, &alpha, X.begin(), &n, m->W.data(), &p, &beta, Z.data(), &n FCONE FCONE);

  vector<double> logits(k);

  for(int i=0;i<n;i++){

    for(int c=0;c<k;c++){
      logits[c] = Z[i + c*n] + m->b[c];
    }

    double lse = log_sum_exp(logits);

    for(int c=0;c<k;c++)
      out(i,c) = std::exp(logits[c] - lse);
  }

  return out;
}
