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

  vector<double> logits(k);
  vector<double> probs(k);
  vector<double> gradW(p*k);
  vector<double> gradb(k);

  for(int e=0; e<epochs; e++){

    std::fill(gradW.begin(), gradW.end(), 0.0);
    std::fill(gradb.begin(), gradb.end(), 0.0);

    for(int i=0;i<n;i++){

      // logits
      for(int c=0;c<k;c++){
        double z = m->b[c];
        for(int j=0;j<p;j++)
          z += X(i,j) * m->W[j + c*p];
        logits[c] = z;
      }

      double lse = log_sum_exp(logits);

      // probs
      for(int c=0;c<k;c++)
        probs[c] = std::exp(logits[c] - lse);

      // gradients
      for(int c=0;c<k;c++){
        double err = probs[c] - (y[i]==c);

        gradb[c] += err;

        for(int j=0;j<p;j++)
          gradW[j + c*p] += err * X(i,j);
      }
    }

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

  for(int i=0;i<n;i++){

    int best = 0;
    double bestv = -1e300;

    for(int c=0;c<k;c++){
      double z = m->b[c];
      for(int j=0;j<p;j++)
        z += X(i,j) * m->W[j + c*p];

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
  vector<double> logits(k);

  for(int i=0;i<n;i++){

    for(int c=0;c<k;c++){
      double z = m->b[c];
      for(int j=0;j<p;j++)
        z += X(i,j) * m->W[j + c*p];
      logits[c] = z;
    }

    double lse = log_sum_exp(logits);

    for(int c=0;c<k;c++)
      out(i,c) = std::exp(logits[c] - lse);
  }

  return out;
}
