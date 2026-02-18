#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace Rcpp;
using std::vector;

struct KNNModel {
  vector<double> X;
  vector<double> y;
  int n;
  int p;
  int k;
  int mode; // 0 = classification, 1 = regression
};


// [[Rcpp::export]]
SEXP knn_create(int k, int mode){
  XPtr<KNNModel> m(new KNNModel(), true);
  m->k = k;
  m->mode = mode;
  return m;
}


// [[Rcpp::export]]
void knn_fit(SEXP ptr, NumericMatrix X, NumericVector y){

  XPtr<KNNModel> m(ptr);

  m->n = X.nrow();
  m->p = X.ncol();

  m->X.assign(X.begin(), X.end());
  m->y.assign(y.begin(), y.end());
}


// distance squared
inline double dist2(const double* a, const double* b, int p){
  double s=0.0;
  for(int j=0;j<p;j++){
    double d = a[j]-b[j];
    s += d*d;
  }
  return s;
}


// [[Rcpp::export]]
NumericVector knn_predict(SEXP ptr, NumericMatrix X){

  XPtr<KNNModel> m(ptr);

  int ntest = X.nrow();
  int p = m->p;
  int k = m->k;

  NumericVector out(ntest);

  vector<double> dists(m->n);
  vector<int> idx(m->n);

  const double* train = m->X.data();
  const double* yy = m->y.data();

  for(int i=0;i<ntest;i++){

    const double* test = X.begin() + i*p;

    // compute distances
    for(int t=0;t<m->n;t++){
      dists[t] = dist2(test, train + t*p, p);
      idx[t] = t;
    }

    // partial sort k nearest
    std::partial_sort(
      idx.begin(),
      idx.begin()+k,
      idx.end(),
      [&](int a,int b){
        return dists[a] < dists[b];
      }
    );


    if(m->mode == 0){
      // classification → majority vote

      std::map<double,int> counts;

      for(int j=0;j<k;j++)
        counts[ yy[idx[j]] ]++;

      int bestCount=-1;
      double bestClass=0;

      for(auto &it : counts){
        if(it.second > bestCount){
          bestCount = it.second;
          bestClass = it.first;
        }
      }

      out[i] = bestClass;
    }
    else{
      // regression → mean
      double s=0;
      for(int j=0;j<k;j++)
        s += yy[idx[j]];

      out[i] = s/k;
    }
  }

  return out;
}
