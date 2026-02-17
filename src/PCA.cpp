#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>

using namespace Rcpp;
using std::vector;

struct PCAModel {
  vector<double> components;
  vector<double> mean;
  int n_features;
  int n_components;
};

// [[Rcpp::export]]
SEXP pca_create(int ncomp){
  XPtr<PCAModel> m(new PCAModel(), true);
  m->n_components = ncomp;
  return m;
}

// [[Rcpp::export]]
void pca_fit(SEXP ptr, NumericMatrix X){

  XPtr<PCAModel> m(ptr);

  const int n = X.nrow();
  const int p = X.ncol();

  m->n_features = p;
  m->mean.assign(p,0.0);

  const double* x = X.begin();

  // ===== compute column means =====
  for(int j=0;j<p;j++){
    double s=0;
    for(int i=0;i<n;i++)
      s += x[j*n + i];
    m->mean[j] = s / n;
  }

  // ===== covariance matrix =====
  vector<double> C(p*p,0.0);

  for(int i=0;i<p;i++){
    for(int j=i;j<p;j++){

      double s=0;

      for(int k=0;k<n;k++){
        double xi = x[i*n + k] - m->mean[i];
        double xj = x[j*n + k] - m->mean[j];
        s += xi * xj;
      }

      double v = s/(n-1);
      C[i*p + j] = v;
      C[j*p + i] = v;
    }
  }

  // ===== eigen decomposition =====
  vector<double> eigvals(p);
  vector<double> eigvecs = C;

  char jobz='V';
  char uplo='U';
  int info;
  int lwork = -1;
  double wkopt;

  // workspace query
  F77_CALL(dsyev)(
    &jobz,&uplo,&p,
    eigvecs.data(),&p,
    eigvals.data(),
    &wkopt,&lwork,&info
    FCONE FCONE
  );

  lwork = (int)wkopt;
  vector<double> work(lwork);

  // actual solve
  F77_CALL(dsyev)(
    &jobz,&uplo,&p,
    eigvecs.data(),&p,
    eigvals.data(),
    work.data(),&lwork,&info
    FCONE FCONE
  );

  if(info != 0)
    stop("Eigen decomposition failed");

  // ===== take top components =====
  const int k = m->n_components;
  m->components.resize(p*k);

  for(int c=0;c<k;c++){
    int eig_col = p-1-c;   // largest eigenvalues last
    for(int r=0;r<p;r++)
      m->components[c*p + r] = eigvecs[eig_col*p + r];
  }
}

// [[Rcpp::export]]
NumericMatrix pca_transform(SEXP ptr, NumericMatrix X){

  XPtr<PCAModel> m(ptr);

  const int n = X.nrow();
  const int p = m->n_features;
  const int k = m->n_components;

  NumericMatrix out(n,k);

  const double* x = X.begin();
  double* o = out.begin();

  for(int i=0;i<n;i++){
    for(int c=0;c<k;c++){

      double s=0;
      const double* comp = &m->components[c*p];

      for(int j=0;j<p;j++)
        s += (x[j*n + i] - m->mean[j]) * comp[j];

      o[c*n + i] = s;
    }
  }

  return out;
}
