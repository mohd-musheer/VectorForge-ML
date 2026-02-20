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
  vector<double> Xc(n*p);
  
  for(int j=0;j<p;j++){
    for(int i=0;i<n;i++){
      Xc[j*n + i] = x[j*n + i] - m->mean[j];
    }
  }

  const char* uplo_syrk = "U";
  const char* trans_syrk = "T";
  double alpha = 1.0 / (n - 1);
  double beta = 0.0;

  F77_CALL(dsyrk)(uplo_syrk, trans_syrk, &p, &n, &alpha, Xc.data(), &n, &beta, C.data(), &p FCONE FCONE);

  // fill lower triangle
  for(int i=0;i<p;i++){
    for(int j=i+1;j<p;j++){
      C[j*p + i] = C[i*p + j];
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

  vector<double> Xc(n*p);
  for(int j=0;j<p;j++){
    for(int i=0;i<n;i++){
      Xc[j*n + i] = x[j*n + i] - m->mean[j];
    }
  }

  const char* transN = "N";
  double alpha = 1.0;
  double beta = 0.0;

  F77_CALL(dgemm)(transN, transN, &n, &k, &p, &alpha, Xc.data(), &n, m->components.data(), &p, &beta, o, &n FCONE FCONE);

  return out;
}
