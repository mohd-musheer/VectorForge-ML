#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

using namespace Rcpp;
using std::vector;

struct RidgeModel {
  vector<double> coef;
  double intercept = 0.0;
  int p = 0;
};

inline void checkDims(const NumericMatrix& X, const NumericVector& y){
  if(X.nrow() != y.size())
    stop("X and y must have same rows");
}


// [[Rcpp::export]]
SEXP ridge_create(){
  XPtr<RidgeModel> model(new RidgeModel(), true);
  return model;
}


// [[Rcpp::export]]
void ridge_fit(SEXP ptr,
               NumericMatrix X,
               NumericVector y,
               double lambda = 1.0){

  XPtr<RidgeModel> m(ptr);

  checkDims(X,y);

  int n = X.nrow();
  int p = X.ncol();

  m->p = p;
  m->coef.assign(p,0.0);

  vector<double> XtX(p*p,0.0);
  vector<double> Xty(p,0.0);

  const char* trans="T";
  const char* notrans="N";
  double one=1.0;
  double zero=0.0;
  int inc=1;

  // ---- XtX = X'X ----
  F77_CALL(dgemm)(
    trans, notrans,
    &p, &p, &n,
    &one,
    X.begin(), &n,
    X.begin(), &n,
    &zero,
    XtX.data(), &p
    FCONE FCONE
  );

  // ---- Xty = X'y ----
  F77_CALL(dgemv)(
    trans,
    &n, &p,
    &one,
    X.begin(), &n,
    y.begin(), &inc,
    &zero,
    Xty.data(), &inc
    FCONE
  );

  // ---- Ridge penalty ----
  for(int j=0;j<p;j++)
    XtX[j + j*p] += lambda;


  // ---- Solve (XtX) β = Xty ----
  int nrhs=1;
  int lda=p;
  int ldb=p;
  int info=0;
  const char* uplo="L";

  F77_CALL(dposv)(
    uplo, &p, &nrhs,
    XtX.data(), &lda,
    Xty.data(), &ldb,
    &info
    FCONE
  );


  // ---- fallback solver if Cholesky fails ----
  if(info != 0){

    int* ipiv = new int[p];

    F77_CALL(dgesv)(
      &p, &nrhs,
      XtX.data(), &lda,
      ipiv,
      Xty.data(), &ldb,
      &info
    );

    delete[] ipiv;

    if(info != 0)
      stop("Matrix solve failed");
  }

  m->coef = Xty;


  // ---- intercept calculation ----
  double meanY = mean(y);
  vector<double> meanX(p);

  for(int j=0;j<p;j++){
    double s=0;
    for(int i=0;i<n;i++)
      s+=X(i,j);
    meanX[j]=s/n;
  }

  int inc_dot = 1;
  double dot = F77_CALL(ddot)(&p, m->coef.data(), &inc_dot, meanX.data(), &inc_dot);

  m->intercept = meanY - dot;
}


// [[Rcpp::export]]
NumericVector ridge_predict(SEXP ptr, NumericMatrix X){

  XPtr<RidgeModel> m(ptr);

  int n = X.nrow();
  int p = X.ncol();

  if(p != m->p)
    stop("Feature mismatch");

  NumericVector out(n, m->intercept);

  const char* notrans="N";
  double one=1.0;
  int inc=1;

  F77_CALL(dgemv)(
    notrans,
    &n, &p,
    &one,
    X.begin(), &n,
    m->coef.data(), &inc,
    &one,
    out.begin(), &inc
    FCONE
  );

  return out;
}
