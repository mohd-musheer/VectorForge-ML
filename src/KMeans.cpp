#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

using namespace Rcpp;
using std::vector;

struct KMeansModel{
  vector<double> centroids;
  int k;
  int p;
};

// squared euclidean
inline double dist2(const double* x, const double* c, int p){
  double s=0;
  for(int j=0;j<p;j++){
    double d=x[j]-c[j];
    s+=d*d;
  }
  return s;
}

// [[Rcpp::export]]
SEXP kmeans_create(int k){
  XPtr<KMeansModel> m(new KMeansModel(),true);
  m->k=k;
  return m;
}

// [[Rcpp::export]]
void kmeans_fit(SEXP ptr, NumericMatrix X, int max_iter=100){

  XPtr<KMeansModel> m(ptr);

  const int n=X.nrow();
  const int p=X.ncol();
  const int k=m->k;

  m->p=p;
  m->centroids.assign(k*p,0.0);

  const double* x=X.begin();

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> uni(0,n-1);

  // init random centroids
  for(int c=0;c<k;c++){
    int r=uni(rng);
    for(int j=0;j<p;j++)
      m->centroids[c*p+j]=x[j*n+r];
  }

  vector<int> labels(n);

  for(int it=0; it<max_iter; it++){

    bool changed=false;

    // assign step
    for(int i=0;i<n;i++){


      double best=std::numeric_limits<double>::max();
      int bestk=0;

      for(int c=0;c<k;c++){
        double d=0;
        const double* cent=&m->centroids[c*p];

        for(int j=0;j<p;j++){
          double diff=x[j*n+i]-cent[j];
          d+=diff*diff;
        }

        if(d<best){
          best=d;
          bestk=c;
        }
      }

      if(labels[i]!=bestk){
        labels[i]=bestk;
        changed=true;
      }
    }

    if(!changed) break;

    // recompute
    vector<double> sums(k*p,0.0);
    vector<int> counts(k,0);

    for(int i=0;i<n;i++){
      int c=labels[i];
      counts[c]++;
      for(int j=0;j<p;j++)
        sums[c*p+j]+=x[j*n+i];
    }

    for(int c=0;c<k;c++){
      if(counts[c]==0) continue;
      for(int j=0;j<p;j++)
        m->centroids[c*p+j]=sums[c*p+j]/counts[c];
    }
  }
}

// [[Rcpp::export]]
IntegerVector kmeans_predict(SEXP ptr, NumericMatrix X){

  XPtr<KMeansModel> m(ptr);

  const int n=X.nrow();
  const int p=m->p;
  const int k=m->k;

  const double* x=X.begin();

  IntegerVector out(n);

  for(int i=0;i<n;i++){

    double best=1e100;
    int bestk=0;

    for(int c=0;c<k;c++){
      double d=0;
      const double* cent=&m->centroids[c*p];

      for(int j=0;j<p;j++){
        double diff=x[j*n+i]-cent[j];
        d+=diff*diff;
      }

      if(d<best){
        best=d;
        bestk=c;
      }
    }

    out[i]=bestk+1;
  }

  return out;
}
