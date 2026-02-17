#include <Rcpp.h>
#include <vector>
#include <algorithm>
using namespace Rcpp;
using std::vector;

struct Node{
  int feature;
  double thresh;
  double value;
  Node* left;
  Node* right;
};

double mse(const vector<double>& y){
  double m=0;
  for(double v:y) m+=v;
  m/=y.size();
  double s=0;
  for(double v:y) s+=(v-m)*(v-m);
  return s;
}

Node* build(NumericMatrix X, NumericVector y, int depth,int maxd){

  int n=X.nrow();
  int p=X.ncol();

  Node* node=new Node();

  double best=1e18;
  int bf=-1;
  double bt=0;

  for(int j=0;j<p;j++){
    for(int i=0;i<n;i++){
      double t=X(i,j);

      vector<double> L,R;

      for(int k=0;k<n;k++)
        (X(k,j)<=t?L:R).push_back(y[k]);

      if(L.empty()||R.empty()) continue;

      double score=mse(L)+mse(R);
      if(score<best){
        best=score;
        bf=j;
        bt=t;
      }
    }
  }

  if(depth>=maxd||bf==-1){
    double s=0;
    for(double v:y) s+=v;
    node->value=s/n;
    node->left=node->right=NULL;
    return node;
  }

  node->feature=bf;
  node->thresh=bt;

  vector<int> li,ri;
  for(int i=0;i<n;i++)
    (X(i,bf)<=bt?li:ri).push_back(i);

  NumericMatrix XL(li.size(),p);
  NumericVector yL(li.size());
  for(size_t i=0;i<li.size();i++){
    XL(i,_)=X(li[i],_);
    yL[i]=y[li[i]];
  }

  NumericMatrix XR(ri.size(),p);
  NumericVector yR(ri.size());
  for(size_t i=0;i<ri.size();i++){
    XR(i,_)=X(ri[i],_);
    yR[i]=y[ri[i]];
  }

  node->left=build(XL,yL,depth+1,maxd);
  node->right=build(XR,yR,depth+1,maxd);
  return node;
}

double predict(Node* n, NumericVector x){
  if(!n->left) return n->value;
  return x[n->feature]<=n->thresh ?
    predict(n->left,x):
    predict(n->right,x);
}

struct Tree{
  Node* root;
  int max_depth;
};

// [[Rcpp::export]]
SEXP dt_create(int depth){
  XPtr<Tree> t(new Tree(),true);
  t->max_depth=depth;
  return t;
}

// [[Rcpp::export]]
void dt_fit(SEXP ptr, NumericMatrix X, NumericVector y){
  XPtr<Tree> t(ptr);
  t->root=build(X,y,0,t->max_depth);
}

// [[Rcpp::export]]
NumericVector dt_predict(SEXP ptr, NumericMatrix X){
  XPtr<Tree> t(ptr);
  int n=X.nrow();
  NumericVector out(n);
  for(int i=0;i<n;i++)
    out[i]=predict(t->root,X(i,_));
  return out;
}
