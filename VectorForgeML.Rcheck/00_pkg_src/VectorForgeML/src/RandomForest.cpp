#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <random>

using namespace Rcpp;
using std::vector;

struct Node{
  int feature;
  double thresh;
  double value;
  Node* left;
  Node* right;
};

double variance(const vector<double>& y){
  double m=0;
  for(double v:y) m+=v;
  m/=y.size();
  double s=0;
  for(double v:y) s+=(v-m)*(v-m);
  return s;
}

double majority(const vector<double>& y){
  std::map<double,int> c;
  for(double v:y) c[v]++;
  return std::max_element(c.begin(),c.end(),
        [](auto&a,auto&b){return a.second<b.second;})->first;
}

Node* build(NumericMatrix X, NumericVector y,
            int depth,int max_depth,
            int mtry,bool cls,
            std::mt19937& rng){

  int n=X.nrow();
  int p=X.ncol();

  Node* node=new Node();

  if(depth>=max_depth || n<=2){
    vector<double> v(y.begin(),y.end());
    node->value = cls ? majority(v) : mean(y);
    node->left=node->right=NULL;
    return node;
  }

  std::uniform_int_distribution<int> fd(0,p-1);
  vector<int> feats;
  for(int i=0;i<mtry;i++)
    feats.push_back(fd(rng));

  double best=1e18;
  int bf=-1;
  double bt=0;

  for(int f:feats){

    for(int i=0;i<n;i++){
      double t=X(i,f);

      vector<double>L,R;

      for(int k=0;k<n;k++)
        (X(k,f)<=t?L:R).push_back(y[k]);

      if(L.empty()||R.empty()) continue;

      double score = cls ?
        (variance(L)+variance(R)) :
        (variance(L)+variance(R));

      if(score<best){
        best=score;
        bf=f;
        bt=t;
      }
    }
  }

  if(bf==-1){
    vector<double> v(y.begin(),y.end());
    node->value = cls ? majority(v) : mean(y);
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

  node->left = build(XL,yL,depth+1,max_depth,mtry,cls,rng);
  node->right = build(XR,yR,depth+1,max_depth,mtry,cls,rng);

  return node;
}

double predict_tree(Node* n, NumericVector x){
  if(!n->left) return n->value;
  return x[n->feature]<=n->thresh ?
    predict_tree(n->left,x):
    predict_tree(n->right,x);
}

struct Forest{
  vector<Node*> trees;
  int max_depth;
  int mtry;
  bool classification;
};



// [[Rcpp::export]]
SEXP rf_create(int ntrees,int depth,int mtry,bool cls){
  XPtr<Forest> f(new Forest(),true);
  f->trees.resize(ntrees);
  f->max_depth=depth;
  f->mtry=mtry;
  f->classification=cls;
  return f;
}



// [[Rcpp::export]]
void rf_fit(SEXP ptr, NumericMatrix X, NumericVector y){

  XPtr<Forest> f(ptr);

  int n=X.nrow();
  int p=X.ncol();

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0,n-1);

  for(size_t t=0;t<f->trees.size();t++){

    NumericMatrix XB(n,p);
    NumericVector yb(n);

    for(int i=0;i<n;i++){
      int id=dist(rng);
      XB(i,_)=X(id,_);
      yb[i]=y[id];
    }

    f->trees[t] = build(XB,yb,0,f->max_depth,f->mtry,f->classification,rng);
  }
}



// [[Rcpp::export]]
NumericVector rf_predict(SEXP ptr, NumericMatrix X){

  XPtr<Forest> f(ptr);

  int n=X.nrow();
  NumericVector out(n);

  for(int i=0;i<n;i++){

    vector<double> preds;

    for(Node* t:f->trees)
      preds.push_back(predict_tree(t,X(i,_)));

    if(f->classification){

      std::map<double,int> c;
      for(double v:preds) c[v]++;

      out[i]=std::max_element(c.begin(),c.end(),
          [](auto&a,auto&b){return a.second<b.second;})->first;
    }
    else{
      out[i]=std::accumulate(preds.begin(),preds.end(),0.0)/preds.size();
    }
  }

  return out;
}
