#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace std;


// =========================
// MATRIX HELPERS
// =========================

vector<vector<double>> toStd(NumericMatrix X){
    int r=X.nrow(), c=X.ncol();
    vector<vector<double>> out(r, vector<double>(c));

    #pragma omp parallel for
    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            out[i][j]=X(i,j);

    return out;
}


// transpose
vector<vector<double>> transpose(const vector<vector<double>>& A){
    int r=A.size(), c=A[0].size();
    vector<vector<double>> T(c, vector<double>(r));

    #pragma omp parallel for
    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            T[j][i]=A[i][j];

    return T;
}


// multiply
vector<vector<double>> matmul(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B){

    int r=A.size(), c=B[0].size(), k=B.size();
    vector<vector<double>> C(r, vector<double>(c,0));

    #pragma omp parallel for
    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            for(int t=0;t<k;t++)
                C[i][j]+=A[i][t]*B[t][j];

    return C;
}


// =========================
// CHOLESKY SOLVER
// solves Ax=b
// =========================

vector<double> choleskySolve(
    vector<vector<double>> A,
    vector<double> b){

    int n=A.size();

    vector<vector<double>> L(n, vector<double>(n,0));

    // decomposition
    for(int i=0;i<n;i++){
        for(int j=0;j<=i;j++){

            double s=0;
            for(int k=0;k<j;k++)
                s+=L[i][k]*L[j][k];

            if(i==j)
                L[i][j]=sqrt(A[i][i]-s);
            else
                L[i][j]=(1.0/L[j][j])*(A[i][j]-s);
        }
    }

    // forward solve Ly=b
    vector<double> y(n);
    for(int i=0;i<n;i++){
        double s=0;
        for(int k=0;k<i;k++)
            s+=L[i][k]*y[k];
        y[i]=(b[i]-s)/L[i][i];
    }

    // backward solve Lᵀx=y
    vector<double> x(n);
    for(int i=n-1;i>=0;i--){
        double s=0;
        for(int k=i+1;k<n;k++)
            s+=L[k][i]*x[k];
        x[i]=(y[i]-s)/L[i][i];
    }

    return x;
}


// =========================
// MODEL CLASS
// =========================

class LinearRegression{
public:

    vector<double> W;

    void fit(NumericMatrix X, NumericVector y){

        auto Xs = toStd(X);

        // add intercept
        for(auto &row : Xs)
            row.insert(row.begin(),1.0);

        auto Xt = transpose(Xs);
        auto XtX = matmul(Xt,Xs);

        // ridge stabilization
        for(size_t i=0;i<XtX.size();i++)
            XtX[i][i]+=1e-8;

        // XtY
        vector<double> XtY(Xt.size(),0);

        #pragma omp parallel for
        for(size_t i=0;i<Xt.size();i++)
            for(size_t j=0;j<Xt[0].size();j++)
                XtY[i]+=Xt[i][j]*y[j];

        // solve system
        W = choleskySolve(XtX,XtY);
    }



    NumericVector predict(NumericMatrix X){

        auto Xs = toStd(X);

        for(auto &row : Xs)
            row.insert(row.begin(),1.0);

        NumericVector out(Xs.size());

        #pragma omp parallel for
        for(size_t i=0;i<Xs.size();i++){

            double s=0;
            for(size_t j=0;j<W.size();j++)
                s+=Xs[i][j]*W[j];

            out[i]=s;
        }

        return out;
    }
};



// =========================
// R WRAPPERS
// =========================

// [[Rcpp::export]]
SEXP lr_create(){
    XPtr<LinearRegression> ptr(new LinearRegression(), true);
    return ptr;
}

// [[Rcpp::export]]
void lr_fit(SEXP model, NumericMatrix X, NumericVector y){
    XPtr<LinearRegression> ptr(model);
    ptr->fit(X,y);
}

// [[Rcpp::export]]
NumericVector lr_predict(SEXP model, NumericMatrix X){
    XPtr<LinearRegression> ptr(model);
    return ptr->predict(X);
}
