#include "matrix_ops.h"
#include <Rcpp.h>
using namespace std;
using namespace Rcpp;

vector<vector<double>> transpose(const vector<vector<double>>& X){
    int r=X.size(), c=X[0].size();
    vector<vector<double>> T(c, vector<double>(r));
    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            T[j][i]=X[i][j];
    return T;
}

vector<vector<double>> matmul(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B){

    int r=A.size(), c=B[0].size(), k=B.size();
    vector<vector<double>> C(r, vector<double>(c,0));

    for(int i=0;i<r;i++)
        for(int j=0;j<c;j++)
            for(int t=0;t<k;t++)
                C[i][j]+=A[i][t]*B[t][j];

    return C;
}

vector<vector<double>> inverse(vector<vector<double>> A){

    int n=A.size();
    vector<vector<double>> I(n, vector<double>(n,0));

    for(int i=0;i<n;i++)
        I[i][i]=1;

    for(int i=0;i<n;i++){
        double d=A[i][i];
        if(d==0) stop("Singular matrix");

        for(int j=0;j<n;j++){
            A[i][j]/=d;
            I[i][j]/=d;
        }

        for(int k=0;k<n;k++){
            if(k==i) continue;
            double f=A[k][i];
            for(int j=0;j<n;j++){
                A[k][j]-=f*A[i][j];
                I[k][j]-=f*I[i][j];
            }
        }
    }
    return I;
}
