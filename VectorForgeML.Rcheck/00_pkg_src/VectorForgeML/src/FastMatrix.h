#pragma once
#include <vector>
using namespace std;

class FastMatrix{
public:
    int rows, cols;
    vector<double> data;

    FastMatrix(int r,int c): rows(r), cols(c), data(r*c,0.0){}

    inline double& operator()(int i,int j){
        return data[i*cols+j];
    }

    inline const double& operator()(int i,int j) const{
        return data[i*cols+j];
    }

    double* ptr(){ return data.data(); }
    const double* ptr() const { return data.data(); }
};
