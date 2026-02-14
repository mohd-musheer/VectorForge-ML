#pragma once
#include <vector>

class FastMatrix{

public:
    int rows, cols;
    std::vector<double> data;

    FastMatrix(int r,int c)
        : rows(r), cols(c), data(r*c){}

    inline double& operator()(int i,int j){
        return data[i*cols+j];
    }

    inline double operator()(int i,int j) const{
        return data[i*cols+j];
    }
};
