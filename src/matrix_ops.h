#pragma once
#include "FastMatrix.h"
#include <vector>

FastMatrix transpose(const FastMatrix& A);
FastMatrix matmul(const FastMatrix& A,const FastMatrix& B);
std::vector<double> matvec(const FastMatrix& A,const std::vector<double>& x);
std::vector<double> choleskySolve(FastMatrix A,std::vector<double> b);
