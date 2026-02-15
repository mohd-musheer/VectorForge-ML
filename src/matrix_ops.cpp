#include "matrix_ops.h"

#include <cmath>
#include <stdexcept>

FastMatrix transpose(const FastMatrix& A) {
  FastMatrix T(A.cols, A.rows);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      T(j, i) = A(i, j);
    }
  }

  return T;
}

FastMatrix matmul(const FastMatrix& A, const FastMatrix& B) {
  if (A.cols != B.rows) {
    throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
  }

  FastMatrix C(A.rows, B.cols);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < A.rows; ++i) {
    for (int k = 0; k < A.cols; ++k) {
      const double aik = A(i, k);
      for (int j = 0; j < B.cols; ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }

  return C;
}

std::vector<double> matvec(const FastMatrix& A, const std::vector<double>& x) {
  if (A.cols != static_cast<int>(x.size())) {
    throw std::invalid_argument("Incompatible matrix/vector dimensions");
  }

  std::vector<double> y(A.rows, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int i = 0; i < A.rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < A.cols; ++j) {
      sum += A(i, j) * x[j];
    }
    y[i] = sum;
  }

  return y;
}

std::vector<double> choleskySolve(FastMatrix A, std::vector<double> b) {
  const int n = A.rows;
  if (A.cols != n || static_cast<int>(b.size()) != n) {
    throw std::invalid_argument("Cholesky solver requires square matrix and matching RHS");
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = A(i, j);
      for (int k = 0; k < j; ++k) {
        sum -= A(i, k) * A(j, k);
      }

      if (i == j) {
        if (sum <= 0.0) {
          sum = 1e-12;
        }
        A(i, j) = std::sqrt(sum);
      } else {
        A(i, j) = sum / A(j, j);
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < i; ++k) {
      b[i] -= A(i, k) * b[k];
    }
    b[i] /= A(i, i);
  }

  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k) {
      b[i] -= A(k, i) * b[k];
    }
    b[i] /= A(i, i);
  }

  return b;
}