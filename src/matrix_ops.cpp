#include "matrix_ops.h"

#include <cmath>
#include <stdexcept>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

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

  const char* transa = "N";
  const char* transb = "N";
  const int m = A.rows;
  const int n = B.cols;
  const int k = A.cols;
  const double alpha = 1.0;
  const double beta = 0.0;
  
  // Note: FastMatrix is row-major. BLAS uses column-major.
  // Using property: (A_row * B_row)^T = B_row^T * A_row^T = B_col * A_col
  // So C_col = dgemm(B_col, A_col).
  // C_col (n x m) has leading dimension n.
  
  F77_CALL(dgemm)(
      transb, transa,
      &n, &m, &k,
      &alpha,
      B.ptr(), &n,
      A.ptr(), &k,
      &beta,
      C.ptr(), &n
      FCONE FCONE);

  return C;
}

std::vector<double> matvec(const FastMatrix& A, const std::vector<double>& x) {
  if (A.cols != static_cast<int>(x.size())) {
    throw std::invalid_argument("Incompatible matrix/vector dimensions");
  }

  std::vector<double> y(A.rows, 0.0);

  const char* trans = "T";
  const int m = A.cols;
  const int n = A.rows;
  const double alpha = 1.0;
  const double beta = 0.0;
  const int inc = 1;

  // A is row-major (m cols, n rows). BLAS expects column-major.
  // Treating A as transpose of an (m x n) column-major matrix.
  // We want A_row * x = A_col^T * x.
  F77_CALL(dgemv)(
      trans, &m, &n,
      &alpha,
      A.ptr(), &m,
      x.data(), &inc,
      &beta,
      y.data(), &inc
      FCONE);

  return y;
}

std::vector<double> choleskySolve(FastMatrix A, std::vector<double> b) {
  const int n = A.rows;
  if (A.cols != n || static_cast<int>(b.size()) != n) {
    throw std::invalid_argument("Cholesky solver requires square matrix and matching RHS");
  }

  // A is row-major (n x n). Since A is symmetric, row-major = column-major.
  const char* uplo = "U"; // Lower in row-major is Upper in column-major
  int info = 0;
  
  F77_CALL(dpotrf)(
      uplo, &n,
      A.ptr(), &n,
      &info
      FCONE);
      
  if (info != 0) {
    // Factorization failed, perhaps not positive definite.
    // Fallback to avoid outright crashing, though dpotrf is stable.
    throw std::runtime_error("Cholesky factorization failed");
  }

  int nrhs = 1;
  F77_CALL(dpotrs)(
      uplo, &n, &nrhs,
      A.ptr(), &n,
      b.data(), &n,
      &info
      FCONE);

  if (info != 0) {
    throw std::runtime_error("Cholesky solve failed");
  }

  return b;
}