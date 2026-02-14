#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <malloc.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define RESTRICT
#define ALWAYS_INLINE inline
#endif

using namespace Rcpp;
using std::size_t;
using std::vector;

template <typename T, size_t Align = 32>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > (std::numeric_limits<std::size_t>::max() / sizeof(T))) {
            throw std::bad_alloc();
        }
        void* p = nullptr;
#if defined(_WIN32)
        p = _aligned_malloc(n * sizeof(T), Align);
        if (!p) throw std::bad_alloc();
#else
        if (posix_memalign(&p, Align, n * sizeof(T)) != 0) throw std::bad_alloc();
#endif
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template <class T, class U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) { return true; }
template <class T, class U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) { return false; }

using aligned_vec = std::vector<double, AlignedAllocator<double, 32>>;

struct LinearModel {
    aligned_vec coef;
    aligned_vec meanX;
    aligned_vec XtX;
    aligned_vec Xty;
    double intercept = 0.0;
    int p = 0;
};

ALWAYS_INLINE static void checkDims(const NumericMatrix& X, const NumericVector& y) {
    if (y.size() != X.nrow()) {
        stop("X and y must have the same number of rows");
    }
}

static vector<double> solveCholeskyFallback(vector<double> A, vector<double> b, int n) {
    for (int j = 0; j < n; ++j) {
        double d = A[j + j * n];
        for (int k = 0; k < j; ++k) d -= A[j + k * n] * A[j + k * n];
        if (d <= 1e-12) d = 1e-12;
        A[j + j * n] = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double s = A[i + j * n];
            for (int k = 0; k < j; ++k) s -= A[i + k * n] * A[j + k * n];
            A[i + j * n] = s / A[j + j * n];
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i + k * n] * b[k];
        b[i] = s / A[i + i * n];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < n; ++k) s -= A[k + i * n] * b[k];
        b[i] = s / A[i + i * n];
    }
    return b;
}

ALWAYS_INLINE static double mean_col(const double* RESTRICT col, int n) {
    double s = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : s)
#endif
    for (int i = 0; i < n; ++i) s += col[i];
    return (n > 0) ? (s / n) : 0.0;
}

static void fit_core(const NumericMatrix& X, const NumericVector& y, LinearModel& m, double ridge = 1e-8) {
    checkDims(X, y);
    const int n = X.nrow();
    const int p = X.ncol();
    if (n == 0) stop("X must have at least one row");

    m.p = p;
    m.coef.assign(p, 0.0);
    m.meanX.assign(p, 0.0);
    m.XtX.assign(static_cast<size_t>(p) * p, 0.0);
    m.Xty.assign(p, 0.0);

    const double* RESTRICT x = X.begin();
    const double* RESTRICT yy = y.begin();
    const char* trans = "T";
    const char* notrans = "N";
    const double one = 1.0;
    const double zero = 0.0;
    const int inc = 1;

    double meanY = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : meanY)
#endif
    for (int i = 0; i < n; ++i) meanY += yy[i];
    meanY /= static_cast<double>(n);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < p; ++j) {
        const double* RESTRICT col = x + static_cast<size_t>(j) * n;
        m.meanX[j] = mean_col(col, n);
    }

    F77_CALL(dgemm)(
        trans, notrans, &p, &p, &n,
        &one, x, &n, x, &n, &zero, m.XtX.data(), &p
        FCONE FCONE
    );
    F77_CALL(dgemv)(
        trans, &n, &p, &one,
        x, &n, yy, &inc, &zero, m.Xty.data(), &inc
        FCONE
    );

    const double alpha_rank1 = -static_cast<double>(n);
    F77_CALL(dger)(
        &p, &p, &alpha_rank1,
        m.meanX.data(), &inc, m.meanX.data(), &inc, m.XtX.data(), &p
    );

    const double alpha_axpy = -static_cast<double>(n) * meanY;
    F77_CALL(daxpy)(&p, &alpha_axpy, m.meanX.data(), &inc, m.Xty.data(), &inc);

#if defined(_OPENMP)
#pragma omp simd
#endif
    for (int j = 0; j < p; ++j) m.XtX[j + j * p] += ridge;

    const char* uplo = "L";
    int nsys = p;
    int nrhs = 1;
    int lda = p;
    int ldb = p;
    int info = 0;
    F77_CALL(dposv)(uplo, &nsys, &nrhs, m.XtX.data(), &lda, m.Xty.data(), &ldb, &info FCONE);

    if (info == 0) {
        std::copy(m.Xty.begin(), m.Xty.end(), m.coef.begin());
    } else {
        vector<double> A(m.XtX.begin(), m.XtX.end());
        vector<double> b(m.Xty.begin(), m.Xty.end());
        vector<double> coef = solveCholeskyFallback(std::move(A), std::move(b), p);
        std::copy(coef.begin(), coef.end(), m.coef.begin());
    }

    m.intercept = meanY;
    double dot = F77_CALL(ddot)(&p, m.coef.data(), &inc, m.meanX.data(), &inc);
    m.intercept -= dot;
}

// [[Rcpp::export]]
SEXP lr_create() {
    XPtr<LinearModel> model(new LinearModel(), true);
    return model;
}

// [[Rcpp::export]]
void lr_fit(SEXP ptr, NumericMatrix X, NumericVector y) {
    XPtr<LinearModel> model(ptr);
    fit_core(X, y, *model);
}

// [[Rcpp::export]]
NumericVector lr_predict(SEXP ptr, NumericMatrix X) {
    XPtr<LinearModel> model(ptr);
    const int n = X.nrow();
    const int p = X.ncol();
    if (p != model->p) stop("Input feature count does not match trained model");

    NumericVector pred(n, model->intercept);
    const char* notrans = "N";
    const int inc = 1;
    const double one = 1.0;
    F77_CALL(dgemv)(
        notrans, &n, &p, &one,
        X.begin(), &n, model->coef.data(), &inc, &one, pred.begin(), &inc
        FCONE
    );
    return pred;
}

// [[Rcpp::export]]
NumericVector fastLm(NumericMatrix X, NumericVector y) {
    LinearModel tmp;
    fit_core(X, y, tmp);
    return NumericVector(tmp.coef.begin(), tmp.coef.end());
}

// [[Rcpp::export]]
List cpp_scale_fit_transform(NumericMatrix X, double eps = 1e-12) {
    const int n = X.nrow();
    const int p = X.ncol();
    NumericVector means(p);
    NumericVector sds(p);
    NumericMatrix Z(n, p);

    const double* RESTRICT src = X.begin();
    double* RESTRICT dst = Z.begin();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < p; ++j) {
        const double* RESTRICT col = src + static_cast<size_t>(j) * n;
        double* RESTRICT out = dst + static_cast<size_t>(j) * n;
        double sum = 0.0;
        double sumsq = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : sum, sumsq)
#endif
        for (int i = 0; i < n; ++i) {
            const double v = col[i];
            sum += v;
            sumsq += v * v;
        }
        const double m = (n > 0) ? (sum / n) : 0.0;
        double var = 0.0;
        if (n > 1) {
            var = (sumsq - n * m * m) / (n - 1.0);
            if (var < 0.0) var = 0.0;
        }
        double sd = std::sqrt(var);
        if (sd <= eps) sd = 1.0;
        means[j] = m;
        sds[j] = sd;
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (int i = 0; i < n; ++i) out[i] = (col[i] - m) / sd;
    }
    return List::create(_["X"] = Z, _["mean"] = means, _["sd"] = sds);
}

// [[Rcpp::export]]
NumericMatrix cpp_scale_transform(NumericMatrix X, NumericVector means, NumericVector sds, double eps = 1e-12) {
    const int n = X.nrow();
    const int p = X.ncol();
    if (means.size() != p || sds.size() != p) stop("mean/sd length must match number of columns");

    NumericMatrix Z(n, p);
    const double* RESTRICT src = X.begin();
    double* RESTRICT dst = Z.begin();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < p; ++j) {
        const double m = means[j];
        double sd = sds[j];
        if (sd <= eps) sd = 1.0;
        const double* RESTRICT col = src + static_cast<size_t>(j) * n;
        double* RESTRICT out = dst + static_cast<size_t>(j) * n;
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (int i = 0; i < n; ++i) out[i] = (col[i] - m) / sd;
    }
    return Z;
}

// [[Rcpp::export]]
List cpp_drop_constant_cols(NumericMatrix X, double eps = 1e-12) {
    const int n = X.nrow();
    const int p = X.ncol();
    IntegerVector keep_mask(p);
    int keep_count = 0;
    const double* RESTRICT src = X.begin();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+ : keep_count)
#endif
    for (int j = 0; j < p; ++j) {
        const double* RESTRICT col = src + static_cast<size_t>(j) * n;
        double sum = 0.0;
        double sumsq = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : sum, sumsq)
#endif
        for (int i = 0; i < n; ++i) {
            const double v = col[i];
            sum += v;
            sumsq += v * v;
        }
        double var = 0.0;
        if (n > 1) {
            const double m = sum / n;
            var = (sumsq - n * m * m) / (n - 1.0);
            if (var < 0.0) var = 0.0;
        }
        if (var > eps) {
            keep_mask[j] = 1;
            keep_count += 1;
        }
    }

    IntegerVector keep(keep_count);
    int pos = 0;
    for (int j = 0; j < p; ++j) {
        if (keep_mask[j] == 1) keep[pos++] = j + 1;
    }

    NumericMatrix out(n, keep_count);
    double* RESTRICT dst = out.begin();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int jj = 0; jj < keep_count; ++jj) {
        const int src_col = keep[jj] - 1;
        const double* RESTRICT in_col = src + static_cast<size_t>(src_col) * n;
        double* RESTRICT out_col = dst + static_cast<size_t>(jj) * n;
        std::copy(in_col, in_col + n, out_col);
    }

    return List::create(_["X"] = out, _["keep"] = keep);
}
