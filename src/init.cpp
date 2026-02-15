#include <Rcpp.h>
#include <thread>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#include <cstdlib>
#endif

using namespace Rcpp;

namespace {
using openblas_set_num_threads_fn = void (*)(int);

int optimal_threads() {
  const unsigned int n = std::thread::hardware_concurrency();
  return (n > 0U) ? static_cast<int>(n) : 1;
}

openblas_set_num_threads_fn find_openblas_thread_setter() {
#if defined(_WIN32)
  const char* modules[] = {"libopenblas.dll", "openblas.dll", "Rblas.dll", "R.dll"};
  for (const char* mod_name : modules) {
    HMODULE mod = GetModuleHandleA(mod_name);
    if (mod != nullptr) {
      FARPROC p = GetProcAddress(mod, "openblas_set_num_threads");
      if (p != nullptr) {
        return reinterpret_cast<openblas_set_num_threads_fn>(p);
      }
    }
  }
  return nullptr;
#else
  void* p = dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
  return reinterpret_cast<openblas_set_num_threads_fn>(p);
#endif
}

void set_thread_env_vars(int nthreads) {
#if defined(_WIN32)
  _putenv_s("OPENBLAS_NUM_THREADS", std::to_string(nthreads).c_str());
  _putenv_s("OMP_NUM_THREADS", std::to_string(nthreads).c_str());
#else
  setenv("OPENBLAS_NUM_THREADS", std::to_string(nthreads).c_str(), 0);
  setenv("OMP_NUM_THREADS", std::to_string(nthreads).c_str(), 0);
#endif
}
}  // namespace

// [[Rcpp::export]]
void cpp_set_blas_threads() {
  const int nthreads = optimal_threads();
  openblas_set_num_threads_fn setter = find_openblas_thread_setter();

  if (setter != nullptr) {
    setter(nthreads);
  }

  set_thread_env_vars(nthreads);
}