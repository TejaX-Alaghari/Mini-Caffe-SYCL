#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

#include "caffe/base.hpp"
#include "./thread_local.hpp"
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#ifdef USE_CUDA

  // cuda driver types

//
// CUDA macros
//

// CUDA: various checks for different function calls.
/*
DPCT1009:1: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define CUDA_CHECK(condition)                                                  \
  /* Code block avoids redefinition of cudaError_t error */                    \
  do {                                                                         \
    int error = condition;                                                     \
    CHECK_EQ(error, 0)                                                         \
        << " "                                                                 \
        << "cudaGetErrorString not supported" /*cudaGetErrorString(error)*/;   \
  } while (0)

#define CUBLAS_CHECK(condition)                                                \
  do {                                                                         \
    int status = condition;                                                    \
    CHECK_EQ(status, 0) << " " << caffe::cublasGetErrorString(status);         \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +     \
               item_ct1.get_local_id(2);                                       \
       i < (n);                                                                \
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2))

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(0)

namespace caffe {

// CUDA: library error reporting.
const char *cublasGetErrorString(int error);

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace caffe

#endif  // USE_CUDA

#define STUB_GPU(classname)                                        \
void classname::Forward_gpu(const vector<Blob*>& bottom,           \
                            const vector<Blob*>& top) { NO_GPU; }

#define STUB_GPU_FORWARD(classname, funcname)                          \
void classname::funcname##_##gpu(const vector<Blob*>& bottom,          \
                                 const vector<Blob*>& top) { NO_GPU; }

namespace caffe {

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();

  static Caffe& Get();

  enum Brew { CPU, GPU };

#ifdef USE_CUDA
  inline static sycl::queue *cublas_handle() { return Get().cublas_handle_; }
#endif  // USE_CUDA

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);

 protected:
#ifdef USE_CUDA
  sycl::queue *cublas_handle_;
#endif
  Brew mode_;

 private:
  friend ThreadLocalStore<Caffe>;
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
