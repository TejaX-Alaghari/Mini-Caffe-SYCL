#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "caffe/base.hpp"
#include "./common.hpp"
#include "./thread_local.hpp"

namespace caffe {

Caffe& Caffe::Get() {
  auto ret = ThreadLocalStore<Caffe>::Get();
  return *ret;
}

#ifndef USE_CUDA

Caffe::Caffe()
  : mode_(Caffe::CPU) { }

Caffe::~Caffe() { }

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe() try : cublas_handle_(NULL), mode_(Caffe::CPU) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  /*
  DPCT1003:85: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if ((cublas_handle_ = &dpct::get_default_queue(), 0) != 0) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

Caffe::~Caffe() try {
  /*
  DPCT1003:86: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if (cublas_handle_) CUBLAS_CHECK((cublas_handle_ = nullptr, 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Caffe::SetDevice(const int device_id) try {
  int current_device;
  CUDA_CHECK(current_device = dpct::dev_mgr::instance().current_device_id());
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  /*
  DPCT1003:87: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((dpct::dev_mgr::instance().select_device(device_id), 0));
  /*
  DPCT1003:88: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if (Get().cublas_handle_) CUBLAS_CHECK((Get().cublas_handle_ = nullptr, 0));
  /*
  DPCT1003:89: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((Get().cublas_handle_ = &dpct::get_default_queue(), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Caffe::CheckDevice(const int device_id) try {
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  /*
  DPCT1003:91: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  bool r = ((0 == (dpct::dev_mgr::instance().select_device(device_id), 0)) &&
            /*
            DPCT1003:92: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            (0 == (sycl::free(0, dpct::get_default_queue()), 0)));
  // reset any error that may have occurred.
  /*
  DPCT1026:90: The call to cudaGetLastError was removed because the function
  call is redundant in DPC++.
  */
  return r;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int Caffe::FindDevice(const int start_id) try {
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  int count = 0;
  /*
  DPCT1003:93: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_CHECK((count = dpct::dev_mgr::instance().device_count(), 0));
  for (int i = start_id; i < count; i++) {
    if (CheckDevice(i)) return i;
  }
  return -1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

const char *cublasGetErrorString(int error) {
  switch (error) {
  case 0:
    return "CUBLAS_STATUS_SUCCESS";
  case 1:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case 3:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case 7:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case 8:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case 11:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case 13:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case 14:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case 15:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case 16:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

#endif  // USE_CUDA

bool GPUAvailable() {
#ifdef USE_CUDA
  return true;
#else
  return false;
#endif  // USE_CUDA
}

void SetMode(DeviceMode mode, int device) {
  switch (mode) {
  case CPU:
    Caffe::Get().set_mode(Caffe::CPU);
    break;
  case GPU:
    Caffe::Get().set_mode(Caffe::GPU);
    Caffe::Get().SetDevice(device);
    break;
  default:
    LOG(FATAL) << "Unsupported Device Mode: " << mode;
    break;
  }
}

}  // namespace caffe
