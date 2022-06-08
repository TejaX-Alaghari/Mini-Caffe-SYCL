#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
  // CUDA's, not caffe's, for fabs, signbit
#include <dpct/dpl_utils.hpp>
  // thrust::plus

#include <cmath>

#include "./math_functions.hpp"

namespace caffe {

void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K, const float alpha,
                    const float *A, const float *B, const float beta,
                    float *C) try {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  oneapi::mkl::transpose cuTransA = (TransA == CblasNoTrans)
                                        ? oneapi::mkl::transpose::nontrans
                                        : oneapi::mkl::transpose::trans;
  oneapi::mkl::transpose cuTransB = (TransB == CblasNoTrans)
                                        ? oneapi::mkl::transpose::nontrans
                                        : oneapi::mkl::transpose::trans;
  /*
  DPCT1003:97: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((oneapi::mkl::blas::column_major::gemm(
                    *Caffe::cublas_handle(), cuTransB, cuTransA, N, M, K,
                    dpct::get_value(&alpha, *Caffe::cublas_handle()), B, ldb, A,
                    lda, dpct::get_value(&beta, *Caffe::cublas_handle()), C, N),
                0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
                    const float alpha, const float *A, const float *x,
                    const float beta, float *y) try {
  oneapi::mkl::transpose cuTransA = (TransA == CblasNoTrans)
                                        ? oneapi::mkl::transpose::trans
                                        : oneapi::mkl::transpose::nontrans;
  /*
  DPCT1003:98: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((oneapi::mkl::blas::column_major::gemv(
                    *Caffe::cublas_handle(), cuTransA, N, M,
                    dpct::get_value(&alpha, *Caffe::cublas_handle()), A, N, x,
                    1, dpct::get_value(&beta, *Caffe::cublas_handle()), y, 1),
                0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_axpy(const int N, const float alpha, const float *X,
                    float *Y) try {
  /*
  DPCT1003:99: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK(
      (oneapi::mkl::blas::column_major::axpy(
           *Caffe::cublas_handle(), N,
           dpct::get_value(&alpha, *Caffe::cublas_handle()), X, 1, Y, 1),
       0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y) try {
  std::cout << "Size of mem in caffe_gpu_memcpy: " << N << "B" << std::endl;

  if (X != Y) {
    std::cout << "Using CUDA memcpy" << std::endl;
    /*
    DPCT1003:100: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_CHECK((dpct::get_default_queue().memcpy(Y, X, N).wait(),
                0)); // NOLINT(caffe/alt_fn)
    /*
    CUDA_CHECK((dpct::get_default_queue().memcpy(Y, X, N),
                0)); // NOLINT(caffe/alt_fn)
    */
  }
  else {
    std::cout << "Pointers are same" << std::endl;
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_scal(const int N, const float alpha, float *X) try {
  /*
  DPCT1003:101: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((oneapi::mkl::blas::column_major::scal(
                    *Caffe::cublas_handle(), N,
                    dpct::get_value(&alpha, *Caffe::cublas_handle()), X, 1),
                0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_axpby(const int N, const float alpha, const float* X,
                     const float beta, float* Y) {
  caffe_gpu_scal(N, beta, Y);
  caffe_gpu_axpy(N, alpha, X, Y);
}

void caffe_gpu_dot(const int n, const float *x, const float *y,
                   float *out) try {
  /*
  DPCT1034:102: Migrated API does not return error code. 0 is returned in the
  lambda. You may need to rewrite this code.
  */
  CUBLAS_CHECK([&]() {
    float *res_temp_ptr_ct1 = out;
    if (sycl::get_pointer_type(out, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(out, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::shared) {
      res_temp_ptr_ct1 =
          sycl::malloc_shared<float>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::dot(*Caffe::cublas_handle(), n, x, 1, y, 1,
                                         res_temp_ptr_ct1);
    if (sycl::get_pointer_type(out, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(out, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::shared) {
      Caffe::cublas_handle()->wait();
      *out = *res_temp_ptr_ct1;
      sycl::free(res_temp_ptr_ct1, dpct::get_default_queue());
    }
    return 0;
  }());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_asum(const int n, const float *x, float *y) try {
  /*
  DPCT1034:103: Migrated API does not return error code. 0 is returned in the
  lambda. You may need to rewrite this code.
  */
  CUBLAS_CHECK([&]() {
    float *res_temp_ptr_ct2 = y;
    if (sycl::get_pointer_type(y, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(y, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::shared) {
      res_temp_ptr_ct2 =
          sycl::malloc_shared<float>(1, dpct::get_default_queue());
    }
    oneapi::mkl::blas::column_major::asum(*Caffe::cublas_handle(), n, x, 1,
                                          res_temp_ptr_ct2);
    if (sycl::get_pointer_type(y, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::device &&
        sycl::get_pointer_type(y, Caffe::cublas_handle()->get_context()) !=
            sycl::usm::alloc::shared) {
      Caffe::cublas_handle()->wait();
      *y = *res_temp_ptr_ct2;
      sycl::free(res_temp_ptr_ct2, dpct::get_default_queue());
    }
    return 0;
  }());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void caffe_gpu_scale(const int n, const float alpha, const float *x,
                     float *y) try {
  /*
  DPCT1003:104: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((oneapi::mkl::blas::column_major::copy(*Caffe::cublas_handle(),
                                                      n, x, 1, y, 1),
                0));
  /*
  DPCT1003:105: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUBLAS_CHECK((oneapi::mkl::blas::column_major::scal(
                    *Caffe::cublas_handle(), n,
                    dpct::get_value(&alpha, *Caffe::cublas_handle()), y, 1),
                0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void set_kernel(const int n, const real_t alpha, real_t* y,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

void caffe_gpu_set(const int N, const real_t alpha, real_t *Y) try {
  if (alpha == 0) {
    /*
    DPCT1003:106: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_CHECK(
        (dpct::get_default_queue().memset(Y, 0, sizeof(real_t) * N).wait(),
         0)); // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:107: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        set_kernel(N, alpha, Y, item_ct1);
      });
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void add_scalar_kernel(const int n, const real_t alpha, real_t* y,
                       sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:108: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        add_scalar_kernel(N, alpha, Y, item_ct1);
      });
}

void add_kernel(const int n, const real_t* a,
    const real_t* b, real_t* y, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

void caffe_gpu_add(const int N, const float* a, const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:109: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        add_kernel(N, a, b, y, item_ct1);
      });
}

void sub_kernel(const int n, const real_t* a,
    const real_t* b, real_t* y, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

void caffe_gpu_sub(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:110: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        sub_kernel(N, a, b, y, item_ct1);
      });
}

void mul_kernel(const int n, const real_t* a,
    const real_t* b, real_t* y, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

void caffe_gpu_mul(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:111: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        mul_kernel(N, a, b, y, item_ct1);
      });
}

void div_kernel(const int n, const real_t* a,
    const real_t* b, real_t* y, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

void caffe_gpu_div(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:112: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        div_kernel(N, a, b, y, item_ct1);
      });
}

void abs_kernel(const int n, const real_t* a, real_t* y,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sycl::fabs(a[index]);
  }
}

void caffe_gpu_abs(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:113: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        abs_kernel(N, a, y, item_ct1);
      });
}

void exp_kernel(const int n, const real_t* a, real_t* y,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sycl::exp((float)(a[index]));
  }
}

void caffe_gpu_exp(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:114: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        exp_kernel(N, a, y, item_ct1);
      });
}

void log_kernel(const int n, const real_t* a, real_t* y,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sycl::log((float)(a[index]));
  }
}

void caffe_gpu_log(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:115: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        log_kernel(N, a, y, item_ct1);
      });
}

void powx_kernel(const int n, const real_t* a,
    const real_t alpha, real_t* y, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sycl::pow<double>(a[index], alpha);
  }
}

void caffe_gpu_powx(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:116: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(N)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        powx_kernel(N, a, alpha, y, item_ct1);
      });
}

}  // namespace caffe
