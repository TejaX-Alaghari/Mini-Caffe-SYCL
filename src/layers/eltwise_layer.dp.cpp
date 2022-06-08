#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cfloat>
#include <vector>
#include <limits>

#include "./eltwise_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void MaxForward(const int nthreads, const real_t* bottom_data_a,
                           const real_t* bottom_data_b, const int blob_idx,
                           real_t* top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] =
        sycl::max((float)(bottom_data_a[index]), (float)(bottom_data_b[index]));
  }
}

void EltwiseLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const int count = top[0]->count();
  real_t* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, static_cast<real_t>(0), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    /*
    DPCT1049:44: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto bottom_gpu_data_ct1 = bottom[0]->gpu_data();
      auto bottom_gpu_data_ct2 = bottom[1]->gpu_data();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            MaxForward(count, bottom_gpu_data_ct1, bottom_gpu_data_ct2, 0,
                       top_data, item_ct1);
          });
    });
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      /*
      DPCT1049:45: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto bottom_i_gpu_data_ct2 = bottom[i]->gpu_data();

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                  sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
              MaxForward(count, top_data, bottom_i_gpu_data_ct2, i - 1,
                         top_data, item_ct1);
            });
      });
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

}  // namespace caffe
