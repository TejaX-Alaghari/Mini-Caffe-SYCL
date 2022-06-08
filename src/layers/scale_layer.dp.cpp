#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cfloat>
#include <vector>

#include "./scale_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ScaleForward(const int n, const real_t* in,
    const real_t* scale, const int scale_dim, const int inner_dim,
    real_t* out, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

void ScaleBiasForward(const int n, const real_t* in,
    const real_t* scale, const real_t* bias,
    const int scale_dim, const int inner_dim, real_t* out,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    const int scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

void ScaleLayer::Forward_gpu(const vector<Blob *> &bottom,
                             const vector<Blob *> &top) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int count = top[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  if (bias_layer_) {
    const real_t* bias_data = this->blobs_[bias_param_id_]->gpu_data();
    /*
    DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto scale_dim__ct4 = scale_dim_;
      auto inner_dim__ct5 = inner_dim_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            ScaleBiasForward(count, bottom_data, scale_data, bias_data,
                             scale_dim__ct4, inner_dim__ct5, top_data,
                             item_ct1);
          });
    });
  } else {
    /*
    DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto scale_dim__ct3 = scale_dim_;
      auto inner_dim__ct4 = inner_dim_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            ScaleForward(count, bottom_data, scale_data, scale_dim__ct3,
                         inner_dim__ct4, top_data, item_ct1);
          });
    });
  }
}

}  // namespace caffe
