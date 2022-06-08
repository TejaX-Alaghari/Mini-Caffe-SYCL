#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./bias_layer.hpp"
#include "../filler.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void BiasForward(const int n, const real_t* in,
                            const real_t* bias, const int bias_dim,
                            const int inner_dim, real_t* out,
                            sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

void BiasLayer::Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const int count = top[0]->count();
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  /*
  DPCT1049:94: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto bias_dim__ct3 = bias_dim_;
    auto inner_dim__ct4 = inner_dim_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          BiasForward(count, bottom_data, bias_data, bias_dim__ct3,
                      inner_dim__ct4, top_data, item_ct1);
        });
  });
}

}  // namespace caffe
