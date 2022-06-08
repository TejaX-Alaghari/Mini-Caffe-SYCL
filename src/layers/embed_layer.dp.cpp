#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./embed_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void EmbedForward(const int nthreads, const real_t* bottom_data,
    const real_t* weight, const int M, const int N, const int K,
    real_t* top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(top_index, nthreads) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = static_cast<int>(bottom_data[n]);
    const int weight_index = index * N + d;
    top_data[top_index] = weight[weight_index];
  }
}

void EmbedLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* weight = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  /*
  DPCT1049:127: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto M__ct3 = M_;
    auto N__ct4 = N_;
    auto K__ct5 = K_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          EmbedForward(count, bottom_data, weight, M__ct3, N__ct4, K__ct5,
                       top_data, item_ct1);
        });
  });
  if (bias_term_) {
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, static_cast<real_t>(1),
      bias_multiplier_.gpu_data(),
      this->blobs_[1]->gpu_data(), static_cast<real_t>(1), top_data);
  }
}

}  // namespace caffe
