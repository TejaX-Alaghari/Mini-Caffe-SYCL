// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./tanh_layer.hpp"

namespace caffe {

void TanHForward(const int n, const real_t* in, real_t* out,
                 sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = sycl::tanh((float)(in[index]));
  }
}

void TanHLayer::Forward_gpu(const vector<Blob *> &bottom,
                            const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:79: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        TanHForward(count, bottom_data, top_data, item_ct1);
      });
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace caffe
