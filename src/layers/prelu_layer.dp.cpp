#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <vector>

#include "./prelu_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

// CUDA kernele for forward
void PReLUForward(const int n, const int channels, const int dim,
    const real_t* in, real_t* out, const real_t* slope_data,
    const int div_factor, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

void PReLULayer::Forward_gpu(const vector<Blob *> &bottom,
                             const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const real_t* slope_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:47: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        PReLUForward(count, channels, dim, bottom_data, top_data, slope_data,
                     div_factor, item_ct1);
      });
  /*
  DPCT1010:48: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace caffe
