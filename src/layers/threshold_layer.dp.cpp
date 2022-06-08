#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./threshold_layer.hpp"

namespace caffe {

void ThresholdForward(const int n, const real_t threshold,
    const real_t* in, real_t* out, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

void ThresholdLayer::Forward_gpu(const vector<Blob *> &bottom,
                                 const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:80: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto threshold__ct1 = threshold_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          ThresholdForward(count, threshold__ct1, bottom_data, top_data,
                           item_ct1);
        });
  });
  /*
  DPCT1010:81: SYCL uses exceptions to report errors and does not use the error
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
