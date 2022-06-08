#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "./permute_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void PermuteKernel(const int nthreads,
    real_t* const bottom_data, const bool forward, const int* permute_order,
    const int* old_steps, const int* new_steps, const int num_axes,
    real_t* const top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    if (forward) {
      top_data[index] = bottom_data[old_idx];
    } else {
      bottom_data[old_idx] = top_data[index];
    }
  }
}

void PermuteLayer::Forward_gpu(const vector<Blob *> &bottom,
                               const vector<Blob *> &top) try {
  if (need_permute_) {
    real_t* bottom_data = bottom[0]->mutable_gpu_data();
    real_t* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = true;
    // NOLINT_NEXT_LINE(whitespace/operators)
    /*
    DPCT1049:35: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto num_axes__ct6 = num_axes_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            PermuteKernel(count, bottom_data, foward, permute_order, old_steps,
                          new_steps, num_axes__ct6, top_data, item_ct1);
          });
    });
    /*
    DPCT1010:36: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    CUDA_POST_KERNEL_CHECK;
  } else {
    // If there is no need to permute
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace caffe
