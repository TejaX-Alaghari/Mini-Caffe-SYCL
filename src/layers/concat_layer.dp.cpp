#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./concat_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void Concat(const int nthreads, const real_t* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, real_t* out_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

void ConcatLayer::Forward_gpu(const vector<Blob*>& bottom,
                              const vector<Blob*>& top) {
  if (bottom.size() == 1) { return; }
  real_t* top_data = top[0]->mutable_gpu_data();
  int offset_concat_axis = 0;
  const int top_concat_axis = top[0]->shape(concat_axis_);
  const bool kForward = true;
  for (int i = 0; i < bottom.size(); ++i) {
    const real_t* bottom_data = bottom[i]->gpu_data();
    const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
    const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    const int nthreads = bottom_concat_size * num_concats_;
    /*
    DPCT1049:50: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto num_concats__ct3 = num_concats_;
      auto concat_input_size__ct4 = concat_input_size_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            Concat(nthreads, bottom_data, kForward, num_concats__ct3,
                   concat_input_size__ct4, top_concat_axis, bottom_concat_axis,
                   offset_concat_axis, top_data, item_ct1);
          });
    });
    offset_concat_axis += bottom_concat_axis;
  }
}

}  // namespace caffe
