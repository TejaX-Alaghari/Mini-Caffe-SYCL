#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./slice_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void Slice(const int nthreads, const real_t* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, real_t* out_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

void SliceLayer::Forward_gpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top) {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const real_t* bottom_data = bottom[0]->gpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    real_t* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    /*
    DPCT1049:49: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto num_slices__ct3 = num_slices_;
      auto slice_size__ct4 = slice_size_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            Slice(nthreads, bottom_data, kForward, num_slices__ct3,
                  slice_size__ct4, bottom_slice_axis, top_slice_axis,
                  offset_slice_axis, top_data, item_ct1);
          });
    });
    offset_slice_axis += top_slice_axis;
  }
}

}  // namespace caffe
