#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./tile_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void Tile(const int nthreads, const real_t* bottom_data,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    real_t* top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int b = (index / tile_size / num_tiles) % bottom_tile_axis;
    const int n = index / tile_size / num_tiles / bottom_tile_axis;
    const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}

void TileLayer::Forward_gpu(const vector<Blob*>& bottom,
                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const int bottom_tile_axis = bottom[0]->shape(axis_);
  const int nthreads = top[0]->count();
  /*
  DPCT1049:95: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto inner_dim__ct2 = inner_dim_;
    auto tiles__ct3 = tiles_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          Tile(nthreads, bottom_data, inner_dim__ct2, tiles__ct3,
               bottom_tile_axis, top_data, item_ct1);
        });
  });
}

}  // namespace caffe
