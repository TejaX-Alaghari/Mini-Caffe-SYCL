#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <vector>

#include "./shuffle_channel_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void ShuffleChannelKernel(const int nthreads, const int feature_map_size,
	                                   real_t *output, const real_t *input,
                                     int group_row, int group_column, int len,
                                     sycl::nd_item<3> item_ct1) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / group_row / group_column;
		const int i = (index / group_column) % group_row;
		const int j = index % group_column;

		const real_t* p_i = input + n * feature_map_size + (i * group_column + j) * len;
		real_t* p_o = output + n * feature_map_size + (j * group_row + i) * len;

		for (int k = 0; k < len; k++)
			p_o[k] = p_i[k];
	}
}

void ShuffleChannelLayer::Forward_gpu(const vector<Blob*>& bottom,
                                      const vector<Blob*>& top) {
    const real_t* bottom_data = bottom[0]->gpu_data();
    real_t* top_data = top[0]->mutable_gpu_data();

    const int num = bottom[0]->num();
    const int feature_map_size = bottom[0]->count(1);
    const int sp_sz = bottom[0]->count(2);
    const int chs = bottom[0]->channels();

    int group_row = group_;
    int group_column = int(chs / group_row);
    CHECK_EQ(chs, (group_column * group_row)) << "Wrong group size.";
	  int count = num * group_column * group_row;
          /*
          DPCT1049:37: The workgroup size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the workgroup size if
          needed.
          */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        ShuffleChannelKernel(count, feature_map_size, top_data, bottom_data,
                             group_row, group_column, sp_sz, item_ct1);
      });
}

}  // namespace caffe
