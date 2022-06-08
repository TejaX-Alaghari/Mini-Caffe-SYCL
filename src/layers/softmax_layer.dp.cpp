#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "./softmax_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const real_t* data, real_t* out,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    real_t maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = sycl::max((float)(data[(n * channels + c) * spatial_dim + s]),
                         (float)maxval);
    }
    out[index] = maxval;
  }
}

void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const real_t* channel_max, real_t* data,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

void kernel_exp(const int count, const real_t* data, real_t* out,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = sycl::exp((float)(data[index]));
  }
}

void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const real_t* data, real_t* channel_sum,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    real_t sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const real_t* channel_sum, real_t* data,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const real_t* data_1, const real_t* data_2,
    real_t* channel_dot, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    real_t dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

void SoftmaxLayer::Forward_gpu(const vector<Blob*>& bottom,
                               const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  real_t* scale_data = scale_.mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:134: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto outer_num__ct0 = outer_num_;
    auto inner_num__ct2 = inner_num_;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(outer_num_ * inner_num_)) *
                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_channel_max(outer_num__ct0, channels, inner_num__ct2, top_data,
                             scale_data, item_ct1);
        });
  });
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:135: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto outer_num__ct1 = outer_num_;
    auto inner_num__ct3 = inner_num_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_channel_subtract(count, outer_num__ct1, channels,
                                  inner_num__ct3, scale_data, top_data,
                                  item_ct1);
        });
  });
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:136: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_exp(count, top_data, top_data, item_ct1);
      });
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:137: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto outer_num__ct0 = outer_num_;
    auto inner_num__ct2 = inner_num_;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(outer_num_ * inner_num_)) *
                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_channel_sum(outer_num__ct0, channels, inner_num__ct2, top_data,
                             scale_data, item_ct1);
        });
  });
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:138: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto outer_num__ct1 = outer_num_;
    auto inner_num__ct3 = inner_num_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel_channel_div(count, outer_num__ct1, channels, inner_num__ct3,
                             scale_data, top_data, item_ct1);
        });
  });
}

}  // namespace caffe
