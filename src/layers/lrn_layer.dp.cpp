#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>

#include "./lrn_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void LRNFillScale(const int nthreads, const real_t* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const real_t alpha_over_size,
    const real_t k, real_t* const scale, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const real_t* const in_off = in + offset;
    real_t* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    real_t accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

void LRNLayer::Forward_gpu(const vector<Blob*>& bottom,
                           const vector<Blob*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

// TODO: check if it would be faster to just put it into the previous kernel.
void LRNComputeOutput(const int nthreads, const real_t* const in,
    const real_t* const scale, const real_t negative_beta, real_t* const out,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * sycl::pow<double>(scale[index], negative_beta);
  }
}

void LRNLayer::CrossChannelForward_gpu(const vector<Blob *> &bottom,
                                       const vector<Blob *> &top) try {
  // First, compute scale
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  real_t* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:38: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto num__ct2 = num_;
    auto channels__ct3 = channels_;
    auto height__ct4 = height_;
    auto width__ct5 = width_;
    auto size__ct6 = size_;
    auto alpha__size__ct7 = alpha_ / size_;
    auto k__ct8 = k_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(n_threads)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          LRNFillScale(n_threads, bottom_data, num__ct2, channels__ct3,
                       height__ct4, width__ct5, size__ct6, alpha__size__ct7,
                       k__ct8, scale_data, item_ct1);
        });
  });
  /*
  DPCT1010:39: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:40: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto beta__ct3 = -beta_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(n_threads)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          LRNComputeOutput(n_threads, bottom_data, scale_data, beta__ct3,
                           top_data, item_ct1);
        });
  });
  /*
  DPCT1010:41: SYCL uses exceptions to report errors and does not use the error
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
