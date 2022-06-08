#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "./pooling_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void MaxPoolForward(const int nthreads,
    const real_t* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    real_t* const top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = sycl::min((int)(hstart + kernel_h), height);
    const int wend = sycl::min((int)(wstart + kernel_w), width);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    real_t maxval = -FLT_MAX;
    const real_t* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxval = bottom_slice[h * width + w];
        }
      }
    }
    top_data[index] = maxval;
  }
}

void AvePoolForward(const int nthreads,
    const real_t* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    real_t* const top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min((int)(hstart + kernel_h), (int)(height + pad_h));
    int wend = sycl::min((int)(wstart + kernel_w), (int)(width + pad_w));
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, height);
    wend = sycl::min(wend, width);
    real_t aveval = 0;
    const real_t* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

void PoolingLayer::Forward_gpu(const vector<Blob *> &bottom,
                               const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // NOLINT_NEXT_LINE(whitespace/operators)
    /*
    DPCT1049:82: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto bottom_num_ct2 = bottom[0]->num();
      auto channels__ct3 = channels_;
      auto height__ct4 = height_;
      auto width__ct5 = width_;
      auto pooled_height__ct6 = pooled_height_;
      auto pooled_width__ct7 = pooled_width_;
      auto kernel_h__ct8 = kernel_h_;
      auto kernel_w__ct9 = kernel_w_;
      auto stride_h__ct10 = stride_h_;
      auto stride_w__ct11 = stride_w_;
      auto pad_h__ct12 = pad_h_;
      auto pad_w__ct13 = pad_w_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            MaxPoolForward(count, bottom_data, bottom_num_ct2, channels__ct3,
                           height__ct4, width__ct5, pooled_height__ct6,
                           pooled_width__ct7, kernel_h__ct8, kernel_w__ct9,
                           stride_h__ct10, stride_w__ct11, pad_h__ct12,
                           pad_w__ct13, top_data, item_ct1);
          });
    });
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    /*
    DPCT1049:83: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto bottom_num_ct2 = bottom[0]->num();
      auto channels__ct3 = channels_;
      auto height__ct4 = height_;
      auto width__ct5 = width_;
      auto pooled_height__ct6 = pooled_height_;
      auto pooled_width__ct7 = pooled_width_;
      auto kernel_h__ct8 = kernel_h_;
      auto kernel_w__ct9 = kernel_w_;
      auto stride_h__ct10 = stride_h_;
      auto stride_w__ct11 = stride_w_;
      auto pad_h__ct12 = pad_h_;
      auto pad_w__ct13 = pad_w_;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            AvePoolForward(count, bottom_data, bottom_num_ct2, channels__ct3,
                           height__ct4, width__ct5, pooled_height__ct6,
                           pooled_width__ct7, kernel_h__ct8, kernel_w__ct9,
                           stride_h__ct10, stride_w__ct11, pad_h__ct12,
                           pad_w__ct13, top_data, item_ct1);
          });
    });
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  /*
  DPCT1010:84: SYCL uses exceptions to report errors and does not use the error
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
