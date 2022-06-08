#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include "./conv_dw_layer.hpp"
//#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
void ConvolutionDepthwiseWeightForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const weight_data, const int num, const int channels,
    const int top_height, const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
    Dtype* const top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh)
    {
      for (int kw = 0; kw < kernel_w; ++kw)
      {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width))
        {
          const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <typename Dtype>
void ConvolutionDepthwiseBiasForward(const int nthreads,
    const Dtype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const top_data,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}

void ConvolutionDepthwiseLayer::Forward_gpu(const vector<Blob*>& bottom,
                                            const vector<Blob*>& top) {
  const real_t* bottom_data = bottom[0]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  const real_t* weight_data = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  /*
  DPCT1049:42: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto kernel_h__ct9 = kernel_h_;
    auto kernel_w__ct10 = kernel_w_;
    auto stride_h__ct11 = stride_h_;
    auto stride_w__ct12 = stride_w_;
    auto pad_h__ct13 = pad_h_;
    auto pad_w__ct14 = pad_w_;
    auto dilation_h__ct15 = dilation_h_;
    auto dilation_w__ct16 = dilation_w_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          ConvolutionDepthwiseWeightForward(
              count, bottom_data, weight_data, num, channels, top_height,
              top_width, bottom_height, bottom_width, kernel_h__ct9,
              kernel_w__ct10, stride_h__ct11, stride_w__ct12, pad_h__ct13,
              pad_w__ct14, dilation_h__ct15, dilation_w__ct16, top_data,
              item_ct1);
        });
  });
  if (this->layer_param_.convolution_param().bias_term()) {
    const real_t* bias_data = this->blobs_[1]->gpu_data();
    /*
    DPCT1049:43: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          ConvolutionDepthwiseBiasForward(count, bias_data, num, channels,
                                          top_height, top_width, top_data,
                                          item_ct1);
        });
  }
}

}  // namespace caffe
