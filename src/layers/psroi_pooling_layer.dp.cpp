// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "./psroi_pooling_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void PSROIPoolingForward(const int nthreads,
                                    const real_t* bottom_data,
                                    const real_t spatial_scale,
                                    const int channels,
                                    const int height, const int width,
                                    const int pooled_height, const int pooled_width,
                                    const real_t* bottom_rois,
                                    const int output_dim,
                                    const int group_size,
                                    real_t* top_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    real_t roi_start_w =
        static_cast<real_t>(sycl::round((float)(bottom_rois[1]))) *
        spatial_scale;
    real_t roi_start_h =
        static_cast<real_t>(sycl::round((float)(bottom_rois[2]))) *
        spatial_scale;
    real_t roi_end_w =
        static_cast<real_t>(sycl::round((float)(bottom_rois[3])) + 1.) *
        spatial_scale;
    real_t roi_end_h =
        static_cast<real_t>(sycl::round((float)(bottom_rois[4])) + 1.) *
        spatial_scale;

    // Force too small ROIs to be 1x1
    real_t roi_width = sycl::max((float)(roi_end_w - roi_start_w),
                                 (float)(static_cast<real_t>(0.1))); // avoid 0
    real_t roi_height = sycl::max((float)(roi_end_h - roi_start_h),
                                  (float)(static_cast<real_t>(0.1)));

    // Compute w and h at bottom
    real_t bin_size_h = roi_height / static_cast<real_t>(pooled_height);
    real_t bin_size_w = roi_width / static_cast<real_t>(pooled_width);

    int hstart =
        sycl::floor(static_cast<real_t>(ph) * bin_size_h + roi_start_h);
    int wstart =
        sycl::floor(static_cast<real_t>(pw) * bin_size_w + roi_start_w);
    int hend =
        sycl::ceil(static_cast<real_t>(ph + 1) * bin_size_h + roi_start_h);
    int wend =
        sycl::ceil(static_cast<real_t>(pw + 1) * bin_size_w + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = sycl::min(sycl::max(hstart, 0), height);
    hend = sycl::min(sycl::max(hend, 0), height);
    wstart = sycl::min(sycl::max(wstart, 0), width);
    wend = sycl::min(sycl::max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop*group_size + gh)*group_size + gw;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    real_t out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h*width + w;
        out_sum += bottom_data[bottom_index];
      }
    }

    real_t bin_area = (hend - hstart)*(wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum/bin_area;
  }
}

void PSROIPoolingLayer::Forward_gpu(const vector<Blob *> &bottom,
                                    const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bottom_rois = bottom[1]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, static_cast<real_t>(0), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:118: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto spatial_scale__ct2 = spatial_scale_;
    auto channels__ct3 = channels_;
    auto height__ct4 = height_;
    auto width__ct5 = width_;
    auto pooled_height__ct6 = pooled_height_;
    auto pooled_width__ct7 = pooled_width_;
    auto output_dim__ct9 = output_dim_;
    auto group_size__ct10 = group_size_;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          PSROIPoolingForward(count, bottom_data, spatial_scale__ct2,
                              channels__ct3, height__ct4, width__ct5,
                              pooled_height__ct6, pooled_width__ct7,
                              bottom_rois, output_dim__ct9, group_size__ct10,
                              top_data, item_ct1);
        });
  });
  /*
  DPCT1010:119: SYCL uses exceptions to report errors and does not use the error
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
