#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "./roi_pooling_layer.hpp"
#include <cmath>

using sycl::max;
using sycl::min;

namespace caffe {

void ROIPoolForward(const int nthreads, const real_t* bottom_data,
                               const real_t spatial_scale, const int channels, const int height,
                               const int width, const int pooled_height, const int pooled_width,
                               const real_t* bottom_rois, real_t* top_data,
                               sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = sycl::round(bottom_rois[1] * spatial_scale);
    int roi_start_h = sycl::round(bottom_rois[2] * spatial_scale);
    int roi_end_w = sycl::round(bottom_rois[3] * spatial_scale);
    int roi_end_h = sycl::round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = sycl::max((int)(roi_end_w - roi_start_w + 1), 1);
    int roi_height = sycl::max((int)(roi_end_h - roi_start_h + 1), 1);
    real_t bin_size_h = static_cast<real_t>(roi_height) /
                        static_cast<real_t>(pooled_height);
    real_t bin_size_w = static_cast<real_t>(roi_width) /
                        static_cast<real_t>(pooled_width);

    int hstart =
        static_cast<int>(sycl::floor(static_cast<real_t>(ph) * bin_size_h));
    int wstart =
        static_cast<int>(sycl::floor(static_cast<real_t>(pw) * bin_size_w));
    int hend =
        static_cast<int>(sycl::ceil(static_cast<real_t>(ph + 1) * bin_size_h));
    int wend =
        static_cast<int>(sycl::ceil(static_cast<real_t>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = sycl::min(sycl::max((int)(hstart + roi_start_h), 0), height);
    hend = sycl::min(sycl::max((int)(hend + roi_start_h), 0), height);
    wstart = sycl::min(sycl::max((int)(wstart + roi_start_w), 0), width);
    wend = sycl::min(sycl::max((int)(wend + roi_start_w), 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    real_t maxval = is_empty ? 0 : -FLT_MAX;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        maxval = sycl::max((float)maxval, (float)(bottom_data[bottom_index]));
      }
    }
    top_data[index] = maxval;
  }
}

void ROIPoolingLayer::Forward_gpu(const vector<Blob *> &bottom,
                                  const vector<Blob *> &top) try {
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* bottom_rois = bottom[1]->gpu_data();
  real_t* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:128: The workgroup size passed to the SYCL kernel may exceed the
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

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          ROIPoolForward(count, bottom_data, spatial_scale__ct2, channels__ct3,
                         height__ct4, width__ct5, pooled_height__ct6,
                         pooled_width__ct7, bottom_rois, top_data, item_ct1);
        });
  });
  /*
  DPCT1010:129: SYCL uses exceptions to report errors and does not use the error
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
