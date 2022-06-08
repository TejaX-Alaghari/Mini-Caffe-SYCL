#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <algorithm>
#include <functional>
#include <map>
#include <vector>
#include <cfloat>

#include "./bbox_util.hpp"
#include "../common.hpp"

namespace caffe {

template <typename Dtype>
Dtype BBoxSizeGPU(const Dtype* bbox,
    const bool normalized = true) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template <typename Dtype>
Dtype JaccardOverlapGPU(const Dtype* bbox1,
    const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = sycl::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = sycl::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = sycl::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = sycl::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSizeGPU(bbox1);
    const Dtype bbox2_size = BBoxSizeGPU(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template <typename Dtype>
Dtype Min(const Dtype x, const Dtype y) {
  return x < y ? x : y;
}

template <typename Dtype>
Dtype Max(const Dtype x, const Dtype y) {
  return x > y ? x : y;
}

template <typename Dtype>
void ClipBBoxGPU(const Dtype* bbox, Dtype* clip_bbox) {
  for (int i = 0; i < 4; ++i) {
    clip_bbox[i] = Max(Min(bbox[i], Dtype(1.)), Dtype(0.));
  }
}

template <typename Dtype>
void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = sycl::exp((float)xmax) * prior_width;
        decode_bbox_height = sycl::exp((float)ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width = sycl::exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
            sycl::exp(prior_data[vi + 3] * ymax) * prior_height;
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    } else {
      // Unknown code type.
    }
    if (clip_bbox) {
      bbox_data[index] =
          sycl::max(sycl::min(bbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads, const Dtype *loc_data,
                     const Dtype *prior_data, const CodeType code_type,
                     const bool variance_encoded_in_target,
                     const int num_priors, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const bool clip_bbox, Dtype *bbox_data) try {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:25: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        DecodeBBoxesKernel<Dtype>(nthreads, loc_data, prior_data, code_type,
                                  variance_encoded_in_target, num_priors,
                                  share_location, num_loc_classes,
                                  background_label_id, clip_bbox, bbox_data,
                                  item_ct1);
      });
  /*
  DPCT1010:22: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template void DecodeBBoxesGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, float* bbox_data);

template <typename Dtype>
void PermuteDataKernel(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
    new_data[new_index] = data[index];
  }
}

template <typename Dtype>
void PermuteDataGPU(const int nthreads, const Dtype *data,
                    const int num_classes, const int num_data,
                    const int num_dim, Dtype *new_data) try {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:26: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        PermuteDataKernel<Dtype>(nthreads, data, num_classes, num_data, num_dim,
                                 new_data, item_ct1);
      });
  /*
  DPCT1010:23: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template void PermuteDataGPU(const int nthreads,
          const float* data, const int num_classes, const int num_data,
          const int num_dim, float* new_data);

template <typename Dtype>
void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = sycl::max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_data, const Dtype* channel_max,
    Dtype* data, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = channel_data[index] - channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
void kernel_exp(const int count, const Dtype* data, Dtype* out,
                sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = sycl::exp((float)(data[index]));
  }
}

template <typename Dtype>
void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data,
    sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int outer_num,
    const int channels, const int inner_num, Dtype* prob) {
  vector<int> shape(4, 1);
  shape[0] = outer_num;
  shape[1] = channels;
  shape[2] = inner_num;
  Blob scale(shape);
  Dtype* scale_data = scale.mutable_gpu_data();
  int count = outer_num * channels * inner_num;
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:27: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(outer_num * inner_num)) *
              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_channel_max<Dtype>(outer_num, channels, inner_num, data,
                                  scale_data, item_ct1);
      });
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:28: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_channel_subtract<Dtype>(count, outer_num, channels, inner_num,
                                       data, scale_data, prob, item_ct1);
      });
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:29: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_exp<Dtype>(count, prob, prob, item_ct1);
      });
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:30: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(outer_num * inner_num)) *
              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_channel_sum<Dtype>(outer_num, channels, inner_num, prob,
                                  scale_data, item_ct1);
      });
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:31: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(count)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        kernel_channel_div<Dtype>(count, outer_num, channels, inner_num,
                                  scale_data, prob, item_ct1);
      });
}

template void SoftMaxGPU(const float* data, const int outer_num,
    const int channels, const int inner_num, float* prob);

template <typename Dtype>
void ComputeOverlappedKernel(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data,
          sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_bboxes;
    const int i = (index / num_bboxes) % num_bboxes;
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    const int c = (index / num_bboxes / num_bboxes) % num_classes;
    const int n = index / num_bboxes / num_bboxes / num_classes;
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = ((n * num_bboxes + i) * num_classes + c) * 4;
    const int start_loc_j = ((n * num_bboxes + j) * num_classes + c) * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads, const Dtype *bbox_data,
                          const int num_bboxes, const int num_classes,
                          const Dtype overlap_threshold,
                          bool *overlapped_data) try {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:32: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        ComputeOverlappedKernel<Dtype>(nthreads, bbox_data, num_bboxes,
                                       num_classes, overlap_threshold,
                                       overlapped_data, item_ct1);
      });
  /*
  DPCT1010:24: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template void ComputeOverlappedGPU(const int nthreads,
          const float* bbox_data, const int num_bboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);

template <typename Dtype>
void ComputeOverlappedByIdxKernel(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, int* overlapped_data,
          sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_idx;
    const int i = (index / num_idx);
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = idx[i] * 4;
    const int start_loc_j = idx[j] * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = 1;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads, const Dtype *bbox_data,
                               const Dtype overlap_threshold, const int *idx,
                               const int num_idx, int *overlapped_data) try {
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:33: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(nthreads)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        ComputeOverlappedByIdxKernel<Dtype>(nthreads, bbox_data,
                                            overlap_threshold, idx, num_idx,
                                            overlapped_data, item_ct1);
      });
  /*
  DPCT1010:34: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices) {
  // Keep part of detections whose scores are higher than confidence threshold.
  vector<int> idx;
  vector<Dtype> confidences;
  for (int i = 0; i < num_bboxes; ++i) {
    if (conf_data[i] > confidence_threshold) {
      idx.push_back(i);
      confidences.push_back(conf_data[i]);
    }
  }
  int num_remain = confidences.size();
  if (num_remain == 0) {
    return;
  }
  // Sort detections based on score.
  dpct::sort(
      // oneapi::dpl::execution::seq,
      oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
      &confidences[0], &confidences[0] + num_remain, &idx[0],
      std::greater<Dtype>());
  if (top_k > -1 && top_k < num_remain) {
    num_remain = top_k;
  }

  // Compute overlap between remaining detections.
  BlobInt idx_blob(1, 1, 1, num_remain);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(idx.begin(), idx.begin() + num_remain, idx_data);

  BlobInt overlapped(1, 1, num_remain, num_remain);
  const int total_bboxes = overlapped.count();
  int* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedByIdxGPU<Dtype>(total_bboxes, bbox_data, nms_threshold,
      idx_blob.gpu_data(), num_remain, overlapped_data);

  // Do non-maximum suppression based on overlapped results.
  const int* overlapped_results = overlapped.cpu_data();
  vector<int> selected_indices;
  ApplyNMS(overlapped_results, num_remain, &selected_indices);

  // Put back the selected information.
  for (int i = 0; i < selected_indices.size(); ++i) {
    indices->push_back(idx[selected_indices[i]]);
  }
}

template
void ApplyNMSGPU(const float* bbox_data, const float* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

}  // namespace caffe
