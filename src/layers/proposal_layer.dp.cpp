#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "./proposal_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

static
int TransformBBox(real_t* box,
                  const real_t dx, const real_t dy,
                  const real_t d_log_w, const real_t d_log_h,
                  const real_t img_width, const real_t img_height,
                  const real_t min_box_size) {
  // width & height of box
  const real_t w = box[2] - box[0] + 1;
  const real_t h = box[3] - box[1] + 1;
  // center location of box
  const real_t ctr_x = box[0] + 0.5f * w;
  const real_t ctr_y = box[1] + 0.5f * h;

  // new center location according to gradient (dx, dy)
  const real_t pred_ctr_x = dx * w + ctr_x;
  const real_t pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const real_t pred_w = sycl::exp((float)d_log_w) * w;
  const real_t pred_h = sycl::exp((float)d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - 0.5f * pred_w;
  box[1] = pred_ctr_y - 0.5f * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + 0.5f * pred_w;
  box[3] = pred_ctr_y + 0.5f * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = sycl::max((float)(static_cast<real_t>(0)),
                     sycl::min((float)(box[0]), (float)(img_width - 1)));
  box[1] = sycl::max((float)(static_cast<real_t>(0)),
                     sycl::min((float)(box[1]), (float)(img_height - 1)));
  box[2] = sycl::max((float)(static_cast<real_t>(0)),
                     sycl::min((float)(box[2]), (float)(img_width - 1)));
  box[3] = sycl::max((float)(static_cast<real_t>(0)),
                     sycl::min((float)(box[3]), (float)(img_height - 1)));

  // recompute new width & height
  const real_t box_w = box[2] - box[0] + 1;
  const real_t box_h = box[3] - box[1] + 1;

  // check if new box's size >= threshold
  return (box_w >= min_box_size) && (box_h >= min_box_size);
}

/*! \brief sort rois by score */
static void SortBBox(real_t* rois, const int left, const int right,
                     const int num_top) {
  int first = left;
  int last = right;
  auto __Copy__ = [](real_t* from, real_t* to) {
    for (int i = 0; i < 5; i++) {
      to[i] = from[i];
    }
  };
  real_t key[5];
  __Copy__(rois + 5 * first, key);
  while (first < last) {
    while (first < last && rois[last * 5 + 4] <= key[4]) last--;
    __Copy__(rois + 5 * last, rois + 5 * first);
    while (first < last && rois[first * 5 + 4] >= key[4]) first++;
    __Copy__(rois + 5 * first, rois + 5 * last);
  }
  // first == last
  __Copy__(key, rois + 5 * first);
  // sort [left, first)
  if (left < first - 1) SortBBox(rois, left, first - 1, num_top);
  // sort (first, right], if first >= num_top, no need for rest
  if (first + 1 < num_top && first + 1 < right) SortBBox(rois, first + 1, right, num_top);
}

static
void GenerataProposalsGPU(const int num_proposals,
                          const real_t* score_map,
                          const real_t* bbox_map,
                          const real_t anchors[],
                          real_t* proposals,
                          const int num_anchors,
                          const int fm_height, const int fm_width,
                          const real_t img_height, const real_t img_width,
                          const real_t min_bbox_size, const int feat_stride,
                          sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num_proposals) {
    const int h = index / num_anchors / fm_width;
    const int w = (index / num_anchors) % fm_width;
    const int k = index % num_anchors;
    const real_t x = w * feat_stride;
    const real_t y = h * feat_stride;
    const real_t* box = bbox_map + h * fm_width + w;
    const real_t* score = score_map + h * fm_width + w;

    const int fm_stride = fm_height * fm_width;
    const real_t dx = box[(k * 4 + 0) * fm_stride];
    const real_t dy = box[(k * 4 + 1) * fm_stride];
    const real_t d_log_w = box[(k * 4 + 2) * fm_stride];
    const real_t d_log_h = box[(k * 4 + 3) * fm_stride];

    real_t* proposal = proposals + index * 5;
    proposal[0] = x + anchors[k * 4 + 0];
    proposal[1] = y + anchors[k * 4 + 1];
    proposal[2] = x + anchors[k * 4 + 2];
    proposal[3] = y + anchors[k * 4 + 3];
    proposal[4] = TransformBBox(proposal, dx, dy, d_log_w, d_log_h,
                                img_width, img_height, min_bbox_size) *
                      score[k * fm_stride];
  }
}

static
void RetrieveRoisGPU(const int num_rois,
                     const real_t* proposals,
                     const int* roi_indices,
                     real_t* rois,
                     real_t* roi_scores,
                     sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, num_rois) {
    const real_t* proposal = proposals + roi_indices[index] * 5;
    rois[index * 5 + 0] = 0;
    rois[index * 5 + 1] = proposal[0];
    rois[index * 5 + 2] = proposal[1];
    rois[index * 5 + 3] = proposal[2];
    rois[index * 5 + 4] = proposal[3];
    if (roi_scores) {
      roi_scores[index] = proposal[4];
    }
  }
}

inline
real_t IoU(const real_t* A, const real_t* B) {
  const real_t x1 = sycl::max((float)(A[0]), (float)(B[0]));
  const real_t y1 = sycl::max((float)(A[1]), (float)(B[1]));
  const real_t x2 = sycl::min((float)(A[2]), (float)(B[2]));
  const real_t y2 = sycl::min((float)(A[3]), (float)(B[3]));
  const real_t w =
      sycl::max((float)(static_cast<real_t>(0)), (float)(x2 - x1 + 1));
  const real_t h =
      sycl::max((float)(static_cast<real_t>(0)), (float)(y2 - y1 + 1));
  const real_t s = w * h;
  const real_t sA = (A[2]-A[0]+1)*(A[3]-A[1]+1);
  const real_t sB = (B[2]-B[0]+1)*(B[3]-B[1]+1);
  return s / (sA + sB - s);
}

#define DIVUP(x, y) (((x) + (y) - 1) / (y))
static const int kThreadsPerBlock = sizeof(unsigned long long)*8;

static
void NMSKernel(const int num_rois, const real_t nms_th,
               const real_t* rois, unsigned long long* mask,
               sycl::nd_item<3> item_ct1, real_t *block_rois) {
  const int row_start = item_ct1.get_group(1) * kThreadsPerBlock;
  const int col_start = item_ct1.get_group(2) * kThreadsPerBlock;
  const int row_end = sycl::min((int)(num_rois - row_start), kThreadsPerBlock);
  const int col_end = sycl::min((int)(num_rois - col_start), kThreadsPerBlock);

  const int tid = item_ct1.get_local_id(2);
  if (tid < col_end) {
    block_rois[tid*4 + 0] = rois[(col_start + tid)*5 + 0];
    block_rois[tid*4 + 1] = rois[(col_start + tid)*5 + 1];
    block_rois[tid*4 + 2] = rois[(col_start + tid)*5 + 2];
    block_rois[tid*4 + 3] = rois[(col_start + tid)*5 + 3];
  }
  /*
  DPCT1065:121: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (tid < row_end) {
    const int cur_roi_idx = row_start + tid;
    const real_t* cur_roi = rois + cur_roi_idx*5;
    unsigned long long t = 0;
    int start = (row_start == col_start) ? tid+1 : 0;
    for (int i = start; i < col_end; i++) {
      if (IoU(cur_roi, block_rois + i*4) > nms_th) {
        t |= 1ULL << i;
      }
    }
    const int num_blocks = DIVUP(num_rois, kThreadsPerBlock);
    mask[cur_roi_idx * num_blocks + item_ct1.get_group(2)] = t;
  }
}

static void NonMaximumSuppressionGPU(const int num_proposals,
                                     const real_t *proposals, BlobInt *mask,
                                     int *rois_indices, int &num_rois,
                                     const real_t nms_th,
                                     const int max_num_rois) try {
  const int num_blocks = DIVUP(num_proposals, kThreadsPerBlock);
  const sycl::range<3> blocks(1, num_blocks, num_blocks);
  vector<int> mask_shape(2);
  mask_shape[0] = num_proposals;
  mask_shape[1] = num_blocks*sizeof(unsigned long long)/sizeof(int);
  mask->Reshape(mask_shape);
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<real_t, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        block_rois_acc_ct1(sycl::range<1>(256 /*kThreadsPerBlock*4*/), cgh);

    auto mask_mutable_gpu_data_ct3 =
        reinterpret_cast<unsigned long long *>(mask->mutable_gpu_data());

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * sycl::range<3>(1, 1, kThreadsPerBlock),
                          sycl::range<3>(1, 1, kThreadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          NMSKernel(num_proposals, nms_th, proposals, mask_mutable_gpu_data_ct3,
                    item_ct1, block_rois_acc_ct1.get_pointer());
        });
  });
  /*
  DPCT1010:122: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
  const unsigned long long* mask_cpu = reinterpret_cast<const unsigned long long*>(mask->cpu_data());
  vector<unsigned long long> remv(num_blocks, 0);
  int num_to_keep = 0;
  for (int i = 0; i < num_proposals; i++) {
    const int row = i / kThreadsPerBlock;
    const int col = i % kThreadsPerBlock;
    if (!(remv[row] & (1ULL<<col))) {
      rois_indices[num_to_keep++] = i;
      if (num_to_keep == max_num_rois) break;
      const unsigned long long* p = mask_cpu + i * num_blocks;
      for (int j = row; j < num_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  num_rois = num_to_keep;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ProposalLayer::Forward_gpu(const vector<Blob *> &bottom,
                                const vector<Blob *> &top) try {
  const real_t* anchors_score_map = bottom[0]->gpu_data();
  const real_t* anchors_bbox_map = bottom[1]->gpu_data();
  const real_t* im_info = bottom[2]->cpu_data();
  real_t* rois = top[0]->mutable_gpu_data();
  real_t* rois_score = (top.size() > 1) ? top[1]->mutable_gpu_data() : nullptr;

  CHECK_EQ(bottom[0]->shape(0), 1) << "Only support single scale.";

  // bottom shape: (2 x num_anchors) x H x W
  const int fm_height = bottom[0]->height();
  const int fm_width = bottom[0]->width();
  // input image height & width
  const real_t img_height = im_info[0];
  const real_t img_width = im_info[1];
  // scale factor for height & width
  const real_t scale_factor = im_info[2];
  // minimum box width & height
  const real_t min_bbox_size = min_size_ * scale_factor;
  // number of all proposals = num_anchors * H * W
  const int num_proposals = anchors_.shape(0) * fm_height * fm_width;
  // number of top-n proposals before NMS
  const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
  // number of final RoIs
  int num_rois = 0;

  // enumerate all proposals
  //   num_proposals = num_anchors * H * W
  //   (x1, y1, x2, y2, score) for each proposal
  // NOTE: for bottom, only foreground scores are passed
  // also clip bbox inside bbox boundary and filter bbox with min_bbox_size
  vector<int> proposals_shape{num_proposals, 5};
  proposals_.Reshape(proposals_shape);
  /*
  DPCT1049:123: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto anchors__gpu_data_ct3 = anchors_.gpu_data();
    auto proposals__mutable_gpu_data_ct4 = proposals_.mutable_gpu_data();
    auto anchors__shape_ct5 = anchors_.shape(0);
    auto feat_stride__ct11 = feat_stride_;

    cgh.parallel_for(sycl::nd_range<3>(
                         sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_proposals)) *
                             sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                         sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                     [=](sycl::nd_item<3> item_ct1) {
                       GenerataProposalsGPU(
                           num_proposals, anchors_score_map + num_proposals,
                           anchors_bbox_map, anchors__gpu_data_ct3,
                           proposals__mutable_gpu_data_ct4, anchors__shape_ct5,
                           fm_height, fm_width, img_height, img_width,
                           min_bbox_size, feat_stride__ct11, item_ct1);
                     });
  });
  /*
  DPCT1010:124: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;

  SortBBox(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn);

  NonMaximumSuppressionGPU(pre_nms_topn, proposals_.mutable_gpu_data(),
                           &nms_mask_, roi_indices_.mutable_cpu_data(),
                           num_rois, nms_thresh_, post_nms_topn_);

  /*
  DPCT1049:125: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    auto proposals__gpu_data_ct1 = proposals_.gpu_data();
    auto roi_indices__gpu_data_ct2 = roi_indices_.gpu_data();

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_rois)) *
                              sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                          sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
          RetrieveRoisGPU(num_rois, proposals__gpu_data_ct1,
                          roi_indices__gpu_data_ct2, rois, rois_score,
                          item_ct1);
        });
  });
  /*
  DPCT1010:126: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;

  // reshape if num_rois < post_nms_topn_
  vector<int> top_shape{num_rois, 5};
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace caffe
