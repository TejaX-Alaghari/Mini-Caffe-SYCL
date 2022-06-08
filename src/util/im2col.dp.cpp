#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>

#include "./im2col.hpp"
#include "../common.hpp"

namespace caffe {

template <typename Dtype>
void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                Dtype *data_col) try {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:55: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        im2col_gpu_kernel<Dtype>(num_kernels, data_im, height, width, kernel_h,
                                 kernel_w, pad_h, pad_w, stride_h, stride_w,
                                 dilation_h, dilation_w, height_col, width_col,
                                 data_col, item_ct1);
      });
  /*
  DPCT1010:52: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);

template <typename Dtype, int num_axes>
void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col, sycl::nd_item<3> item_ct1,
    int *shared_dilation, int *shared_kernel_shape, int *shared_pad,
    int *shared_stride, int *shared_col_shape, int *shared_im_shape) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)

  if (item_ct1.get_local_id(2) < num_axes) {
    shared_dilation[item_ct1.get_local_id(2)] =
        dilation[item_ct1.get_local_id(2)];
    shared_kernel_shape[item_ct1.get_local_id(2)] =
        kernel_shape[item_ct1.get_local_id(2)];
    shared_pad[item_ct1.get_local_id(2)] = pad[item_ct1.get_local_id(2)];
    shared_stride[item_ct1.get_local_id(2)] = stride[item_ct1.get_local_id(2)];
  }
  if (item_ct1.get_local_id(2) < num_axes + 1) {
    shared_col_shape[item_ct1.get_local_id(2)] =
        col_shape[item_ct1.get_local_id(2)];
    shared_im_shape[item_ct1.get_local_id(2)] =
        im_shape[item_ct1.get_local_id(2)];
  }
  /*
  DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void im2col_nd_gpu(const Dtype *data_im, const int num_spatial_axes,
                   const int num_kernels, const int *im_shape,
                   const int *col_shape, const int *kernel_shape,
                   const int *pad, const int *stride, const int *dilation,
                   Dtype *data_col) try {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    /*
    DPCT1049:57: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(1 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(1 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 1>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 2:
    /*
    DPCT1049:58: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(2 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(2 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 2>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 3:
    /*
    DPCT1049:59: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(3 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(3 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 3>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 4:
    /*
    DPCT1049:60: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(4 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(4 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 4>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 5:
    /*
    DPCT1049:61: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(5 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(5 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 5>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 6:
    /*
    DPCT1049:62: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(6 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(6 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 6>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 7:
    /*
    DPCT1049:63: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(7 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(7 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 7>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 8:
    /*
    DPCT1049:64: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(8 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(8 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 8>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 9:
    /*
    DPCT1049:65: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(9 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(9 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 9>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  case 10:
    /*
    DPCT1049:66: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(10 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(10 + 1), cgh);

      cgh.parallel_for(sycl::nd_range<3>(
                           sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                               sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                           sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         im2col_nd_gpu_kernel<Dtype, 10>(
                             num_kernels, data_im, im_shape, col_shape,
                             kernel_shape, pad, stride, dilation, data_col,
                             item_ct1, shared_dilation_acc_ct1.get_pointer(),
                             shared_kernel_shape_acc_ct1.get_pointer(),
                             shared_pad_acc_ct1.get_pointer(),
                             shared_stride_acc_ct1.get_pointer(),
                             shared_col_shape_acc_ct1.get_pointer(),
                             shared_im_shape_acc_ct1.get_pointer());
                       });
    });
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  /*
  DPCT1010:51: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);

template <typename Dtype>
void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im, sycl::nd_item<3> item_ct1) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = sycl::min((int)(w_im / stride_w + 1), width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = sycl::min((int)(h_im / stride_h + 1), height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype *data_col, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                Dtype *data_im) try {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  /*
  DPCT1049:67: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                        sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
      [=](sycl::nd_item<3> item_ct1) {
        col2im_gpu_kernel<Dtype>(num_kernels, data_col, height, width, channels,
                                 kernel_h, kernel_w, pad_h, pad_w, stride_h,
                                 stride_w, dilation_h, dilation_w, height_col,
                                 width_col, data_im, item_ct1);
      });
  /*
  DPCT1010:54: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);

template <typename Dtype, int num_axes>
void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im, sycl::nd_item<3> item_ct1,
    int *shared_dilation, int *shared_kernel_shape, int *shared_pad,
    int *shared_stride, int *shared_col_shape, int *shared_im_shape) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)

  if (item_ct1.get_local_id(2) < num_axes) {
    shared_dilation[item_ct1.get_local_id(2)] =
        dilation[item_ct1.get_local_id(2)];
    shared_kernel_shape[item_ct1.get_local_id(2)] =
        kernel_shape[item_ct1.get_local_id(2)];
    shared_pad[item_ct1.get_local_id(2)] = pad[item_ct1.get_local_id(2)];
    shared_stride[item_ct1.get_local_id(2)] = stride[item_ct1.get_local_id(2)];
  }
  if (item_ct1.get_local_id(2) < num_axes + 1) {
    shared_col_shape[item_ct1.get_local_id(2)] =
        col_shape[item_ct1.get_local_id(2)];
    shared_im_shape[item_ct1.get_local_id(2)] =
        im_shape[item_ct1.get_local_id(2)];
  }
  /*
  DPCT1065:68: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] = sycl::min((int)(d_im[i] / shared_stride[i] + 1),
                               shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype *data_col, const int num_spatial_axes,
                   const int im_size, const int *im_shape, const int *col_shape,
                   const int *kernel_shape, const int *pad, const int *stride,
                   const int *dilation, Dtype *data_im) try {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    /*
    DPCT1049:69: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(1 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(1 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 1>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 2:
    /*
    DPCT1049:70: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(2), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(2 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(2 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 2>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 3:
    /*
    DPCT1049:71: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(3), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(3 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(3 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 3>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 4:
    /*
    DPCT1049:72: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(4), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(4 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(4 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 4>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 5:
    /*
    DPCT1049:73: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(5), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(5 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(5 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 5>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 6:
    /*
    DPCT1049:74: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(6), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(6 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(6 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 6>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 7:
    /*
    DPCT1049:75: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(7), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(7 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(7 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 7>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 8:
    /*
    DPCT1049:76: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(8), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(8 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(8 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 8>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 9:
    /*
    DPCT1049:77: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(9), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(9 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(9 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 9>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  case 10:
    /*
    DPCT1049:78: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_dilation_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_kernel_shape_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_pad_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_stride_acc_ct1(sycl::range<1>(10), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_col_shape_acc_ct1(sycl::range<1>(10 + 1), cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local>
          shared_im_shape_acc_ct1(sycl::range<1>(10 + 1), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(im_size)) *
                                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                            sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
            col2im_nd_gpu_kernel<Dtype, 10>(
                im_size, data_col, im_shape, col_shape, kernel_shape, pad,
                stride, dilation, data_im, item_ct1,
                shared_dilation_acc_ct1.get_pointer(),
                shared_kernel_shape_acc_ct1.get_pointer(),
                shared_pad_acc_ct1.get_pointer(),
                shared_stride_acc_ct1.get_pointer(),
                shared_col_shape_acc_ct1.get_pointer(),
                shared_im_shape_acc_ct1.get_pointer());
          });
    });
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  /*
  DPCT1010:53: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_POST_KERNEL_CHECK;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);

}  // namespace caffe
