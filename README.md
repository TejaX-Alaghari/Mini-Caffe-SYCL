# Mini-Caffe-SYCL
Mini Caffe framework with SYCL backend

# Dependencies
The SYCL backend of Mini Caffe has similar dependencies as CUDA version of Mini Caffe which can be found here: https://github.com/luoyetx/mini-caffe#build-on-linux.

Also, note that there's also an optional dependency on OpenCV for building the examples of the project.

# Running on Intel platforms
For running the SYCL backend on Intel platform, Makefile.dpct should be used to build the project. 
And below dependencies need to be fullfilled - 
- oneAPI Base-toolkit should be installed

# Running on Nvidia platform
For running the SYCL backend on Nvidia platform, Makefile_cuda.dpct should be used to build the project. 
And below dependencies need to be fullfilled - 
- To generate the PTX from SYCL backend, the Makefile should be used with Intel LLVM compiler built with CUDA backend - https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda
- In addition to this, the BLAS library used by the backend should also be built for CUDA (Ref: https://github.com/oneapi-src/oneMKL/blob/develop/docs/building_the_project.rst#building-for-cuda)

The Makefile should be updated with appropriate references for "CC", "MKL_HOME" and "OPENCV_ROOT" paths.
Note: The PATH for finding IRC library should be updated accordingly when compiling SYCL backend to run on Nvidia platform.
