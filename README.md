# Mini-Caffe-SYCL
Mini Caffe framework with SYCL backend

# Dependencies
The SYCL backend of Mini Caffe has similar dependencies as CUDA version of Mini Caffe which can be found here: https://github.com/luoyetx/mini-caffe#build-on-linux

# Running on Intel platforms



# Running on Nvidia platform
For running the SYCL backend on Nvidia platform, two more dependencies need to be fullfilled 
- To generate the PTX from SYCL backend, the Makefile should be used with Intel LLVM compiler built with CUDA backend - https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda
- In addition to this, the BLAS library used by the backend should also be built for CUDA (Ref: https://github.com/oneapi-src/oneMKL/blob/develop/docs/building_the_project.rst#building-for-cuda)

The Makefile should be updated with appropriate references for "CC" and "MKL_HOME" paths.
Also, the PATH for IRC library should be updated accordingly.
