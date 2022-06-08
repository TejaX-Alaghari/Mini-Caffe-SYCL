source /opt/intel/oneapi/setvars.sh

COMPILER_PATH=/home/administrator/Milun/jaden/workspace/sycl_workspace
export PATH=${COMPILER_PATH}/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=${COMPILER_PATH}/llvm/build/lib:$LD_LIBRARY_PATH

APP_PATH=/home/administrator/Milun/jaden/workspace/applications/mini-caffe
export LD_LIBRARY_PATH=${APP_PATH}/build_cuda/sycl_cuda:${APP_PATH}/deps/lib/oneMKL/BLAS/lib:$LD_LIBRARY_PATH

export SYCL_DEVICE_FILTER=CUDA
