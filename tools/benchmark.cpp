#include <stdlib.h>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>
#include <random>
#include <memory>

#include <caffe/net.hpp>
#include <caffe/profiler.hpp>

#include <opencv2/core/core.hpp>

void do_normal(int argc, char *argv[]) {
  CHECK_EQ(argc, 5) << "[Usage]: ./benchmark net.prototxt net.caffemodel iterations gpu_id";
  std::string proto = argv[1];
  std::string caffemodel = argv[2];
  int iters = atoi(argv[3]);
  int gpu_id = atoi(argv[4]);
  LOG(INFO) << "net prototxt: " << proto;
  LOG(INFO) << "net caffemodel: " << caffemodel;
  LOG(INFO) << "net forward iterations: " << iters;
  LOG(INFO) << "run on device " << gpu_id;

  if (gpu_id >= 0 && caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, gpu_id);
    std::cout << "Selected GPU device" << std::endl;
  }
  else {
    gpu_id = -1;
  }

  caffe::Net net(proto);
  net.CopyTrainedLayersFrom(caffemodel);
  caffe::Profiler* profiler = caffe::Profiler::Get();
  profiler->TurnON();
  for (int i = 0; i < iters; i++) {
    uint64_t tic = profiler->Now();
    net.Forward();
    uint64_t toc = profiler->Now();
    LOG(INFO) << "Forward costs " << (toc - tic) / 1000. << " ms";
  }
  profiler->TurnOFF();
  profiler->DumpProfile("./profile.json");
}

void do_special(int argc, char *argv[]) {
  CHECK_EQ(argc, 5) << "[Usage]: ./benchmark net.prototxt net.caffemodel iterations gpu_id";
  std::string proto = argv[1];
  std::string caffemodel = argv[2];
  int iters = atoi(argv[3]);
  int gpu_id = atoi(argv[4]);
  LOG(INFO) << "net prototxt: " << proto;
  LOG(INFO) << "net caffemodel: " << caffemodel;
  LOG(INFO) << "net forward iterations: " << iters;
  LOG(INFO) << "run on device " << gpu_id;

  if (gpu_id >= 0 && caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, gpu_id);
    std::cout << "Selected GPU device" << std::endl;
  }
  else {
    gpu_id = -1;
  }

  caffe::Net net(proto);
  net.CopyTrainedLayersFrom(caffemodel);
  caffe::Profiler* profiler = caffe::Profiler::Get();
  profiler->TurnON();
  
  // random noise
  srand(time(NULL));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> nd(0, 1);
  auto input = net.blob_by_name("data");
  input->Reshape({10, 3, 224, 224});
  float *data = input->mutable_cpu_data();
  const int n = input->count();
  for (int i = 0; i < n; ++i) {
    data[i] = nd(gen);
  }
  
  for (int i = 0; i < iters; i++) {
    uint64_t tic = profiler->Now();
    net.Forward();
    uint64_t toc = profiler->Now();
    LOG(INFO) << "Forward costs " << (toc - tic) / 1000. << " ms";
  }
  profiler->TurnOFF();
  profiler->DumpProfile("./profile.json");
}

int main(int argc, char *argv[]) {

  do_normal(argc, argv);

  // do_special(argc, argv);
  
  return 0;
}
