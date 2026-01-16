#include "tensorrt_infer.hpp"
#include <fstream>
#include <iostream>
#include "tools/logger.hpp"

namespace auto_aim
{

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

// 构造函数
TensorRTInfer::TensorRTInfer(const std::string& engine_path)
{
  // 读取引擎文件
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Cannot open engine: " + engine_path);
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  // 反序列化引擎
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  context_ = engine_->createExecutionContext();

  // 分配内存
  input_size_ = 1 * 3 * 640 * 640 * sizeof(float);
  output_size_ = 1 * 40 * 8400 * sizeof(float);

  cudaMalloc(&d_input_, input_size_);
  cudaMalloc(&d_output_, output_size_);
  cudaMallocHost((void**)&h_input_, input_size_);
  cudaMallocHost((void**)&h_output_, output_size_);

  cudaStreamCreate(&stream_);

  tools::logger()->info("[TensorRT] Initialized successfully");
}

// 析构函数
TensorRTInfer::~TensorRTInfer()
{
  cudaStreamDestroy(stream_);
  cudaFreeHost(h_input_);
  cudaFreeHost(h_output_);
  cudaFree(d_input_);
  cudaFree(d_output_);
  
  delete context_;
  delete engine_;
  delete runtime_;
}

// 预处理函数
void TensorRTInfer::preprocess(const cv::Mat& img, float* output)
{
  // img是 [640,640,3] BGR uint8
  int channels = 3;
  int height = 640;
  int width = 640;

  // BGR→RGB, /255.0, HWC→CHW
  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // BGR→RGB: 2-c 表示 B→R, G→G, R→B
        output[c * height * width + h * width + w] = 
          img.at<cv::Vec3b>(h, w)[2 - c] / 255.0f;
      }
    }
  }
}

// 推理函数
cv::Mat TensorRTInfer::infer(const cv::Mat& img)
{
  // 1. 预处理
  preprocess(img, h_input_);

  // 2. 拷贝到GPU
  cudaMemcpyAsync(d_input_, h_input_, input_size_, 
                  cudaMemcpyHostToDevice, stream_);

  // 3. 推理
  void* bindings[] = {d_input_, d_output_};
  context_->enqueueV2(bindings, stream_, nullptr);

  // 4. 拷贝回CPU
  cudaMemcpyAsync(h_output_, d_output_, output_size_, 
                  cudaMemcpyDeviceToHost, stream_);

  // 5. 同步
  cudaStreamSynchronize(stream_);

  // 6. 转换为cv::Mat [40, 8400]
  cv::Mat output(40, 8400, CV_32F, h_output_);
  
  return output.clone();  // 返回副本
}

}  // namespace auto_aim
