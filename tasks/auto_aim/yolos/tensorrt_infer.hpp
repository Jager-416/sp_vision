#ifndef AUTO_AIM__TENSORRT_INFER_HPP
#define AUTO_AIM__TENSORRT_INFER_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace auto_aim
{

// TensorRT推理封装类
// 目标：接口类似OpenVINO，最小化YOLO11的改动
class TensorRTInfer
{
public:
  // 构造函数：加载引擎文件
  TensorRTInfer(const std::string& engine_path);
  
  // 析构函数：自动释放资源
  ~TensorRTInfer();

  // 推理函数：输入图像，输出推理结果
  // img: [640, 640, 3] BGR uint8 (已resize和letterbox)
  // 返回: [40, 8400] float32 的cv::Mat
  cv::Mat infer(const cv::Mat& img);

private:
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  cudaStream_t stream_;

  void* d_input_;
  void* d_output_;
  float* h_input_;
  float* h_output_;

  size_t input_size_;
  size_t output_size_;

  // 预处理：BGR→RGB, /255, HWC→CHW
  void preprocess(const cv::Mat& img, float* output);
};

}  // namespace auto_aim

#endif
