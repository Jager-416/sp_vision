#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <memory>
#include <tuple>

#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

struct InferRequest {
  void* input_buffer;
  void* output_buffer;
  cudaStream_t stream;
  cudaEvent_t event;
  size_t input_size;
  size_t output_size;
};

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::string device_;
  YOLO yolo_;
  int input_index_;
  int output_index_;
  size_t input_size_;
  size_t output_size_;

  std::vector<InferRequest> infer_requests_;
  size_t current_request_idx_;

  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, size_t>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP