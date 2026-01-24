#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <string>
#include <memory>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{
class Classifier
{
public:
  explicit Classifier(const std::string & config_path);
  ~Classifier();

  void classify(Armor & armor);

  void ovclassify(Armor & armor);

private:
  cv::dnn::Net net_;

  // TensorRT members (runtime must outlive engine)
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  void* buffers_[2]; // input and output buffers
  cudaStream_t stream_;
  int input_index_;
  int output_index_;
  size_t input_size_;
  size_t output_size_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP