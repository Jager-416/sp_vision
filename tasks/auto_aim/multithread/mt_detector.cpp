#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>

// TensorRT Logger
namespace {
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT MT] " << msg << std::endl;
  }
};
Logger gLogger;
}

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug), current_request_idx_(0)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  // Load or build TensorRT engine
  std::string engine_path = model_path.substr(0, model_path.find_last_of('.')) + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file.good()) {
    std::cout << "Loading TensorRT engine from " << engine_path << std::endl;
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_data.data(), size),
      [](nvinfer1::ICudaEngine* e) { if(e) e->destroy(); });
  } else {
    std::cout << "Building TensorRT engine from ONNX: " << model_path << std::endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    std::string onnx_path = model_path;
    if (model_path.substr(model_path.find_last_of('.')) == ".xml") {
      onnx_path = model_path.substr(0, model_path.find_last_of('.')) + ".onnx";
    }

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      throw std::runtime_error("Failed to parse ONNX file: " + onnx_path);
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Check if model has dynamic shapes and create optimization profile
    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();
    bool has_dynamic_shapes = false;
    for (int i = 0; i < input_dims.nbDims; i++) {
      if (input_dims.d[i] == -1) {
        has_dynamic_shapes = true;
        break;
      }
    }

    if (has_dynamic_shapes) {
      auto profile = builder->createOptimizationProfile();
      // For YOLO: input shape is [batch, channels, height, width] = [1, 3, 640, 640]
      nvinfer1::Dims4 dims(1, 3, 640, 640);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
      config->addOptimizationProfile(profile);
    }

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));

    if (!serialized_engine) {
      throw std::runtime_error("Failed to build TensorRT engine");
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()),
      [](nvinfer1::ICudaEngine* e) { if(e) e->destroy(); });

    std::ofstream engine_out(engine_path, std::ios::binary);
    engine_out.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    engine_out.close();
    std::cout << "Engine saved to " << engine_path << std::endl;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext(),
    [](nvinfer1::IExecutionContext* c) { if(c) c->destroy(); });

  input_index_ = engine_->getBindingIndex("images");
  if (input_index_ == -1) input_index_ = 0;
  output_index_ = engine_->getBindingIndex("output0");
  if (output_index_ == -1) output_index_ = 1;

  auto input_dims = engine_->getBindingDimensions(input_index_);
  auto output_dims = engine_->getBindingDimensions(output_index_);

  input_size_ = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_size_ *= input_dims.d[i];
  }

  output_size_ = 1;
  for (int i = 0; i < output_dims.nbDims; i++) {
    output_size_ *= output_dims.d[i];
  }

  // Create multiple inference requests for async processing
  infer_requests_.resize(16);
  for (auto& req : infer_requests_) {
    cudaMalloc(&req.input_buffer, input_size_ * sizeof(float));
    cudaMalloc(&req.output_buffer, output_size_ * sizeof(float));
    cudaStreamCreate(&req.stream);
    cudaEventCreate(&req.event);
    req.input_size = input_size_;
    req.output_size = output_size_;
  }

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

MultiThreadDetector::~MultiThreadDetector()
{
  for (auto& req : infer_requests_) {
    if (req.input_buffer) cudaFree(req.input_buffer);
    if (req.output_buffer) cudaFree(req.output_buffer);
    if (req.stream) cudaStreamDestroy(req.stream);
    if (req.event) cudaEventDestroy(req.event);
  }
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // Preprocess
  auto input_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input_bgr(roi), {w, h});

  // Convert BGR to RGB and normalize
  cv::Mat input_rgb;
  cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);
  input_rgb.convertTo(input_rgb, CV_32F, 1.0 / 255.0);

  // Convert from HWC to CHW
  std::vector<cv::Mat> channels(3);
  cv::split(input_rgb, channels);
  std::vector<float> input_data;
  input_data.reserve(640 * 640 * 3);
  for (int c = 0; c < 3; ++c) {
    input_data.insert(input_data.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
  }

  // Get current inference request
  size_t req_idx = current_request_idx_;
  current_request_idx_ = (current_request_idx_ + 1) % infer_requests_.size();
  auto& req = infer_requests_[req_idx];

  // Copy input to GPU
  cudaMemcpyAsync(req.input_buffer, input_data.data(), req.input_size * sizeof(float),
                  cudaMemcpyHostToDevice, req.stream);

  // Execute inference asynchronously
  void* buffers[2] = {req.input_buffer, req.output_buffer};
  context_->enqueueV2(buffers, req.stream, nullptr);

  // Record event for synchronization
  cudaEventRecord(req.event, req.stream);

  queue_.push({img.clone(), t, req_idx});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, req_idx] = queue_.pop();
  auto& req = infer_requests_[req_idx];

  // Wait for inference to complete
  cudaEventSynchronize(req.event);

  // Copy output from GPU
  std::vector<float> output_data(req.output_size);
  cudaMemcpyAsync(output_data.data(), req.output_buffer, req.output_size * sizeof(float),
                  cudaMemcpyDeviceToHost, req.stream);
  cudaStreamSynchronize(req.stream);

  // Get output dimensions
  auto output_dims = engine_->getBindingDimensions(output_index_);
  cv::Mat output(output_dims.d[1], output_dims.d[2], CV_32F, output_data.data());

  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);

  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, req_idx] = queue_.pop();
  auto& req = infer_requests_[req_idx];

  // Wait for inference to complete
  cudaEventSynchronize(req.event);

  // Copy output from GPU
  std::vector<float> output_data(req.output_size);
  cudaMemcpyAsync(output_data.data(), req.output_buffer, req.output_size * sizeof(float),
                  cudaMemcpyDeviceToHost, req.stream);
  cudaStreamSynchronize(req.stream);

  // Get output dimensions
  auto output_dims = engine_->getBindingDimensions(output_index_);
  cv::Mat output(output_dims.d[1], output_dims.d[2], CV_32F, output_data.data());

  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto armors = yolo_.postprocess(scale, output, img, 0);

  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
