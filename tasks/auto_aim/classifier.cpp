#include "classifier.hpp"

#include <yaml-cpp/yaml.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

// TensorRT Logger
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} gLogger;

namespace auto_aim
{
Classifier::Classifier(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model_path = yaml["classify_model"].as<std::string>();
  net_ = cv::dnn::readNetFromONNX(model_path);

  // Initialize TensorRT
  cudaStreamCreate(&stream_);

  // Try to load .engine file first, if not found, build from ONNX
  std::string engine_path = model_path.substr(0, model_path.find_last_of('.')) + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file.good()) {
    // Load serialized engine
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
    // Build engine from ONNX
    std::cout << "Building TensorRT engine from ONNX: " << model_path << std::endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      throw std::runtime_error("Failed to parse ONNX file");
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

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
      // For classifier: input shape is [batch, channels, height, width] = [1, 1, 32, 32]
      nvinfer1::Dims4 dims(1, 1, 32, 32);
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

    // Save engine for future use
    std::ofstream engine_out(engine_path, std::ios::binary);
    engine_out.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    engine_out.close();
    std::cout << "Engine saved to " << engine_path << std::endl;
  }

  // Create execution context
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext(),
    [](nvinfer1::IExecutionContext* c) { if(c) c->destroy(); });

  // Get input/output indices and sizes
  input_index_ = engine_->getBindingIndex("input");
  output_index_ = engine_->getBindingIndex("output");

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

  // Allocate GPU buffers
  cudaMalloc(&buffers_[input_index_], input_size_ * sizeof(float));
  cudaMalloc(&buffers_[output_index_], output_size_ * sizeof(float));
}

Classifier::~Classifier()
{
  // Free GPU buffers
  if (buffers_[input_index_]) cudaFree(buffers_[input_index_]);
  if (buffers_[output_index_]) cudaFree(buffers_[output_index_]);
  if (stream_) cudaStreamDestroy(stream_);
}

void Classifier::classify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  net_.setInput(blob);
  cv::Mat outputs = net_.forward();

  // softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

void Classifier::ovclassify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  // Resize image to 32x32
  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});
  // Normalize the input image to [0, 1] range
  input.convertTo(input, CV_32F, 1.0 / 255.0);

  // Copy input data to GPU
  cudaMemcpyAsync(buffers_[input_index_], input.data, input_size_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Execute inference
  context_->enqueueV2(buffers_, stream_, nullptr);

  // Copy output data from GPU
  std::vector<float> output_data(output_size_);
  cudaMemcpyAsync(output_data.data(), buffers_[output_index_], output_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Create cv::Mat from output data
  cv::Mat outputs(1, 9, CV_32F, output_data.data());

  // Softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

}  // namespace auto_aim