#include "yolo11_buff.hpp"

#include <fstream>
#include <iostream>

const double ConfidenceThreshold = 0.7f;
const double IouThreshold = 0.4f;

// TensorRT Logger
namespace {
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT BUFF] " << msg << std::endl;
  }
};
Logger gLogger;
}
namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path = yaml["model"].as<std::string>();

  // Initialize TensorRT
  cudaStreamCreate(&stream_);

  std::string engine_path = model_path.substr(0, model_path.find_last_of('.')) + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  // Create runtime (must outlive engine)
  runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(gLogger),
    [](nvinfer1::IRuntime* r) { if(r) r->destroy(); });

  if (engine_file.good()) {
    std::cout << "Loading TensorRT engine from " << engine_path << std::endl;
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(engine_data.data(), size),
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

    auto config_builder = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config_builder->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

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
      // For YOLO11_BUFF: input shape is [batch, channels, height, width] = [1, 3, 640, 640]
      nvinfer1::Dims4 dims(1, 3, 640, 640);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
      config_builder->addOptimizationProfile(profile);
    }

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config_builder));

    if (!serialized_engine) {
      throw std::runtime_error("Failed to build TensorRT engine");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()),
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

  cudaMalloc(&buffers_[input_index_], input_size_ * sizeof(float));
  cudaMalloc(&buffers_[output_index_], output_size_ * sizeof(float));
}

YOLO11_BUFF::~YOLO11_BUFF()
{
  if (buffers_[input_index_]) cudaFree(buffers_[input_index_]);
  if (buffers_[output_index_]) cudaFree(buffers_[output_index_]);
  if (stream_) cudaStreamDestroy(stream_);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();  // 设置模型输入

  /// 预处理

  // const float factor = fill_tensor_data_image(input_tensor, image);  // 填充图片到合适的input size

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<YOLO11_BUFF::Object> ();
  }

  cv::Mat bgr_img = image;

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  double factor = 1.0 / scale;  // 将640x640空间坐标转换回原始图像空间的放大因子  

  // Preprocess: resize and pad
  auto input_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input_bgr(roi), {w, h});

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

  // Copy input to GPU
  cudaMemcpyAsync(buffers_[input_index_], input_data.data(), input_size_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Execute inference
  context_->enqueueV2(buffers_, stream_, nullptr);

  // Copy output from GPU
  std::vector<float> output_data(output_size_);
  cudaMemcpyAsync(output_data.data(), buffers_[output_index_], output_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Get output dimensions
  auto output_dims = engine_->getBindingDimensions(output_index_);
  const int out_rows = output_dims.d[1];  // rows
  const int out_cols = output_dims.d[2];  // cols
  const cv::Mat det_output(out_rows, out_cols, CV_32F, output_data.data());
  std::vector<cv::Rect> boxes;                            // 目标框
  std::vector<float> confidences;                         // 置信度
  std::vector<std::vector<float>> objects_keypoints;      // 关键点
  // 输出格式是[15,8400], 每列代表一个框(即最多有8400个框), 前面4行分别是[cx, cy, ow, oh], 中间score, 最后5*2关键点(3代表每个关键点的信息, 包括[x, y, visibility],如果是2，则没有visibility)
  // 15 = 4 + 1 + NUM_POINTS * 2      56
  for (int i = 0; i < det_output.cols; ++i) {
    const float score = det_output.at<float>(4, i);
    // 如果置信度满足条件则放进vector
    if (score > ConfidenceThreshold) {
      // 获取目标框
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      cv::Rect box;
      box.x = static_cast<int>((cx - 0.5 * ow) * factor);
      box.y = static_cast<int>((cy - 0.5 * oh) * factor);
      box.width = static_cast<int>(ow * factor);
      box.height = static_cast<int>(oh * factor);
      boxes.push_back(box);

      // 获取置信度
      confidences.push_back(score);

      // 获取关键点
      std::vector<float> keypoints;
      cv::Mat kpts = det_output.col(i).rowRange(NUM_POINTS, 15);
      for (int j = 0; j < NUM_POINTS; ++j) {
        const float x = kpts.at<float>(j * 2 + 0, 0) * factor;
        const float y = kpts.at<float>(j * 2 + 1, 0) * factor;
        // const float s = kpts.at<float>(j * 3 + 2, 0);
        keypoints.push_back(x);
        keypoints.push_back(y);
        // keypoints.push_back(s);
      }
      objects_keypoints.push_back(keypoints);
    }
  }

  /// NMS,消除具有较低置信度的冗余重叠框,用于处理多个框的情况
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);

  std::vector<Object> object_result;  // 最终得到的object
  for (size_t i = 0; i < indexes.size(); ++i) {
    Object obj;
    const int index = indexes[i];
    obj.rect = boxes[index];
    obj.prob = confidences[index];

    const std::vector<float> & keypoint = objects_keypoints[index];
    for (int i = 0; i < NUM_POINTS; ++i) {
      const float x_coord = keypoint[i * 2];
      const float y_coord = keypoint[i * 2 + 1];
      obj.kpt.push_back(cv::Point2f(x_coord, y_coord));
    }
    object_result.push_back(obj);

    /// 绘制关键点和连线
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);            // 绘制矩形框
    const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);  // 绘制标签
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
      obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    const int radius = 2;  // 绘制关键点
    const cv::Size & shape = image.size();
    for (int i = 0; i < NUM_POINTS; ++i)
      cv::circle(image, obj.kpt[i], radius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
  }
  /// 计算FPS
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);

  // #ifdef SAVE
  //         save("save", image);
  // #endif
  return object_result;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<YOLO11_BUFF::Object>();
  }

  cv::Mat bgr_img = image;

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  double factor = 1.0 / scale;  // 将640x640空间坐标转换回原始图像空间的放大因子

  // Preprocess: resize and pad
  auto input_bgr = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input_bgr(roi), {w, h});

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

  // Copy input to GPU
  cudaMemcpyAsync(buffers_[input_index_], input_data.data(), input_size_ * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // Execute inference
  context_->enqueueV2(buffers_, stream_, nullptr);

  // Copy output from GPU
  std::vector<float> output_data(output_size_);
  cudaMemcpyAsync(output_data.data(), buffers_[output_index_], output_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Get output dimensions
  auto output_dims = engine_->getBindingDimensions(output_index_);
  const int out_rows = output_dims.d[1];
  const int out_cols = output_dims.d[2];
  const cv::Mat det_output(out_rows, out_cols, CV_32F, output_data.data());

  /// 寻找置信度最大的框
  int best_index = -1;
  float max_confidence = 0.0f;
  for (int i = 0; i < det_output.cols; ++i) {
    const float confidence = det_output.at<float>(4, i);
    if (confidence > max_confidence) {
      max_confidence = confidence;
      best_index = i;
    }
  }

  std::vector<Object> object_result;
  if (max_confidence > ConfidenceThreshold) {
    Object obj;
    // 获取目标框
    const float cx = det_output.at<float>(0, best_index);
    const float cy = det_output.at<float>(1, best_index);
    const float ow = det_output.at<float>(2, best_index);
    const float oh = det_output.at<float>(3, best_index);
    obj.rect.x = static_cast<int>((cx - 0.5 * ow) * factor);
    obj.rect.y = static_cast<int>((cy - 0.5 * oh) * factor);
    obj.rect.width = static_cast<int>(ow * factor);
    obj.rect.height = static_cast<int>(oh * factor);
    // 获取置信度
    obj.prob = max_confidence;
    // 获取关键点
    cv::Mat kpts = det_output.col(best_index).rowRange(5, 5 + NUM_POINTS * 2);
    for (int i = 0; i < NUM_POINTS; ++i) {
      const float x = kpts.at<float>(i * 2 + 0, 0) * factor;
      const float y = kpts.at<float>(i * 2 + 1, 0) * factor;
      obj.kpt.push_back(cv::Point2f(x, y));
    }
    object_result.push_back(obj);

    /// 0.3-0.7 save
    if (max_confidence < 0.7) save(std::to_string(start), image);

    /// 绘制关键点和连线
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
    const std::string label = "buff:" + std::to_string(max_confidence).substr(0, 4);
    const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
    const cv::Rect textBox(
      obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
    cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
    cv::putText(
      image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 0));
    const int radius = 2;
    const cv::Size & shape = image.size();
    for (int i = 0; i < NUM_POINTS; ++i) {
      cv::circle(image, obj.kpt[i], radius, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
      cv::putText(
        image, std::to_string(i + 1), obj.kpt[i] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }
  }

  /// 计算FPS
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return object_result;
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
  const std::filesystem::path saveDir = "../result/";
  if (!std::filesystem::exists(saveDir)) {
    std::filesystem::create_directories(saveDir);
  }
  const std::filesystem::path savePath = saveDir / (programName + ".jpg");
  cv::imwrite(savePath.string(), image);
}
}  // namespace auto_buff