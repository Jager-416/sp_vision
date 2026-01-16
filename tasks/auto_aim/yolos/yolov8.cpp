#include "yolov8.hpp"

#include <fmt/chrono.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "tasks/auto_aim/classifier.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

// TensorRT Logger
namespace {
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
};
Logger gLogger;
}

namespace auto_aim
{
YOLOV8::YOLOV8(const std::string & config_path, bool debug)
: classifier_(config_path), detector_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov8_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // Initialize TensorRT
  cudaStreamCreate(&stream_);

  std::string engine_path = model_path_.substr(0, model_path_.find_last_of('.')) + ".engine";
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
    std::cout << "Building TensorRT engine from ONNX: " << model_path_ << std::endl;
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

    std::string onnx_path = model_path_;
    if (model_path_.substr(model_path_.find_last_of('.')) == ".xml") {
      onnx_path = model_path_.substr(0, model_path_.find_last_of('.')) + ".onnx";
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
      // For YOLOV8: input shape is [batch, channels, height, width] = [1, 3, 416, 416]
      nvinfer1::Dims4 dims(1, 3, 416, 416);
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

  cudaMalloc(&buffers_[input_index_], input_size_ * sizeof(float));
  cudaMalloc(&buffers_[output_index_], output_size_ * sizeof(float));
}

YOLOV8::~YOLOV8()
{
  if (buffers_[input_index_]) cudaFree(buffers_[input_index_]);
  if (buffers_[output_index_]) cudaFree(buffers_[output_index_]);
  if (stream_) cudaStreamDestroy(stream_);
}

std::list<Armor> YOLOV8::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(416) / bgr_img.rows;
  auto y_scale = static_cast<double>(416) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // Preprocess: resize and pad
  auto input_bgr = cv::Mat(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input_bgr(roi), {w, h});

  // Convert BGR to RGB and normalize
  cv::Mat input_rgb;
  cv::cvtColor(input_bgr, input_rgb, cv::COLOR_BGR2RGB);
  input_rgb.convertTo(input_rgb, CV_32F, 1.0 / 255.0);

  // Convert from HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(input_rgb, channels);
  std::vector<float> input_data;
  input_data.reserve(416 * 416 * 3);
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
  cv::Mat output(output_dims.d[1], output_dims.d[2], CV_32F, output_data.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> YOLOV8::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classess
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 14);

    std::vector<cv::Point2f> armor_key_points;

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    for (int i = 0; i < 4; i++) {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    it->pattern = get_pattern(bgr_img, *it);
    classifier_.classify(*it);

    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->type = get_type(*it);
    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV8::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于分类器的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV8::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于分类器的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

ArmorType YOLOV8::get_type(const Armor & armor)
{
  // 英雄、基地只能是大装甲板
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
    return ArmorType::big;
  }

  // 工程、哨兵、前哨站只能是小装甲板
  if (
    armor.name == ArmorName::two || armor.name == ArmorName::sentry ||
    armor.name == ArmorName::outpost) {
    return ArmorType::small;
  }

  // 步兵假设为小装甲板
  return ArmorType::small;
}

cv::Point2f YOLOV8::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

cv::Mat YOLOV8::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  // 延长灯条获得装甲板角点
  // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
  auto tl = (armor.points[0] + armor.points[3]) / 2 - (armor.points[3] - armor.points[0]) * 1.125;
  auto bl = (armor.points[0] + armor.points[3]) / 2 + (armor.points[3] - armor.points[0]) * 1.125;
  auto tr = (armor.points[2] + armor.points[1]) / 2 - (armor.points[2] - armor.points[1]) * 1.125;
  auto br = (armor.points[2] + armor.points[1]) / 2 + (armor.points[2] - armor.points[1]) * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  // 检查ROI是否有效
  if (roi_left < 0 || roi_top < 0 || roi_right <= roi_left || roi_bottom <= roi_top) {
    // std::cerr << "Invalid ROI: " << roi << std::endl;
    return cv::Mat();  // 返回一个空的Mat对象
  }

  // 检查ROI是否超出图像边界
  if (roi_right > bgr_img.cols || roi_bottom > bgr_img.rows) {
    // std::cerr << "ROI out of image bounds: " << roi << " Image size: " << bgr_img.size()
    //           << std::endl;
    return cv::Mat();  // 返回一个空的Mat对象
  }

  return bgr_img(roi);
}

void YOLOV8::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, armor.pattern);
}

void YOLOV8::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV8::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

std::list<Armor> YOLOV8::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim