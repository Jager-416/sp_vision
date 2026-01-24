cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2025.4.0/runtime/cmake")

find_package(CUDA REQUIRED)
find_package(Ceres REQUIRED)

# TensorRT paths for JetPack 6.0
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TensorRT_LIBRARIES nvinfer nvonnxparser nvinfer_plugin nvparsers)

add_library(auto_buff OBJECT 
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    yolo11_buff.cpp
)

target_link_libraries(auto_buff auto_aim ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} ${CERES_LIBRARIES})