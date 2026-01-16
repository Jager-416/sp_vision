cmake_minimum_required(VERSION 3.16)

find_package(CUDA REQUIRED)

# TensorRT paths for JetPack 6.0
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TensorRT_LIBRARIES nvinfer nvonnxparser nvinfer_plugin nvparsers)

add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)
target_link_libraries(omniperception ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})