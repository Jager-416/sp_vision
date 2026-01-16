cmake_minimum_required(VERSION 3.16)

find_package(CUDA REQUIRED)

# TensorRT paths for JetPack 6.0
set(TensorRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TensorRT_LIBRARIES nvinfer nvonnxparser nvinfer_plugin nvparsers)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(auto_aim io ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES} tinympcstatic)