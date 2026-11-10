cmake_minimum_required(VERSION 3.16)
project(video_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ORT_DIR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ORT_DIR $ENV{HOME}/video-infer-deps/onnxruntime-linux-aarch64-1.20.1)
    else()
        set(ORT_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1)
    endif()
endif()

find_package(OpenCV REQUIRED)

add_executable(video_infer
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/draw.cpp
    src/pipeline.cpp
)

target_include_directories(video_infer PRIVATE
    ${ORT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(video_infer PRIVATE
    ${ORT_DIR}/lib/libonnxruntime.so
    ${OpenCV_LIBS}
    pthread
)

set_target_properties(video_infer PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)

