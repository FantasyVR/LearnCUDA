cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(LearnCUDA LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-gencode;arch=compute_75;code=sm_75")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(HW1)
