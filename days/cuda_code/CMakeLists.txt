cmake_policy(SET CMP0048 NEW)
project(
  cuda_exercises
  VERSION 0.1
  LANGUAGES CUDA)

# needed for cuda features
cmake_minimum_required(VERSION 3.18)

add_compile_options(-Wall -Wextra -Wmissing-field-initializers
    --expt-relaxed-constexpr --extended-lambda)

set(RELEASE_DEB_ARGS "${RELEASE_ARGS};-UNDEBUG")

# must be quoted for some reason I don't understand?
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_DEB_ARGS}>")
add_compile_options("$<$<CONFIG:DEBUG>:-ftrapv;-fno-omit-frame-pointer;-G>")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_example "${PROJECT_SOURCE_DIR}/src/cuda_example.cu")
target_include_directories(cuda_example PUBLIC "${PROJECT_SOURCE_DIR}/include")
# https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(cuda_example PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(summing_array "${PROJECT_SOURCE_DIR}/src/summing_array.cu")
target_include_directories(summing_array PUBLIC "${PROJECT_SOURCE_DIR}/include")
set_target_properties(summing_array PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(summing_array_solutions "${PROJECT_SOURCE_DIR}/src/summing_array_solutions.cu")
target_include_directories(summing_array_solutions PUBLIC "${PROJECT_SOURCE_DIR}/include")
set_target_properties(summing_array_solutions PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(filtering_array "${PROJECT_SOURCE_DIR}/src/filtering_array.cu")
target_include_directories(filtering_array PUBLIC "${PROJECT_SOURCE_DIR}/include")
set_target_properties(filtering_array PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(filtering_array_solutions "${PROJECT_SOURCE_DIR}/src/filtering_array_solutions.cu")
target_include_directories(filtering_array_solutions PUBLIC "${PROJECT_SOURCE_DIR}/include")
set_target_properties(filtering_array_solutions PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(reduce_by_key "${PROJECT_SOURCE_DIR}/src/reduce_by_key.cu")
target_include_directories(reduce_by_key PUBLIC "${PROJECT_SOURCE_DIR}/include")
set_target_properties(reduce_by_key PROPERTIES CUDA_ARCHITECTURES "72")
