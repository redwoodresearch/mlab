cmake_policy(SET CMP0048 NEW)
project(
  cuda_exercises
  VERSION 0.1
  LANGUAGES CUDA CXX)

# needed for cuda features
cmake_minimum_required(VERSION 3.18)

add_compile_options(-Wall -Wextra -Wmissing-field-initializers)

set(RELEASE_DEB_ARGS "${RELEASE_ARGS};-UNDEBUG")

# must be quoted for some reason I don't understand?
add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_DEB_ARGS}>")
add_compile_options("$<$<CONFIG:DEBUG>:-ftrapv;-fno-omit-frame-pointer;-G>")

option(FORCE_COLORED_OUTPUT
       "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if(${FORCE_COLORED_OUTPUT})
  # gnu/clang only
  add_compile_options(-fdiagnostics-color=always)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_example "${PROJECT_SOURCE_DIR}/src/cuda_example.cu")
target_include_directories(cuda_example PUBLIC "${PROJECT_SOURCE_DIR}/include")
# https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(cuda_example PROPERTIES CUDA_ARCHITECTURES "72")

add_executable(exercise_1 "${PROJECT_SOURCE_DIR}/src/exercise_1.cu")
target_include_directories(exercise_1 PUBLIC "${PROJECT_SOURCE_DIR}/include")
# https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(exercise_1 PROPERTIES CUDA_ARCHITECTURES "72")
