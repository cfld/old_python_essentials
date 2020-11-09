cmake_minimum_required(VERSION 3.6)

set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/include/python3.7m")
set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/lib/libpython3.7m.so")
set(ESSENTIALS_INCLUDE_DIR /home/ubuntu/projects/essentials CACHE STRING "Path to essentials includes")

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

link_directories(
  /usr/local/cuda/lib64
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

cuda_add_library(gunrock_sssp SHARED gunrock_sssp.cu /home/ubuntu/projects/essentials/externals/mtx/mmio.cpp)

target_include_directories(
  gunrock_sssp
  PRIVATE
  "${ESSENTIALS_INCLUDE_DIR}/"
  "${ESSENTIALS_INCLUDE_DIR}/externals/moderngpu/src"
  "${ESSENTIALS_INCLUDE_DIR}/externals/rapidjson/include"
  "${ESSENTIALS_INCLUDE_DIR}/externals/mtx"
)

target_link_libraries(gunrock_sssp ${PYTHON_LIBRARIES} cudart)

set_target_properties(gunrock_sssp PROPERTIES PREFIX "")