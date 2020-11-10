cmake_minimum_required(VERSION 3.6)
project(example_app)

# For Pytorch support
set(CMAKE_PREFIX_PATH "/home/ubuntu/anaconda3/pkgs/pytorch-1.7.0-py3.7_cuda11.0.221_cudnn8.0.3_0/lib/python3.7/site-packages/torch")
set(CUDNN_LIBRARY_PATH "$ENV{HOME}/anaconda3/pkgs/cudnn-8.0.0-cuda11.0_0/lib/libcudnn.so")
set(CUDNN_INCLUDE_PATH "$ENV{HOME}/anaconda3/pkgs/cudnn-8.0.0-cuda11.0_0/include")

set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/include/python3.7m")
set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/lib/libpython3.7m.so")
# set(ESSENTIALS_INCLUDE_DIR /home/ubuntu/projects/essentials CACHE STRING "Path to essentials includes")

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

link_directories(
  /usr/local/cuda/lib64
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++14")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

cuda_add_library(example_app SHARED example_app.cu)

# target_include_directories(
#   gunrock_sssp
#   PRIVATE
#   "${ESSENTIALS_INCLUDE_DIR}/"
#   "${ESSENTIALS_INCLUDE_DIR}/externals/moderngpu/src"
#   "${ESSENTIALS_INCLUDE_DIR}/externals/rapidjson/include"
#   "${ESSENTIALS_INCLUDE_DIR}/externals/mtx"
#   "${TORCH_INCLUDE_DIRS}"
# )

target_link_libraries(example_app ${PYTHON_LIBRARIES} cudart ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

set_target_properties(example_app PROPERTIES PREFIX "")
# set_property(TARGET example_app PROPERTY CXX_STANDARD 14)