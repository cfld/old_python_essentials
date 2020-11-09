#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace py = pybind11;

using namespace gunrock;
using namespace memory;

// --
// Helpers

template <typename T>
auto numpy2cuda(py::array_t<T> x) {
  py::buffer_info ha = x.request();
  T* x_hptr          = reinterpret_cast<T*>(ha.ptr);
  
  T* x_dptr;
  cudaMalloc(&x_dptr, ha.shape[0] * sizeof(T));
  cudaMemcpy(x_dptr, x_hptr, ha.shape[0] * sizeof(T), cudaMemcpyHostToDevice);
  return x_dptr;
}

template <typename T>
void cuda2numpy(py::array_t<T> x, T* x_dptr) {
  py::buffer_info ha = x.request();
  T* x_hptr          = reinterpret_cast<T*>(ha.ptr);

  cudaMemcpy(x_hptr, x_dptr, ha.shape[0] * sizeof(T), cudaMemcpyDeviceToHost);
}

// --
// Runner

template<typename vertex_t, typename edge_t, typename weight_t>
void gunrock_sssp(
  vertex_t              n_vertices,
  edge_t                n_edges,
  py::array_t<vertex_t> offsets_arr,
  py::array_t<edge_t>   indices_arr,
  py::array_t<weight_t> data_arr,
  vertex_t              single_source,
  py::array_t<weight_t> distances_arr,
  py::array_t<vertex_t> predecessors_arr
) {
  
  // Copy data to GPU
  auto d_offsets      = numpy2cuda(offsets_arr);
  auto d_indices      = numpy2cuda(indices_arr);
  auto d_data         = numpy2cuda(data_arr);
  auto d_distances    = numpy2cuda(distances_arr);
  auto d_predecessors = numpy2cuda(predecessors_arr);
  
  // Build graph + meta
  auto G = graph::build::from_csr_t<memory_space_t::device>(
    n_vertices,
    n_vertices,
    n_edges,
    d_offsets,
    d_indices,
    d_data
  );

  auto meta = graph::build::meta_t<vertex_t, edge_t, weight_t>(
    n_vertices,
    n_vertices,
    n_edges
  );

  // Run
  float elapsed = sssp::run(
    G,
    meta,
    single_source,
    d_distances,
    d_predecessors
  );
  
  // Copy results back to numpy
  cuda2numpy(distances_arr, d_distances);
  cuda2numpy(predecessors_arr, d_predecessors);
  
  // Free memory
  cudaFree(d_offsets);
  cudaFree(d_indices);
  cudaFree(d_data);
  cudaFree(d_distances);
  cudaFree(d_predecessors);
}

PYBIND11_MODULE(gunrock_sssp, m) {
  m.def("gunrock_sssp", gunrock_sssp<int,int,float>);
}
