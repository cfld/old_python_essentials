#include <torch/extension.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace py = pybind11;

using namespace gunrock;
using namespace memory;

template<typename vertex_t, typename edge_t, typename weight_t>
void gunrock_sssp(
  vertex_t      n_vertices,
  edge_t        n_edges,
  torch::Tensor offsets_arr,
  torch::Tensor indices_arr,
  torch::Tensor data_arr,
  vertex_t      single_source,
  torch::Tensor distances_arr,
  torch::Tensor predecessors_arr
) {
  auto d_offsets      = offsets_arr.data_ptr<vertex_t>();
  auto d_indices      = indices_arr.data_ptr<edge_t>();
  auto d_data         = data_arr.data_ptr<weight_t>();
  auto d_distances    = distances_arr.data_ptr<weight_t>();
  auto d_predecessors = predecessors_arr.data_ptr<vertex_t>();
  
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
}


PYBIND11_MODULE(gunrock_sssp, m) {
  m.def("gunrock_sssp", gunrock_sssp<int,int,float>);
}
