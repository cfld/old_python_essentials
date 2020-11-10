#include <torch/extension.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace py = pybind11;

using namespace gunrock;
using namespace memory;

using d_graph_type = graph::graph_t<
    memory_space_t::device, int, int, float,
    graph::graph_csr_t<memory_space_t::device, int, int, float>>;

using h_graph_type = graph::graph_t<
    memory_space_t::host, int, int, float,
    graph::graph_csr_t<memory_space_t::host, int, int, float>>;

template <typename edge_t,
          typename vertex_t,
          typename weight_t>
d_graph_type d_graph(vertex_t const& r,
                vertex_t const& c,
                edge_t const& nnz,
                torch::Tensor Ap_arr,
                torch::Tensor Aj_arr,
                torch::Tensor Ax_arr) {
  
  d_graph_type G;
  G.set(
    r, c, nnz,
    Ap_arr.data_ptr<edge_t>(),
    Aj_arr.data_ptr<vertex_t>(),
    Ax_arr.data_ptr<weight_t>()
  );  
  return G;
}

template <typename edge_t,
          typename vertex_t,
          typename weight_t>
h_graph_type h_graph(vertex_t const& r,
                vertex_t const& c,
                edge_t const& nnz,
                torch::Tensor Ap_arr,
                torch::Tensor Aj_arr,
                torch::Tensor Ax_arr) {
  
  h_graph_type G;
  G.set(
    r, c, nnz,
    Ap_arr.data_ptr<edge_t>(),
    Aj_arr.data_ptr<vertex_t>(),
    Ax_arr.data_ptr<weight_t>()
  );  
  return G;
}

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
  
  auto G = graph::build::from_csr_t<memory_space_t::device>(
    n_vertices, n_vertices, n_edges,
    d_offsets, d_indices, d_data
  );

  auto meta = graph::build::meta_t<vertex_t, edge_t, weight_t>(
    n_vertices, n_vertices, n_edges
  );

  float elapsed = sssp::run(
    G, meta,
    single_source,
    d_distances, d_predecessors
  );
}

template<typename vertex_t, typename edge_t, typename weight_t>
void gunrock_sssp2(
  vertex_t      n_vertices,
  edge_t        n_edges,
  d_graph_type  G,
  vertex_t      single_source,
  torch::Tensor distances_arr,
  torch::Tensor predecessors_arr
) {
  auto d_distances    = distances_arr.data_ptr<weight_t>();
  auto d_predecessors = predecessors_arr.data_ptr<vertex_t>();

  auto meta = graph::build::meta_t<vertex_t, edge_t, weight_t>(
    n_vertices, n_vertices, n_edges
  );
  
  std::cout << "n_vertices: " << meta[0].get_number_of_vertices() << std::endl;
  
  float elapsed = sssp::run2(
    &G, meta,
    single_source,
    d_distances, d_predecessors
  );
}


PYBIND11_MODULE(gunrock_sssp, m) {
  py::class_<d_graph_type>(m, "Graph_Device")
    .def("get_number_of_vertices", &d_graph_type::get_number_of_vertices)
    .def("get_number_of_edges", &d_graph_type::get_number_of_edges);

  py::class_<h_graph_type>(m, "Graph_Host")
    .def("get_number_of_vertices", &h_graph_type::get_number_of_vertices)
    .def("get_number_of_edges", &h_graph_type::get_number_of_edges);
  
  m.def("gunrock_sssp",  gunrock_sssp<int,int,float>);
  m.def("gunrock_sssp2", gunrock_sssp2<int,int,float>);
  
  m.def("h_graph",   h_graph<int,int,float>);
  m.def("d_graph",   d_graph<int,int,float>);
}
