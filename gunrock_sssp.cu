#include <torch/extension.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace py = pybind11;

using namespace gunrock;
using namespace memory;

using graph_type = graph::graph_t<
    memory_space_t::device, int, int, float,
    graph::graph_csr_t<memory_space_t::device, int, int, float>>;

template <typename edge_t,
          typename vertex_t,
          typename weight_t>
graph_type make_graph(
                vertex_t const& r,
                vertex_t const& c,
                edge_t const& nnz,
                torch::Tensor Ap_arr,
                torch::Tensor Aj_arr,
                torch::Tensor Ax_arr) {
  
  graph_type G;
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
  graph_type  G,
  vertex_t      single_source,
  torch::Tensor distances_arr,
  torch::Tensor predecessors_arr
) {
  auto d_distances    = distances_arr.data_ptr<weight_t>();
  auto d_predecessors = predecessors_arr.data_ptr<vertex_t>();
  
  auto meta = graph::build::meta_t<vertex_t, edge_t, weight_t>(
    G.get_number_of_vertices(), G.get_number_of_vertices(), G.get_number_of_edges()
  );
  
  // << !! Hack -- wrap graph in vector, otherwise get illegal memory access
  typename vector<graph_type, memory_space_t::device>::type G_vec(1);
  gunrock::graph::build::device::csr_t<graph_type>(G, memory::raw_pointer_cast(G_vec.data()));
  // >>
  
  float elapsed = sssp::run(
    G_vec, meta,
    single_source,
    d_distances, d_predecessors
  );
}


PYBIND11_MODULE(gunrock_sssp, m) {
  py::class_<graph_type>(m, "Graph_Device")
    .def("get_number_of_vertices", &graph_type::get_number_of_vertices)
    .def("get_number_of_edges", &graph_type::get_number_of_edges);
  
  m.def("make_graph", make_graph<int,int,float>);
  
  m.def("gunrock_sssp",  gunrock_sssp<int,int,float>);
}
