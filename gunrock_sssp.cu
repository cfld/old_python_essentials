#include <torch/extension.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <gunrock/applications/sssp/sssp_implementation.hxx>

namespace py = pybind11;

using namespace gunrock;
using namespace memory;

template <
  typename graph_type,
  typename vertex_t = typename graph_type::vertex_type,
  typename edge_t   = typename graph_type::edge_type,
  typename weight_t = typename graph_type::weight_type
>
graph_type make_graph(
                vertex_t const& n_vertices,
                edge_t const& n_edges,
                torch::Tensor Ap_arr,
                torch::Tensor Aj_arr,
                torch::Tensor Ax_arr) {
  
  graph_type G;
  G.set(
    n_vertices, n_vertices, n_edges,
    Ap_arr.data_ptr<edge_t>(),
    Aj_arr.data_ptr<vertex_t>(),
    Ax_arr.data_ptr<weight_t>()
  );
  return G;
}

template <
  typename graph_type,
  typename vertex_t = typename graph_type::vertex_type,
  typename edge_t   = typename graph_type::edge_type,
  typename weight_t = typename graph_type::weight_type
>
void gunrock_sssp(
  graph_type    G,
  vertex_t      single_source,
  torch::Tensor distances,
  torch::Tensor predecessors
) {
  auto n_vertices = G.get_number_of_vertices();
  auto n_edges    = G.get_number_of_edges();
  auto meta = graph::build::meta_t<vertex_t, edge_t, weight_t>(
    n_vertices, n_vertices, n_edges
  );
  
  // << !! Hack -- wrap graph, otherwise get illegal memory access
  //    !! and .. I think this does a copy?
  typename vector<graph_type, memory_space_t::device>::type G_vec(1);
  gunrock::graph::build::device::csr_t<graph_type>(G, memory::raw_pointer_cast(G_vec.data()));
  // >>
  
  float elapsed = sssp::run(
    G_vec, meta,
    single_source,
    distances.data_ptr<weight_t>(),
    predecessors.data_ptr<vertex_t>()
  );
}


PYBIND11_MODULE(gunrock_sssp, m) {
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;
  
  using graph_type = typename graph::graph_t<
    memory_space_t::device, vertex_t, edge_t, weight_t,
    graph::graph_csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>>;

  py::class_<graph_type>(m, "GunrockGraph")
    .def("get_number_of_vertices", &graph_type::get_number_of_vertices)
    .def("get_number_of_edges", &graph_type::get_number_of_edges);
  
  m.def("make_graph", make_graph<graph_type>);
  
  m.def("gunrock_sssp",  gunrock_sssp<graph_type>);
}
