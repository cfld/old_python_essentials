// pygraph.cuh

#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace gunrock;
using namespace memory;

template <typename graph_type, typename meta_type>
struct PyGunrockGraph {
  using vertex_t = typename graph_type::vertex_type;
  using edge_t   = typename graph_type::edge_type;
  using weight_t = typename graph_type::weight_type;

  std::shared_ptr<graph_type> G;
  std::shared_ptr<meta_type> meta;
  
  PyGunrockGraph(
    vertex_t const& n_vertices,
    edge_t const& n_edges,
    torch::Tensor Ap_arr,
    torch::Tensor Aj_arr,
    torch::Tensor Ax_arr
  ) {

    G = graph::build::_from_csr_t<memory_space_t::device>( // !! Hack until we add C++17 support
      n_vertices, n_vertices, n_edges,
      Ap_arr.data_ptr<edge_t>(),
      Aj_arr.data_ptr<vertex_t>(),
      Ax_arr.data_ptr<weight_t>()
    );

    meta = graph::build::_from_csr_t<memory_space_t::host, edge_t, vertex_t, weight_t>(
      n_vertices, n_vertices, n_edges,
      nullptr, nullptr, nullptr
    );
  }

  vertex_t get_number_of_vertices() { return meta->get_number_of_vertices();}
  vertex_t get_number_of_edges()    { return meta->get_number_of_edges();}    // !! Presumably there's some better way to pass things through to `meta`
};