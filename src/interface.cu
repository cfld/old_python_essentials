// interface.cu

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "gunrock/applications/sssp/sssp_implementation.hxx"
#include "pygraph.cuh"

namespace py = pybind11;
using namespace gunrock;
using namespace memory;

template <
  typename pygraph_t,
  typename vertex_t = typename pygraph_t::vertex_t,
  typename edge_t   = typename pygraph_t::edge_t,
  typename weight_t = typename pygraph_t::weight_t
>
void sssp_run(
  pygraph_t&    PyG,
  vertex_t      single_source,
  torch::Tensor distances,
  torch::Tensor predecessors
) {
  sssp::run(
    PyG.G,
    PyG.meta,
    single_source,
    distances.data_ptr<weight_t>(),
    predecessors.data_ptr<vertex_t>()
  );
}

PYBIND11_MODULE(pygunrock, m) {

  // --
  // Types

  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;

  using graph_type = typename graph::graph_t<
    memory_space_t::device, vertex_t, edge_t, weight_t,
    graph::graph_csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>>;

  using meta_type = typename graph::graph_t<
    memory_space_t::host, vertex_t, edge_t, weight_t,
    graph::graph_csr_t<memory_space_t::host, vertex_t, edge_t, weight_t>>;

  using pygraph_t = PyGunrockGraph<graph_type, meta_type>;

  // --
  // Classes

  py::class_<pygraph_t>(m, "Graph")
    .def(py::init<vertex_t, edge_t, torch::Tensor, torch::Tensor, torch::Tensor>())
    .def("get_number_of_vertices", &pygraph_t::get_number_of_vertices)
    .def("get_number_of_edges",    &pygraph_t::get_number_of_edges);

  m.def("sssp", sssp_run<pygraph_t>);
}
