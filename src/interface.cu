// interface.cu

// ?? Is there a way to automatically unwrap torch::Tensor to pointers, to reduce amount of glue code?

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "gunrock/applications/sssp.hxx"

namespace py = pybind11;
using namespace gunrock;
using namespace memory;

// --
// Builder

template <
  typename graph_type,
  typename vertex_type = typename graph_type::vertex_type,
  typename edge_type   = typename graph_type::edge_type,
  typename weight_type = typename graph_type::weight_type
>
graph_type from_csr(
    vertex_type const& n_vertices,
    edge_type const& n_edges,
    torch::Tensor Ap_arr,
    torch::Tensor Aj_arr,
    torch::Tensor Ax_arr 
) {
  return graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
        n_vertices,                      // rows
        n_vertices,                      // columns
        n_edges,                         // nonzeros
        Ap_arr.data_ptr<edge_type>(),    // row_offsets
        Aj_arr.data_ptr<vertex_type>(),  // column_indices
        Ax_arr.data_ptr<weight_type>()   // values
    );
}

// --
// Apps

template <
  typename graph_type,
  typename vertex_t = typename graph_type::vertex_type,
  typename edge_t   = typename graph_type::edge_type,
  typename weight_t = typename graph_type::weight_type
>
void sssp_run(
  graph_type&   G,
  vertex_t      single_source,
  torch::Tensor distances,
  torch::Tensor predecessors
) {
  sssp::run(
    G,
    single_source,
    distances.data_ptr<weight_t>(),
    predecessors.data_ptr<vertex_t>()
  );
}

PYBIND11_MODULE(pygunrock, m) {
  
  // --
  // Typedefs
  
  using vertex_t = int;
  using edge_t   = int;
  using weight_t = float;
  
  using csr_graph_type = gunrock::graph::graph_t<gunrock::memory::device, vertex_t, edge_t, weight_t, 
    std::conditional_t<true, gunrock::graph::graph_csr_t<vertex_t, edge_t, weight_t>, gunrock::graph::empty_csr_t>, 
    std::conditional_t<false, gunrock::graph::graph_csc_t<vertex_t, edge_t, weight_t>, gunrock::graph::empty_csc_t>, 
    std::conditional_t<false, gunrock::graph::graph_coo_t<vertex_t, edge_t, weight_t>, gunrock::graph::empty_coo_t>
  >;
  
  // --
  // Interface
  
  py::class_<csr_graph_type>(m, "CSRGraph")
    .def("get_number_of_vertices", &csr_graph_type::get_number_of_vertices)
    .def("get_number_of_edges",    &csr_graph_type::get_number_of_edges);
  
  m.def("from_csr", from_csr<csr_graph_type>);
  m.def("sssp", sssp_run<csr_graph_type>);
}
