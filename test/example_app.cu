#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>

#include <pybind11/pybind11.h>

int example_app(torch::Tensor x) {
  std::cout << x << std::endl;
  return 0;
}

PYBIND11_MODULE(example_app, m) {
  m.def("example_app", example_app);
}
