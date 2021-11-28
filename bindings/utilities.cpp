#include "utilities.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

namespace py = pybind11;
using namespace fdml::utilities;
using namespace fdml::utilities::kernel_pca;
using namespace fdml::utilities::sobol;
PYBIND11_MODULE(_utilities, module) {
	module.def("standardize", &standardize, py::arg("X"));
	module.def("rmse", &rmse, py::arg("Ypred"), py::arg("Ytrue"));

	module.def("generate_sobol", &generate_sobol, py::arg("N"), py::arg("D"));

	py::class_<KernelPCA> mod(module, "KernelPCA");
	mod
		.def(py::init<unsigned int, std::string>(), py::arg("n_components"), py::arg("kernel"))
		.def("transform", &KernelPCA::transform)
		.def_readwrite("gamma", &KernelPCA::gamma)
		.def_readwrite("constant", &KernelPCA::constant);

}
