#include "./deep_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"

namespace py = pybind11;
void wrap_deep(py::module module) {
    using namespace fdml::base_models;
    using namespace fdml::deep_models::gaussian_process;


    py::class_<GPNode, GP> mNode(module, "CGPNode");
    mNode
        .def(py::init<shared_ptr<Kernel>>(), py::arg("kernel"))
        .def(py::init<shared_ptr<Kernel>, double&, double&>(), py::arg("kernel"), py::arg("likelihood_variance"), py::arg("scale"))
        .def(py::init<shared_ptr<Kernel>, Parameter<double>&, Parameter<double>&>(), py::arg("kernel"), py::arg("likelihood_variance"), py::arg("scale"))
        .def("train", &GPNode::train)
        .def("gradients", &GPNode::gradients)
        .def("log_likelihood", &GPNode::log_likelihood)
        .def("log_prior", &GPNode::log_prior)
        .def("log_prior_gradient", &GPNode::log_prior_gradient)
        .def("objective_fxn", &GPNode::objective_fxn)
        .def("predict", &GPNode::predict)
        .def_readwrite("solver_settings", &GPNode::solver_settings)
        .def_readwrite("likelihood_variance", &GPNode::likelihood_variance)
        .def_readwrite("scale", &GPNode::scale)
        .def_readwrite("kernel", &GPNode::kernel)
        .def_property("inputs", &GPNode::get_inputs, &GPNode::set_inputs)
        .def_property("outputs", &GPNode::get_outputs, &GPNode::set_outputs);

    py::class_<Layer> mLayer(module, "CGPLayer");
    mLayer
        .def(py::init<std::vector<GPNode>&>(), py::arg("nodes"))
        .def("propagate", &Layer::operator())
        .def("train", &Layer::train)
        .def("predict", &Layer::predict)
        .def("set_inputs", &Layer::set_inputs)
        .def("set_outputs_", &Layer::set_outputs)
        .def("observed_inputs", &Layer::get_inputs)
        .def("observed_outputs", &Layer::get_outputs)
        .def("reconstruct_observed", &Layer::reconstruct_observed, py::arg("inputs"), py::arg("outputs"))
        .def_readwrite("index", &Layer::index)
        .def_readwrite("nodes", &Layer::nodes);
        //.def_property_readonly("state", &Layer::state)
        //.def_property_readonly("latent_output", &Layer::latent_output_);

    py::class_<SIDGP> mDGP(module, "CSIDGP");
    mDGP
        .def(py::init<std::vector<Layer>&, bool&>(), py::arg("layers"), py::arg("initialize") = true)
        .def("train", &SIDGP::train, py::arg("n_iter"), py::arg("ess_burn"))
        .def("estimate", &SIDGP::estimate, py::arg("n_burn"))
        .def("predict", &SIDGP::predict, py::arg("X"), py::arg("n_impute"), py::arg("n_thread"))
        .def_property("layers", &SIDGP::get_layers, &SIDGP::set_layers)
        .def_property_readonly("train_iter", &SIDGP::n_iterations);

}


PYBIND11_MODULE(_deep_models, module) {
	wrap_deep(module);
}
