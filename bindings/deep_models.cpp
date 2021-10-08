#include "./deep_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"

namespace py = pybind11;
using namespace::fdsm::base_models::gaussian_process;

void wrap_deep_c2(py::module module) {
    //using namespace fdsm::deep_models::gaussian_process_c2;
   
    //py::class_<Node> mNode(module, "CGPNode");
    //mNode
    //    .def(py::init<shared_ptr<Kernel>, Parameter<double>>(), py::arg("kernel"), py::arg("likelihood_variance"))
    //    .def("get_parameter_history", &Node::get_parameter_history)
    //    .def("sample_mvn", &Node::sample_mvn)
    //    .def_property("inputs", &Node::get_inputs, &Node::set_inputs)
    //    .def_property("outputs", &Node::get_outputs, &Node::set_outputs)
    //    .def_property("missing", &Node::get_missing, &Node::set_missing)
    //    .def_property_readonly("parameters_stored", &Node::parameters_stored)
    //    .def_property_readonly("kernel", &Node::kernel_)
    //    .def_property_readonly("likelihood_variance", &Node::likelihood_variance_);

    //py::class_<Layer> mLayer(module, "CGPLayer");
    //mLayer
    //    .def(py::init<std::vector<Node>&>(), py::arg("nodes"))
    //    .def("propagate", &Layer::operator())
    //    .def("log_likelihood", &Layer::log_likelihood)
    //    .def("estimate_parameters", &Layer::estimate_parameters, py::arg("n_burn"))
    //    .def("train", &Layer::train)
    //    .def("predict", &Layer::predict)
    //    .def("set_inputs", &Layer::set_inputs)
    //    .def("set_outputs_", &Layer::set_outputs)
    //    .def_readwrite("index", &Layer::index)
    //    .def_readwrite("nodes", &Layer::nodes)
    //    .def_property_readonly("state", &Layer::state)
    //    .def_property_readonly("latent_output", &Layer::latent_output_);

    //py::class_<SIDGP> mDGP(module, "CSIDGP");
    //mDGP
    //    .def(py::init<std::vector<Layer>&>(), py::arg("layers"))
    //    .def("train", &SIDGP::train, py::arg("n_iter"), py::arg("ess_burn"))
    //    .def("estimate", &SIDGP::estimate, py::arg("n_burn"))
    //    .def("predict", &SIDGP::predict, py::arg("X"), py::arg("n_impute"), py::arg("n_thread"))
    //    .def_property_readonly("train_iter", &SIDGP::train_iter_)
    //    .def_property_readonly("layers", &SIDGP::layers_);

}

void wrap_deep_c3(py::module module) {
    //using namespace fdsm::deep_models::gaussian_process;

    //py::class_<Layer> mLayer(module, "CGPLayer");
    //mLayer
    //    .def(py::init<std::vector<GPNode>&>(), py::arg("nodes"))
    //    .def("propagate", &Layer::operator())
    //    .def("log_likelihood", &Layer::log_likelihood)
    //    .def("estimate_parameters", &Layer::estimate_parameters, py::arg("n_burn"))
    //    .def("train", &Layer::train)
    //    .def("predict", &Layer::predict)
    //    .def("set_inputs", &Layer::set_inputs)
    //    .def("set_outputs_", &Layer::set_outputs)
    //    .def_readwrite("index", &Layer::index)
    //    .def_readwrite("nodes", &Layer::nodes)
    //    .def_property_readonly("state", &Layer::state)
    //    .def_property_readonly("latent_output", &Layer::latent_output_);

    //py::class_<SIDGP> mDGP(module, "CSIDGP");
    //mDGP
    //    .def(py::init<std::vector<Layer>&>(), py::arg("layers"))
    //    .def("train", &SIDGP::train, py::arg("n_iter"), py::arg("ess_burn"))
    //    .def("estimate", &SIDGP::estimate, py::arg("n_burn"))
    //    .def("predict", &SIDGP::predict, py::arg("X"), py::arg("n_impute"), py::arg("n_thread"))
    //    .def_property_readonly("train_iter", &SIDGP::train_iter_)
    //    .def_property_readonly("layers", &SIDGP::layers_);

}

void wrap_deep(py::module module) {
    using namespace fdsm::deep_models::gaussian_process;

    //module.def("update_f", &update_f,
    //            py::arg("f"), py::arg("nu"), py::arg("mean"), py::arg("theta"));
    //module.def("sample_mvn", &sample_mvn_, py::arg("K"));
    //module.def("log_likelihood", &log_likelihood, py::arg("K"), py::arg("outputs"));
    //module.def("one_sample", &one_sample, py::arg("node"), py::arg("linked_layer"), py::arg("node_index"));

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
        .def_readwrite("nodes", &Layer::nodes)
        .def_property_readonly("state", &Layer::state)
        .def_property_readonly("latent_output", &Layer::latent_output_);

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
