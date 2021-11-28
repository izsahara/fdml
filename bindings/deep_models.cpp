#include "./deep_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "pybind11/operators.h"

namespace py = pybind11;
void wrap_deep(py::module module) {
    using namespace fdml::base_models;
    using namespace fdml::deep_models::gaussian_process;

    py::class_<GPNode> mNode(module, "CGPNode");
    mNode
        .def(py::init<shared_ptr<Kernel>>(), py::arg("kernel"))
        .def(py::init<shared_ptr<Kernel>, const double&, const double&>(), py::arg("kernel"), py::arg("likelihood_variance"), py::arg("scale"))
        .def(py::init<shared_ptr<Kernel>, const Parameter<double>&, const Parameter<double>&>(), py::arg("kernel"), py::arg("likelihood_variance"), py::arg("scale"))
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
        .def_property("outputs", &GPNode::get_outputs, &GPNode::set_outputs)
        .def(py::pickle(
            [/*__getstate__*/](const GPNode& p) {
                return py::make_tuple(p.kernel, p.likelihood_variance, p.scale,
                    p.inputs, p.outputs,
                    p.missing, p.solver_settings,
                    p.objective_value, p.store_parameters, p.history, p.connected);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 11)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                GPNode p = GPNode(t[0].cast<shared_ptr<Kernel>>(), t[1].cast<Parameter<double>>(), t[2].cast<Parameter<double>>());
                p.inputs = t[3].cast<TMatrix>();
                p.outputs = t[4].cast<TMatrix>();
                p.missing = t[5].cast<BoolVector>();
                p.solver_settings = t[6].cast<SolverSettings>();
                p.objective_value = t[7].cast<double>();
                p.store_parameters = t[8].cast<bool>();
                p.history = t[9].cast<std::vector<TVector>>();
                p.connected = t[10].cast<bool>();
                return p;
            })
        );


    py::class_<Layer, shared_ptr<Layer>> mLayer(module, "CGPLayer");
    mLayer
        .def(py::init<const std::vector<GPNode>&, bool&>(), py::arg("nodes"), py::arg("initialize") = true)
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
        .def(py::pickle(
            [/*__getstate__*/](Layer& p) {
                return py::make_tuple(p.nodes, p.get_inputs(), p.get_outputs(),
                    p.last_layer, p.connected, p.n_thread, p.state, p.index); },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 8)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                Layer p = Layer(t[0].cast<std::vector<GPNode>>(), false);
                p.reconstruct_observed(t[1].cast<TMatrix>(), t[2].cast<TMatrix>());
                p.last_layer = t[3].cast<bool>();
                p.connected = t[4].cast<bool>();
                p.n_thread= t[5].cast<int>();
                p.state= t[6].cast<int>();
                p.index= t[7].cast<int>();
                return p;
            })
        );


    py::class_<SIDGP> mDGP(module, "CSIDGP");
    mDGP
        .def(py::init<const std::vector<Layer>&, bool&>(), py::arg("layers"), py::arg("initialize") = true)
        .def("train", &SIDGP::train, py::arg("n_iter"), py::arg("ess_burn"))
        .def("estimate", &SIDGP::estimate, py::arg("n_burn"))
        .def("predict", &SIDGP::predict, py::arg("X"), py::arg("n_impute"), py::arg("n_thread"))
        .def("set_observed", &SIDGP::set_observed, py::arg("inputs"), py::arg("outputs"))
        .def_readwrite("verbosity", &SIDGP::verbosity)
        .def_property_readonly("layers", &SIDGP::get_layers)
        .def_property_readonly("train_iter", &SIDGP::n_iterations)
        .def_property_readonly("model_type", &SIDGP::model_type)
        .def(py::pickle(
            [/*__getstate__*/](SIDGP& p) {return py::make_tuple(p.get_layers(), p.n_iterations(), p.verbosity); },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 3)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                SIDGP p = SIDGP(t[0].cast<std::vector<Layer>>(), false);
                p.set_n_iter(t[1].cast<int>());
                p.verbosity = t[2].cast<int>();
                return p;
            })
        );

}


PYBIND11_MODULE(_deep_models, module) {
	//wrap_deep_old(module);
	wrap_deep(module);
}
