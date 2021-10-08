#include "base_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace fdsm::base_models;
using namespace fdsm::base_models::gaussian_process;
using namespace fdsm::base_models::optimizer;
void wrap_base_models(py::module& module) {

    class PyGP : public GP {

    public:
        using GP::GP;

        void train() override {
            PYBIND11_OVERRIDE_PURE(void, GP, train);
        }

        TVector gradients() override {
            PYBIND11_OVERRIDE(TVector, GP, gradients);
        }

        double objective_fxn() override {
            PYBIND11_OVERRIDE(double, GP, objective_fxn);
        }

        void set_params(const TVector& new_params) override {
            PYBIND11_OVERRIDE_PURE(void, GP, set_params, new_params);
        }

        TVector get_params() override {
            PYBIND11_OVERRIDE(TVector, GP, get_params);
        }

    };


    py::class_<GP, PyGP> mBGP(module, "GP", py::is_final());
    mBGP
        .def(py::init<>())
        .def(py::init<shared_ptr<Kernel>>(), py::arg("kernel"))
        .def(py::init<const TMatrix&, const TMatrix&>(), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, double&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, Parameter<double>&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def("train", &GP::train)
        .def("gradients", &GP::gradients)
        .def("objective_fxn", &GP::objective_fxn)
        .def_readwrite("likelihood_variance", &GP::likelihood_variance)
        .def_readwrite("kernel", &GP::kernel)
        .def_readwrite("kernel", &GP::kernel)
        .def_readwrite("inputs", &GP::inputs)
        .def_readwrite("outputs", &GP::outputs);


    py::class_<GPR, GP> mGPR(module, "GPR");
    mGPR
        .def(py::init<>())
        .def(py::init<shared_ptr<Kernel>>(), py::arg("kernel"))
        .def(py::init<const TMatrix&, const TMatrix&>(), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, double&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, Parameter<double>&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def("train", &GPR::train)
        .def("gradients", &GPR::gradients)
        .def("log_likelihood", &GPR::log_likelihood)
        .def("objective_fxn", &GPR::objective_fxn)
        .def("predict", &GPR::predict)
        .def_readwrite("solver_settings", &GPR::solver_settings)
        .def_readwrite("likelihood_variance", &GPR::likelihood_variance)
        .def_readwrite("kernel", &GPR::kernel)
        .def_readwrite("inputs", &GPR::inputs)
        .def_readwrite("outputs", &GPR::outputs);

    py::class_<GPNode, GP> mNode(module, "GPNode");
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

    py::class_<SolverSettings> mSETTINGS(module, "SolverSettings");
    mSETTINGS
        .def_readwrite("verbosity", &SolverSettings::verbosity)
        .def_readwrite("n_restarts", &SolverSettings::n_restarts)
        .def_readwrite("solver_iterations", &SolverSettings::solver_iterations)
        .def_readwrite("gradient_norm", &SolverSettings::gradient_norm)
        .def_readwrite("x_delta", &SolverSettings::x_delta)
        .def_readwrite("f_delta", &SolverSettings::f_delta)
        .def_readwrite("x_delta_violations", &SolverSettings::x_delta_violations)
        .def_readwrite("f_delta_violations", &SolverSettings::f_delta_violations);

}



PYBIND11_MODULE(_base_models, module) {
	wrap_base_models(module);
}
