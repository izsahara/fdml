#include "base_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

namespace py = pybind11;
void wrap_base_models(py::module& module) {
    using namespace fdml::base_models;
    using namespace fdml::base_models::gaussian_process;
    using namespace fdml::base_models::optimizer;

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
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const double&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
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
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const double&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const Parameter<double>&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def("train", &GPR::train)
        .def("gradients", &GPR::gradients)
        .def("log_likelihood", &GPR::log_likelihood)
        .def("objective_fxn", &GPR::objective_fxn)
        .def("predict", &GPR::predict)
        .def_readwrite("solver_settings", &GPR::solver_settings)
        .def_readwrite("likelihood_variance", &GPR::likelihood_variance)
        .def_readwrite("kernel", &GPR::kernel)
        .def_readwrite("inputs", &GPR::inputs)
        .def_readwrite("outputs", &GPR::outputs)
        .def_property_readonly("model_type", &GPR::model_type)
        .def(py::pickle(
            [/*__getstate__*/](const GPR& p) {
                return py::make_tuple(p.kernel, p.inputs, p.outputs, p.likelihood_variance, p.missing, p.solver_settings, p.objective_value);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 7)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                GPR p = GPR(t[0].cast<shared_ptr<Kernel>>(), t[1].cast<TMatrix>(), t[2].cast<TMatrix>(), t[3].cast<Parameter<double>>());
                p.missing = t[4].cast<BoolVector>();
                p.solver_settings = t[5].cast<SolverSettings>();
                p.objective_value = t[6].cast<double>();
                return p;
            })
        );

    py::class_<SolverSettings> mSETTINGS(module, "SolverSettings");
    mSETTINGS
        .def_readwrite("verbosity", &SolverSettings::verbosity)
        .def_readwrite("n_restarts", &SolverSettings::n_restarts)
        .def_readwrite("solver_iterations", &SolverSettings::solver_iterations)
        .def_readwrite("gradient_norm", &SolverSettings::gradient_norm)
        .def_readwrite("x_delta", &SolverSettings::x_delta)
        .def_readwrite("f_delta", &SolverSettings::f_delta)
        .def_readwrite("x_delta_violations", &SolverSettings::x_delta_violations)
        .def_readwrite("f_delta_violations", &SolverSettings::f_delta_violations)
        .def(py::pickle(
            [/*__getstate__*/](const SolverSettings& p) {
                return py::make_tuple(p.verbosity, p.n_restarts, p.solver_iterations, p.gradient_norm, 
                                      p.x_delta, p.f_delta,
                                      p.x_delta_violations, p.f_delta_violations);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 8)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                SolverSettings p = SolverSettings();
                p.verbosity = t[0].cast<int>();
                p.n_restarts = t[1].cast<int>();
                p.solver_iterations = t[2].cast<int>();
                p.gradient_norm = t[3].cast<double>();
                p.x_delta = t[4].cast<double>();
                p.f_delta = t[5].cast<double>();
                p.x_delta_violations = t[6].cast<int>();
                p.f_delta_violations = t[7].cast<int>();

                return p;
            })
        );
}

PYBIND11_MODULE(_base_models, module) {
	wrap_base_models(module);
}
