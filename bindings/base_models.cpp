#include "base_models.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

namespace py = pybind11;
void wrap_base_models(py::module& module) {
    using namespace fdsm::base_models;
    using namespace fdsm::base_models::gaussian_process;
    using namespace fdsm::base_models::optimizer;

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


}


PYBIND11_MODULE(_base_models, module) {
	wrap_base_models(module);
}
