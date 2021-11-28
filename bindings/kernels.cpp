#include "kernels.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

namespace py = pybind11;
using namespace fdml::kernels;
void wrap_kernels(py::module& module) {

    /*
    //class PyStationary : public Stationary {
    //    using Stationary::Stationary;
    //    const TMatrix diag(const TMatrix& X1) override {
    //        PYBIND11_OVERRIDE(TMatrix, Stationary, diag, X1);
    //    }
    //    void get_bounds(TVector& lower, TVector& upper, bool transformed) override {
    //        PYBIND11_OVERRIDE(void, Stationary, get_bounds, lower, upper, transformed);
    //    }
    //    void set_params(const TVector& params) override {
    //        PYBIND11_OVERRIDE(void, Stationary, set_params, params);
    //    }
    //    TVector get_params() override {
    //        PYBIND11_OVERRIDE(TVector, Stationary, get_params,);
    //    }
    //    TVector gradients(const TMatrix& X) override {
    //        PYBIND11_OVERRIDE(TVector, Stationary, gradients, X);
    //    }
    //    TVector gradients(const TMatrix& X, const TMatrix& dNLL) override {
    //        PYBIND11_OVERRIDE_PURE(TVector, Stationary, gradients, X, dNLL);
    //    }
    //    TVector gradients(const TMatrix& X, const TMatrix& dNLL, const TMatrix& R, const TMatrix& K) override {
    //        PYBIND11_OVERRIDE_PURE(TVector, Stationary, gradients, X, dNLL, R, K);
    //    }
    //};
    */

    class PyKernel : public Kernel {

    public:
        using Kernel::Kernel;
        void dK_dlengthscale(std::vector<TMatrix>& dK, std::vector<double>& grad, const TMatrix& tmp) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, dK_dlengthscale, dK, grad, tmp);
        }
        void dK_dvariance(const TMatrix& K, const TMatrix& dNLL, std::vector<double>& grad) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, dK_dvariance, K, dNLL, grad);
        }
        const TMatrix K(const TMatrix& X1, const TMatrix& X2) override {
            PYBIND11_OVERRIDE(TMatrix, Kernel, K, X1, X2);
        }
        const TMatrix K(const TMatrix& X1, const TMatrix& X2, TMatrix& R) override {
            PYBIND11_OVERRIDE(TMatrix, Kernel, K, X1, X2, R);
        }
        const TMatrix K(const TMatrix& X1, const TMatrix& X2, const double& likelihood_variance) override {
            PYBIND11_OVERRIDE(TMatrix, Kernel, K, X1, X2, likelihood_variance);
        }
        TMatrix diag(const TMatrix& X1) override {
            PYBIND11_OVERRIDE(TMatrix, Kernel, diag, X1);
        }
        void get_bounds(TVector& lower, TVector& upper, bool transformed) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, get_bounds, lower, upper, transformed);
        }

        void set_params(const TVector& params) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, set_params, params);
        }
        void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, IJ, I, J, mean, variance, X, idx);
        }

        void expectations(const TMatrix& mean, const TMatrix& variance) override {
            PYBIND11_OVERRIDE_PURE(void, Kernel, expectations, mean, variance);
        }
        TVector get_params() override {
            PYBIND11_OVERRIDE(TVector, Kernel, get_params, );
        }
        void gradients(const TMatrix& X, const TMatrix& dNLL, const TMatrix& R, const TMatrix& K, std::vector<double>& grad) override
        {
            PYBIND11_OVERRIDE_PURE(void, Kernel, gradients, X, dNLL, R, K, grad);
        }
    };

    class PyStationary : public Stationary
    {
    public:
        using Stationary::Stationary;
        void gradients(const TMatrix& X, const TMatrix& dNLL, const TMatrix& R, const TMatrix& K, std::vector<double>& grad) override
        {
            PYBIND11_OVERRIDE_PURE(void, Stationary, gradients, X, dNLL, R, K, grad);
        }
        void IJ(TMatrix& I, TMatrix& J, const TVector& mean, const TVector& variance, const TMatrix& X, const Eigen::Index& idx) override {
            PYBIND11_OVERRIDE_PURE(void, Stationary, IJ, I, J, mean, variance, X, idx);
        }

        void expectations(const TMatrix& mean, const TMatrix& variance) override {
            PYBIND11_OVERRIDE_PURE(void, Stationary, expectations, mean, variance);
        }
    };


    py::class_<Kernel, PyKernel, shared_ptr<Kernel>> KERNEL(module, "_Kernel", py::is_final());
    KERNEL
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<const Parameter<TVector>&, const Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Kernel::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Kernel::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Kernel::length_scale)
        .def_readwrite("variance", &Kernel::variance);


    py::class_<Stationary, Kernel, PyStationary, shared_ptr<Stationary>> STATIONARY(module, "_Stationary", py::is_final());
    STATIONARY
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<const Parameter<TVector>&, const Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Stationary::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Stationary::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Stationary::length_scale)
        .def_readwrite("variance", &Stationary::variance);
    

    py::class_<SquaredExponential, Stationary, Kernel, shared_ptr<SquaredExponential>> SE(module, "SquaredExponential", py::multiple_inheritance());
    SE
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<const Parameter<TVector>&, const Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&SquaredExponential::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &SquaredExponential::diag, py::arg("X1"))
        .def("IJ", &SquaredExponential::IJ, py::arg("II"), py::arg("JJ"), py::arg("mean"), py::arg("variance"), py::arg("X"), py::arg("idx"))
        .def("expectations", &SquaredExponential::expectations, py::arg("mean"), py::arg("variance"))
        .def_readwrite("length_scale", &SquaredExponential::length_scale)
        .def_readwrite("variance", &SquaredExponential::variance)
        .def(py::pickle(
            [/*__getstate__*/](const SquaredExponential& p) {return py::make_tuple(p.length_scale, p.variance); },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid SquaredExponential state!");
                }
                /* Create a new C++ instance */
                SquaredExponential p = SquaredExponential(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
                return p;
            })
        );

    py::class_<Matern32, Stationary, Kernel, shared_ptr<Matern32>> M32(module, "Matern32", py::multiple_inheritance());
    M32
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<const Parameter<TVector>&, const Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Matern32::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Matern32::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Matern32::length_scale)
        .def_readwrite("variance", &Matern32::variance)
        .def(py::pickle(
            [/*__getstate__*/](const Matern32& p) {return py::make_tuple(p.length_scale, p.variance); },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid Matern32 state!");
                }
                /* Create a new C++ instance */
                Matern32 p = Matern32(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
                return p;
            })
        );

    py::class_<Matern52, Stationary, Kernel, shared_ptr<Matern52>> M52(module, "Matern52", py::multiple_inheritance());
    M52
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<const Parameter<TVector>&, const Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Matern52::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Matern52::diag, py::arg("X1"))
        .def("IJ", &Matern52::IJ, py::arg("II"), py::arg("JJ"), py::arg("mean"), py::arg("variance"), py::arg("X"), py::arg("idx"))
        .def_readwrite("length_scale", &Matern52::length_scale)
        .def_readwrite("variance", &Matern52::variance)
        .def(py::pickle(
            [/*__getstate__*/](const Matern52& p) {return py::make_tuple(p.length_scale, p.variance); },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid Matern52 state!");
                }
                /* Create a new C++ instance */
                Matern52 p = Matern52(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
                return p;
            })
        );

}



PYBIND11_MODULE(_kernels, module) {
    wrap_kernels(module);
}
