#include "kernels.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

namespace py = pybind11;
using namespace fdsm::kernels;
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


    py::class_<Kernel, shared_ptr<Kernel>, PyKernel> KERNEL(module, "_Kernel", py::is_final());
    KERNEL
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<Parameter<TVector>&, Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Kernel::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Kernel::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Kernel::length_scale)
        .def_readwrite("variance", &Kernel::variance);
        //.def(py::pickle(
        //    [](const py::object& self) {
        //        py::dict d;
        //        if (py::hasattr(self, "__dict__"))
        //            d = self.attr("__dict__");
        //        return py::make_tuple(self.attr("length_scale"), self.attr("variance"), d);
        //    },
        //    [](const py::tuple& t) {
        //        if (t.size() != 3)
        //            throw std::runtime_error("Invalid state!");
        //        auto cpp_state = std::unique_ptr<Kernel>(new PyKernel);
        //        cpp_state->variance = t[0].cast<Parameter<double>>();
        //        cpp_state->length_scale = t[1].cast<Parameter<TVector>>();
        //        auto py_state = t[2].cast<py::dict>();
        //        return std::make_pair(std::move(cpp_state), py_state);
        //    }
        //    ));


    py::class_<Stationary, shared_ptr<Stationary>, Kernel, PyStationary> STATIONARY(module, "_Stationary", py::is_final());
    STATIONARY
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<Parameter<TVector>&, Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Stationary::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Stationary::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Stationary::length_scale)
        .def_readwrite("variance", &Stationary::variance);
        //.def(py::pickle(
        //    [](const py::object& self) {
        //        py::dict d;
        //        if (py::hasattr(self, "__dict__"))
        //            d = self.attr("__dict__");
        //        return py::make_tuple(self.attr("length_scale"), self.attr("variance"), d);
        //    },
        //    [](const py::tuple& t) {
        //        if (t.size() != 3)
        //            throw std::runtime_error("Invalid state!");
        //        auto cpp_state = std::unique_ptr<Stationary>(new PyStationary);
        //        cpp_state->variance = t[0].cast<Parameter<double>>();
        //        cpp_state->length_scale = t[1].cast<Parameter<TVector>>();
        //        auto py_state = t[2].cast<py::dict>();
        //        return std::make_pair(std::move(cpp_state), py_state);
        //    }
        //    ));
    

    py::class_<SquaredExponential, shared_ptr<SquaredExponential>, Stationary, Kernel> SE(module, "SquaredExponential");
    SE
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<Parameter<TVector>&, Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&SquaredExponential::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &SquaredExponential::diag, py::arg("X1"))
        .def("IJ", &SquaredExponential::IJ, py::arg("II"), py::arg("JJ"), py::arg("mean"), py::arg("variance"), py::arg("X"), py::arg("idx"))
        .def("expectations", &SquaredExponential::expectations, py::arg("mean"), py::arg("variance"))
        .def_readwrite("length_scale", &SquaredExponential::length_scale)
        .def_readwrite("variance", &SquaredExponential::variance);
        // Make Class Pickable
        //.def("__getstate__", [](const shared_ptr<Kernel>& k)
        //    {return py::make_tuple(k->get_lengthscale(), k->get_variance()); })
        //.def("__setstate__", [](SquaredExponential& k, const py::tuple& t) {
        //if (t.size() != 2) { throw std::runtime_error("Invalid state!"); }

        //auto t0 = t[0].cast<Parameter<TVector>>();       
        //std::cout << "Converted t0 to vector param" << std::endl;
        //auto t1 = t[1].cast<Parameter<double>>();
        //std::cout << "Converted t1 to float param" << std::endl;

        //new (&k) SquaredExponential(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());});

        //.def(py::pickle(
        //    [](const SquaredExponential& k) {
        //        return py::make_tuple(k.length_scale, k.variance);
        //    },
        //    [](py::tuple t) {
        //        if (t.size() != 2){throw std::runtime_error("Invalid state!");}
        //        return new SquaredExponential(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
        //    }));

    py::class_<Matern32, shared_ptr<Matern32>, Stationary, Kernel> M32(module, "Matern32");
    M32
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<Parameter<TVector>&, Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Matern32::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Matern32::diag, py::arg("X1"))
        .def_readwrite("length_scale", &Matern32::length_scale)
        .def_readwrite("variance", &Matern32::variance);
        //.def(py::pickle(
        //    [](const Matern32& k) {
        //        return py::make_tuple(k.length_scale, k.variance);
        //    },
        //    [](py::tuple t) {
        //        if (t.size() != 2) { throw std::runtime_error("Invalid state!"); }
        //        return new Matern32(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
        //    }));
        //.def("__getstate__", [](const py::object& self)
        //    {return py::make_tuple(self.attr("length_scale"), self.attr("variance"), self.attr("__dict__")); })
        //.def("__setstate__", [](const py::object& self, const py::tuple& t) {
        //if (t.size() != 3) { throw std::runtime_error("Invalid state!"); }
        //auto& k = self.cast<Matern32&>();
        //new (&k) Matern32(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());        
        //self.attr("__dict__") = t[2]; });

    py::class_<Matern52, shared_ptr<Matern52>, Stationary, Kernel> M52(module, "Matern52");
    M52
        .def(py::init<>())
        .def(py::init<const double&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<TVector&, const double&>(), py::arg("length_scale"), py::arg("variance"))
        .def(py::init<Parameter<TVector>&, Parameter<double>&>(), py::arg("length_scale"), py::arg("variance"))
        .def("K", py::overload_cast<const TMatrix&, const TMatrix&>(&Matern52::K), py::arg("X1"), py::arg("X2"))
        .def("diag", &Matern52::diag, py::arg("X1"))
        .def("IJ", &Matern52::IJ, py::arg("II"), py::arg("JJ"), py::arg("mean"), py::arg("variance"), py::arg("X"), py::arg("idx"))
        .def_readwrite("length_scale", &Matern52::length_scale)
        .def_readwrite("variance", &Matern52::variance);
        //.def(py::pickle(
        //    [](const Matern52& k) {
        //        return py::make_tuple(k.length_scale, k.variance);
        //    },
        //    [](py::tuple t) {
        //        if (t.size() != 2) { throw std::runtime_error("Invalid state!"); }
        //        return new Matern52(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
        //    }));
        //.def("__getstate__", [](const py::object& self)
        //    {return py::make_tuple(self.attr("length_scale"), self.attr("variance"), self.attr("__dict__")); })
        //.def("__setstate__", [](const py::object& self, const py::tuple& t) {
        //if (t.size() != 3) { throw std::runtime_error("Invalid state!"); }
        //auto& k = self.cast<Kernel&>();
        //new (&k) Matern52(t[0].cast<Parameter<TVector>>(), t[1].cast<Parameter<double>>());
        //self.attr("__dict__") = t[2]; });

}



PYBIND11_MODULE(_kernels, module) {
    wrap_kernels(module);
}
