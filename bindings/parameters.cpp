#include "parameters.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"
#include <pybind11/stl.h>
#include "pybind11/eigen.h"

namespace py = pybind11;
using namespace fdsm::parameters;
void wrap_parameters(py::module& module) {

    module.def("transform_fxn", &transform_fxn<TVector>,
        py::arg("value"), py::arg("method"), py::arg("inverse")
    );
    module.def("transform_fxn", &transform_fxn<double>,
        py::arg("value"), py::arg("method"), py::arg("inverse")
    );

    py::class_<Parameter<TVector>> PAR1(module, "VectorParameter");
    PAR1
        // Constructor Definitions
        .def(py::init<std::string, TVector>(), py::arg("name"), py::arg("value"))
        .def(py::init<std::string, TVector, std::string>(), py::arg("name"), py::arg("value"), py::arg("transform"))
        .def(py::init<std::string, TVector, std::pair<TVector, TVector>>(), py::arg("name"), py::arg("value"), py::arg("bounds"))
        .def(py::init<std::string, TVector, std::string, std::pair<TVector, TVector>>(), py::arg("name"), py::arg("value"), py::arg("transform"), py::arg("bounds"))
        // Method Definitions
        .def("fix", &Parameter<TVector>::fix)
        .def("unfix", &Parameter<TVector>::unfix)
        // Attributes Definitions
        .def_property("value", &Parameter<TVector>::get_value, &Parameter<TVector>::set_value)
        .def_property("name", &Parameter<TVector>::get_name, &Parameter<TVector>::set_name)
        .def_property("constraint", &Parameter<TVector>::get_transform, &Parameter<TVector>::set_transform)
        .def_property("bounds", &Parameter<TVector>::get_bounds, &Parameter<TVector>::set_bounds)
        .def_property_readonly("fixed", &Parameter<TVector>::fixed);
        // Make Class Pickable
        //.def("__getstate__", [](const Parameter<TVector>& p)
        //    {return py::make_tuple(p.get_name(), p.get_value(), p.get_transform(), p.get_bounds(), p.fixed()); })
        //.def("__setstate__", [](Parameter<TVector>& p, const py::tuple& t) {
        //    if (t.size() != 5){ throw std::runtime_error("Invalid state!"); }
        //    new (&p) Parameter<TVector>(t[0].cast<std::string>(), t[1].cast<TVector>(), t[2].cast<std::string>(), t[3].cast<std::pair<TVector, TVector>>());
        //    bool fixed = t[4].cast<bool>();
        //    if (fixed) { p.fix(); }
        //    });


    py::class_<Parameter<double>> PAR2(module, "FloatParameter");
    PAR2
        // Constructor Definitions
        .def(py::init<std::string, double>(), py::arg("name"), py::arg("value"))
        .def(py::init<std::string, double, std::string>(), py::arg("name"), py::arg("value"), py::arg("transform"))
        .def(py::init<std::string, double, std::pair<double, double>>(), py::arg("name"), py::arg("value"), py::arg("bounds"))
        .def(py::init<std::string, double, std::string, std::pair<double, double>>(), py::arg("name"), py::arg("value"), py::arg("transform"), py::arg("bounds"))
        // Method Definitions
        .def("fix", &Parameter<double>::fix)
        .def("unfix", &Parameter<double>::unfix)
        // Attributes Definitions
        .def_property("value", &Parameter<double>::get_value, &Parameter<double>::set_value)
        .def_property("name", &Parameter<double>::get_name, &Parameter<double>::set_name)
        .def_property("constraint", &Parameter<double>::get_transform, &Parameter<double>::set_transform)
        .def_property("bounds", &Parameter<double>::get_bounds, &Parameter<double>::set_bounds)
        .def_property_readonly("fixed", &Parameter<double>::fixed);
        // Make Class Pickable
        //.def("__getstate__", [](const Parameter<double>& p)
        //{return py::make_tuple(p.get_name(), p.get_value(), p.get_transform(), p.get_bounds(), p.fixed()); })
        //.def("__setstate__", [](Parameter<double>& p, const py::tuple& t) {
        //if (t.size() != 5) { throw std::runtime_error("Invalid state!"); }
        //new (&p) Parameter<double>(t[0].cast<std::string>(), t[1].cast<double>(), t[2].cast<std::string>(), t[3].cast<std::pair<double, double>>());
        //bool fixed = t[4].cast<bool>();
        //if (fixed) { p.fix(); }
        //});

}


PYBIND11_MODULE(_parameters, mod) {
    wrap_parameters(mod);
}
