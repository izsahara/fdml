#include "base_models2.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace fdml::base_models2;
void wrap_optimizer(py::module& module) {
    using namespace optimizer;

    class PySolver : public Solver {

    public:
        using Solver::Solver;

        SolverSettings settings() const override {
            PYBIND11_OVERRIDE(SolverSettings, Solver, settings);
        }
        bool solve(TVector& theta,
            std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
            OptData optdata, SolverSettings& settings) const override {
            PYBIND11_OVERRIDE(bool, Solver, solve, theta, objective, optdata, settings);
        }
    };

    py::class_<Solver, PySolver, shared_ptr<Solver>> msol(module, "Solver");
    msol
        .def(py::init<>())
        .def(py::init<const int&, const int&>())
        .def(py::init<const int&, const int&, const std::string&>())
        .def_readwrite("verbosity", &Solver::verbosity)
        .def_readwrite("n_restarts", &Solver::n_restarts)
        .def_readwrite("sampling_method", &Solver::sampling_method)
        .def_readwrite("conv_failure_switch", &Solver::conv_failure_switch)
        .def_readwrite("iter_max", &Solver::iter_max)
        .def_readwrite("err_tol", &Solver::err_tol)
        .def_readwrite("vals_bound", &Solver::vals_bound);

    py::class_<PSO, Solver, shared_ptr<PSO>> mpso(module, "PSO");
    mpso
        .def(py::init<>())
        .def(py::init<const int&, const int&>(), py::arg("verbosity"), py::arg("n_restarts"))
        .def(py::init<const int&, const int&, const std::string&>(), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"))
        .def(py::init<const int&, const int&, const std::string&, const TVector&, const TVector&>(), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"), py::arg("initial_lb"), py::arg("initial_ub"))
        .def_readwrite("center_particle", &PSO::center_particle)
        .def_readwrite("n_pop", &PSO::n_pop)
        .def_readwrite("n_gen", &PSO::n_gen)
        .def_readwrite("inertia_method", &PSO::inertia_method)
        .def_readwrite("initial_w", &PSO::par_initial_w)
        .def_readwrite("w_damp", &PSO::par_w_damp)
        .def_readwrite("w_min", &PSO::par_w_min)
        .def_readwrite("w_max", &PSO::par_w_max)
        .def_readwrite("velocity_method", &PSO::velocity_method)
        .def_readwrite("c_cog", &PSO::par_c_cog)
        .def_readwrite("c_soc", &PSO::par_c_soc)
        .def_readwrite("initial_c_cog", &PSO::par_initial_c_cog)
        .def_readwrite("final_c_cog", &PSO::par_final_c_cog)
        .def_readwrite("initial_c_soc", &PSO::par_initial_c_soc)
        .def_readwrite("final_c_soc", &PSO::par_final_c_soc)
        .def(py::pickle(
            [/*__getstate__*/](const PSO& p) {
                return py::make_tuple
                (p.verbosity, p.n_restarts, p.sampling_method, p.initial_lb, p.initial_ub,
                p.conv_failure_switch,
                p.iter_max,
                p.err_tol,
                p.vals_bound,
                p.center_particle,
                p.n_pop,
                p.n_gen,
                p.inertia_method,
                p.par_initial_w,
                p.par_w_damp,
                p.par_w_min,
                p.par_w_max,
                p.velocity_method,
                p.par_c_cog,
                p.par_c_soc,
                p.par_initial_c_cog,
                p.par_final_c_cog,
                p.par_initial_c_soc,
                p.par_final_c_soc);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 24)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                PSO p = PSO(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<std::string>());
                p.initial_lb            = t[3].cast<TVector>();
                p.initial_ub            = t[4].cast<TVector>();
                p.conv_failure_switch   = t[5].cast<int>();
                p.iter_max              = t[6].cast<int>();
                p.err_tol               = t[7].cast<double>();
                p.vals_bound            = t[8].cast<bool>();
                p.center_particle       = t[9].cast<bool>();
                p.n_pop                 = t[10].cast<int>();
                p.n_gen                 = t[11].cast<int>();
                p.inertia_method        = t[12].cast<int>();
                p.par_initial_w         = t[13].cast<double>();
                p.par_w_damp            = t[14].cast<double>();
                p.par_w_min             = t[15].cast<double>();
                p.par_w_max             = t[16].cast<double>();
                p.velocity_method       = t[17].cast<int>();
                p.par_c_cog             = t[18].cast<double>();
                p.par_c_soc             = t[19].cast<double>();
                p.par_initial_c_cog     = t[20].cast<double>();
                p.par_final_c_cog       = t[21].cast<double>();
                p.par_initial_c_soc     = t[22].cast<double>();
                p.par_final_c_soc       = t[23].cast<double>();

                return p;
            })
        );

    py::class_<DifferentialEvolution, Solver, shared_ptr<DifferentialEvolution>> mde(module, "DifferentialEvolution");
    mde
        .def(py::init<>())
        .def(py::init<const int&, const int&>(), py::arg("verbosity"), py::arg("n_restarts"))
        .def(py::init<const int&, const int&, const std::string&>(), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"))
        .def(py::init<const int&, const int&, const std::string&, const TVector&, const TVector&>(), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"), py::arg("initial_lb"), py::arg("initial_ub"))
        .def_readwrite("n_pop", &DifferentialEvolution::n_pop)
        .def_readwrite("n_pop_best", &DifferentialEvolution::n_pop_best)
        .def_readwrite("n_gen", &DifferentialEvolution::n_gen)
        .def_readwrite("pmax", &DifferentialEvolution::pmax)
        .def_readwrite("max_fn_eval", &DifferentialEvolution::max_fn_eval)
        .def_readwrite("mutation_method", &DifferentialEvolution::mutation_method)
        .def_readwrite("check_freq", &DifferentialEvolution::check_freq)
        .def_readwrite("F", &DifferentialEvolution::par_F)
        .def_readwrite("CR", &DifferentialEvolution::par_CR)
        .def_readwrite("F_l", &DifferentialEvolution::par_F_l)
        .def_readwrite("F_u", &DifferentialEvolution::par_F_u)
        .def_readwrite("tau_F", &DifferentialEvolution::par_tau_F)
        .def_readwrite("tau_CR", &DifferentialEvolution::par_tau_CR)
        .def(py::pickle(
            [/*__getstate__*/](const DifferentialEvolution& p) {
                return py::make_tuple
                (p.verbosity, p.n_restarts, p.sampling_method, p.initial_lb, p.initial_ub,
                p.conv_failure_switch,
                p.iter_max,
                p.err_tol,
                p.vals_bound,
                p.n_pop, 
                p.n_pop_best, 
                p.n_gen, 
                p.pmax, 
                p.max_fn_eval, 
                p.mutation_method, 
                p.check_freq, 
                p.par_F, 
                p.par_CR, 
                p.par_F_l, 
                p.par_F_u, 
                p.par_tau_F, 
                p.par_tau_CR);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 22)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                DifferentialEvolution p = DifferentialEvolution(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<std::string>());
                p.initial_lb            = t[3].cast<TVector>();
                p.initial_ub            = t[4].cast<TVector>();
                p.conv_failure_switch   = t[5].cast<int>();
                p.iter_max              = t[6].cast<int>();
                p.err_tol               = t[7].cast<double>();
                p.vals_bound            = t[8].cast<bool>();
                p.n_pop                 = t[9].cast<int>();
                p.n_pop_best            = t[10].cast<int>();
                p.n_gen                 = t[11].cast<int>();
                p.pmax                  = t[12].cast<int>();
                p.max_fn_eval           = t[13].cast<int>();
                p.mutation_method       = t[14].cast<int>();
                p.check_freq            = t[15].cast<int>();
                p.par_F                 = t[16].cast<double>();
                p.par_CR                = t[17].cast<double>();
                p.par_F_l               = t[18].cast<double>();
                p.par_F_u               = t[19].cast<double>();
                p.par_tau_F             = t[20].cast<double>();
                p.par_tau_CR            = t[21].cast<double>();
                return p;
            })
        );

    py::class_<LBFGSB, Solver, shared_ptr<LBFGSB>> mlbfgs(module, "LBFGSB");
    mlbfgs
        .def(py::init<>())
        .def(py::init<const int&, const int&>(), py::arg("verbosity"), py::arg("n_restarts"))
        .def(py::init<const int&, const int&, std::string&>(), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"))
        .def_readwrite("solver_iterations", &LBFGSB::solver_iterations)
        .def_readwrite("gradient_norm", &LBFGSB::gradient_norm)
        .def_readwrite("x_delta", &LBFGSB::x_delta)
        .def_readwrite("f_delta", &LBFGSB::f_delta)
        .def_readwrite("x_delta_violations", &LBFGSB::x_delta_violations)
        .def_readwrite("f_delta_violations", &LBFGSB::f_delta_violations)
        .def(py::pickle(
            [/*__getstate__*/](const LBFGSB& p) {
                return py::make_tuple(p.verbosity, p.n_restarts, p.sampling_method,
                    p.conv_failure_switch,
                    p.iter_max,
                    p.err_tol,
                    p.vals_bound,
                    p.solver_iterations,
                    p.gradient_norm,
                    p.x_delta,
                    p.f_delta,
                    p.x_delta_violations,
                    p.f_delta_violations
                );
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 8)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                LBFGSB p = LBFGSB(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<std::string>());
                p.conv_failure_switch   = t[3].cast<int>();
                p.iter_max              = t[4].cast<int>();
                p.err_tol               = t[5].cast<double>();
                p.vals_bound            = t[6].cast<bool>();
                p.solver_iterations     = t[7].cast<int>();
                p.gradient_norm         = t[8].cast<double>();
                p.x_delta               = t[9].cast<double>();
                p.f_delta               = t[10].cast<double>();
                p.x_delta_violations    = t[11].cast<int>();
                p.f_delta_violations    = t[12].cast<int>();
                return p;
            })
        );

    py::class_<GradientDescent, Solver, shared_ptr<GradientDescent>> mgd(module, "GradientDescent");
    mgd
        .def(py::init<const int&>(), py::arg("method"))
        .def(py::init<const int&, const int&, const int&>(), py::arg("method"), py::arg("verbosity"), py::arg("n_restarts"))
        .def(py::init<const int& , const int&, const int&, const std::string&>(), py::arg("method"), py::arg("verbosity"), py::arg("n_restarts"), py::arg("sampling_method"))
        .def_readwrite("method", &GradientDescent::method)
        .def_readwrite("step_size", &GradientDescent::step_size)
        .def_readwrite("step_decay", &GradientDescent::step_decay)
        .def_readwrite("step_decay_periods", &GradientDescent::step_decay_periods)
        .def_readwrite("step_decay_val", &GradientDescent::step_decay_val)
        .def_readwrite("momentum", &GradientDescent::momentum)
        .def_readwrite("ada_norm", &GradientDescent::ada_norm)
        .def_readwrite("ada_rho", &GradientDescent::ada_rho)
        .def_readwrite("ada_max", &GradientDescent::ada_max)
        .def_readwrite("adam_beta_1", &GradientDescent::adam_beta_1)
        .def_readwrite("adam_beta_2", &GradientDescent::adam_beta_2)
        .def(py::pickle(
            [/*__getstate__*/](const GradientDescent& p) {
                return py::make_tuple(p.method, p.verbosity, p.n_restarts, p.sampling_method,
                    p.conv_failure_switch,
                    p.iter_max,
                    p.err_tol,
                    p.vals_bound,
                    p.method,
                    p.step_size,
                    p.step_decay,
                    p.step_decay_periods,
                    p.step_decay_val,
                    p.momentum,
                    p.ada_norm,
                    p.ada_rho,
                    p.ada_max,
                    p.adam_beta_1,
                    p.adam_beta_2
                    );
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 18)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                GradientDescent p = GradientDescent(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<std::string>());
                p.conv_failure_switch = t[4].cast<int>();
                p.iter_max            = t[5].cast<int>();
                p.err_tol             = t[6].cast<double>();
                p.vals_bound          = t[7].cast<bool>();
                p.step_size           = t[8].cast<double>();
                p.step_decay          = t[9].cast<bool>();
                p.step_decay_periods  = t[10].cast<unsigned int>();
                p.step_decay_val      = t[11].cast<double>();
                p.momentum            = t[12].cast<double>();
                p.ada_norm            = t[13].cast<double>();
                p.ada_rho             = t[14].cast<double>();
                p.ada_max             = t[15].cast<bool>();
                p.adam_beta_1         = t[16].cast<double>();
                p.adam_beta_2         = t[17].cast<double>();
                return p;
            })
        );



}

void wrap_models(py::module& module) {
    using namespace models;
    class PyGP : public GP {

    public:
        using GP::GP;

        void train() override {
            PYBIND11_OVERRIDE_PURE(void, GP, train);
        }
        double log_marginal_likelihood() override {
            PYBIND11_OVERRIDE(double, GP, log_marginal_likelihood);
        }
        TVector gradients() override {
            PYBIND11_OVERRIDE(TVector, GP, gradients);
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
        .def(py::init<const TMatrix&, const TMatrix&>(), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, shared_ptr<Solver>>(), py::arg("kernel"), py::arg("solver"))
        .def(py::init<shared_ptr<Solver>, const TMatrix&, const TMatrix&>(), py::arg("solver"), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const double&>(), py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Solver>, const TMatrix&, const TMatrix&, const double&>(), py::arg("solver"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Kernel>, shared_ptr<Solver>, const TMatrix&, const TMatrix&>(), py::arg("kernel"), py::arg("solver"), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, shared_ptr<Solver>, const TMatrix&, const TMatrix&, const double&>(), py::arg("kernel"), py::arg("solver"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def(py::init<shared_ptr<Kernel>, shared_ptr<Solver>, const TMatrix&, const TMatrix&, const Parameter<double>&>(), py::arg("kernel"), py::arg("solver"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"))
        .def("train", &GP::train)
        .def("gradients", &GP::gradients)
        .def("log_marginal_likelihood", &GP::log_marginal_likelihood)
        .def_readwrite("likelihood_variance", &GP::likelihood_variance)
        .def_readwrite("kernel", &GP::kernel)
        .def_readwrite("solver", &GP::solver)
        .def_readwrite("inputs", &GP::inputs)
        .def_readwrite("outputs", &GP::outputs);
}

void wrap_gaussian_process(py::module& module) {
    using namespace gaussian_process;

    py::class_<GPR, GP> mGPR(module, "GPR");
    mGPR
        .def(py::init<const TMatrix&, const TMatrix&>(), py::arg("inputs"), py::arg("outputs"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const double&, const double&>(),
            py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"), py::arg("scale"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const double&, const double&, shared_ptr<Solver>>(),
            py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"), py::arg("scale"), py::arg("solver"))
        .def(py::init<shared_ptr<Kernel>, const TMatrix&, const TMatrix&, const Parameter<double>&, const Parameter<double>&, shared_ptr<Solver>>(),
            py::arg("kernel"), py::arg("inputs"), py::arg("outputs"), py::arg("likelihood_variance"), py::arg("scale"), py::arg("solver"))
        .def("train", &GPR::train)
        .def("gradients", &GPR::gradients)
        .def("log_marginal_likelihood", &GPR::log_marginal_likelihood)
        .def("log_prior", &GPR::log_prior)
        .def("log_prior_gradient", &GPR::log_prior_gradient)
        .def("predict", &GPR::predict)
        .def_readwrite("likelihood_variance", &GPR::likelihood_variance)
        .def_readwrite("scale", &GPR::scale)
        .def_readwrite("kernel", &GPR::kernel)
        .def_readwrite("inputs", &GPR::inputs)
        .def_readwrite("outputs", &GPR::outputs)
        .def_readwrite("solver", &GPR::solver)
        .def_readwrite("objective_value", &GPR::objective_value)
        .def_property_readonly("model_type", &GPR::model_type)
        .def(py::pickle(
            [/*__getstate__*/](const GPR& p) {
                return py::make_tuple(p.kernel, p.inputs, p.outputs, p.likelihood_variance, p.scale, p.solver, p.missing, p.objective_value);
            },
            [/*__setstate__*/](py::tuple t) {
                if (t.size() != 8)
                {
                    throw std::runtime_error("Invalid state!");
                }
                /* Create a new C++ instance */
                GPR p = GPR(t[0].cast<shared_ptr<Kernel>>(), 
                            t[1].cast<TMatrix>(), 
                            t[2].cast<TMatrix>(),
                            t[3].cast<Parameter<double>>(), 
                            t[4].cast<Parameter<double>>(),
                            t[5].cast<shared_ptr<Solver>>());

                p.missing = t[6].cast<BoolVector>();
                p.objective_value = t[7].cast<double>();
                return p;
            })
        );
}


PYBIND11_MODULE(_base_models2, module) {
	wrap_optimizer(module);
	wrap_models(module);
	wrap_gaussian_process(module);
}
