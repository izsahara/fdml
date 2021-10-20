#ifndef BASEMODELS2_H
#define BASEMODELS2_H
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <sstream>
#include <iomanip>
#include <cppoptlibmod/function.h>
#include <cppoptlibmod/lbfgsb.h>
#include <optim/optim.hpp>
#include "./kernels.h"

// Apply Fail-Safe Strategy for Optimization

namespace fdml::base_models2 {
	using namespace fdml::kernels;

	namespace optimizer {
		using SolverSettings = optim::algo_settings_t;
		struct OptData {};
		// TODO : 
		struct Solver {
			Solver(const std::string& type) : type(type) {}
			Solver(const int& verbosity, const int& n_restarts, const std::string& type) :
				verbosity(verbosity), n_restarts(n_restarts), type(type) {}
			Solver(const int& verbosity, const int& n_restarts, const std::string& sampling_method, const std::string& type) :
				verbosity(verbosity), n_restarts(n_restarts), sampling_method(sampling_method), type(type) {}

			virtual SolverSettings settings() const { SolverSettings settings_; return settings_; }
			virtual bool
			solve(TVector& theta,
				std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
				OptData optdata, SolverSettings& settings) const
			{
				return false;
			}

			// FDML
			int verbosity = 0;
			int n_restarts = 10;
			std::string sampling_method = "uniform";
			// Optim
			int conv_failure_switch = 0;
			int iter_max = 2000;
			double err_tol = 1E-08;
			bool vals_bound = true;
			const std::string type;
		};
		
		struct PSO : public Solver {
			PSO() : Solver("PSO") {}
			PSO(const int& verbosity, const int& n_restarts) :
				Solver(verbosity, n_restarts, "PSO") {}
			PSO(const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "PSO") {}
			PSO(const int& verbosity, const int& n_restarts, const std::string& sampling_method,
				const TVector& initial_lb, const TVector& initial_ub) :
				Solver(verbosity, n_restarts, sampling_method, "PSO"),
				initial_lb(initial_lb), initial_ub(initial_ub) {}

			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return optim::pso(theta, objective, &optdata, settings);
			}

			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.conv_failure_switch = conv_failure_switch;
				settings_.iter_max = iter_max;
				settings_.grad_err_tol = err_tol;
				settings_.vals_bound = vals_bound;
				settings_.print_level = verbosity;
				settings_.pso_settings.center_particle = center_particle;
				settings_.pso_settings.n_pop = n_pop;
				settings_.pso_settings.n_gen = n_gen;
				settings_.pso_settings.inertia_method = inertia_method;
				settings_.pso_settings.velocity_method = velocity_method;
				settings_.pso_settings.par_initial_w = par_initial_w;
				settings_.pso_settings.par_w_damp = par_w_damp;
				settings_.pso_settings.par_w_min = par_w_min;
				settings_.pso_settings.par_w_max = par_w_max;
				settings_.pso_settings.par_c_cog = par_c_cog;
				settings_.pso_settings.par_c_soc = par_c_soc;
				settings_.pso_settings.par_initial_c_cog = par_initial_c_cog;
				settings_.pso_settings.par_final_c_cog = par_final_c_cog;
				settings_.pso_settings.par_initial_c_soc = par_initial_c_soc;
				settings_.pso_settings.par_final_c_soc = par_final_c_soc;
				if (initial_lb.size()) { settings_.pso_settings.initial_lb = initial_lb; }
				if (initial_ub.size()) { settings_.pso_settings.initial_ub = initial_ub; }
				return settings_;
			}
			bool center_particle = true;
			int n_pop = 100;
			int n_gen = 1000;
			int inertia_method = 1; // 1 for linear decreasing between w_min and w_max; 2 for dampening
			double par_initial_w = 1.0;
			double par_w_damp = 0.99;
			double par_w_min = 0.10;
			double par_w_max = 0.99;
			int velocity_method = 1; // 1 for fixed; 2 for linear
			double par_c_cog = 2.0;
			double par_c_soc = 2.0;
			double par_initial_c_cog = 2.5;
			double par_final_c_cog = 0.5;
			double par_initial_c_soc = 0.5;
			double par_final_c_soc = 2.5;
			TVector initial_lb; // this will default to -0.5
			TVector initial_ub; // this will default to  0.5
		};

		struct DifferentialEvolution : public Solver {
			DifferentialEvolution() : Solver("DE") {}
			DifferentialEvolution(const int& verbosity, const int& n_restarts) :
				Solver(verbosity, n_restarts, "DE") {}
			DifferentialEvolution(const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "DE") {}
			DifferentialEvolution(const int& verbosity, const int& n_restarts, const std::string& sampling_method,
				const TVector& initial_lb, const TVector& initial_ub) :
				Solver(verbosity, n_restarts, sampling_method, "DE"),
				initial_lb(initial_lb), initial_ub(initial_ub) {}

			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.conv_failure_switch = conv_failure_switch;
				settings_.iter_max = iter_max;
				settings_.grad_err_tol = err_tol;
				settings_.vals_bound = vals_bound;
				settings_.print_level = verbosity;
				settings_.de_settings.n_pop = n_pop;
				settings_.de_settings.n_pop_best = n_pop_best;
				settings_.de_settings.n_gen = n_gen;
				settings_.de_settings.pmax = pmax;
				settings_.de_settings.max_fn_eval = max_fn_eval;
				settings_.de_settings.mutation_method = mutation_method;
				settings_.de_settings.check_freq = check_freq;
				settings_.de_settings.par_F = par_F;
				settings_.de_settings.par_CR = par_CR;
				settings_.de_settings.par_F_l = par_F_l;
				settings_.de_settings.par_F_u = par_F_u;
				settings_.de_settings.par_tau_F = par_tau_F;
				settings_.de_settings.par_tau_CR = par_tau_CR;
				if (initial_lb.size()) { settings_.de_settings.initial_lb = initial_lb; }
				if (initial_ub.size()) { settings_.de_settings.initial_ub = initial_ub; }
				return settings_;
			}
			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return optim::de(theta, objective, &optdata, settings);
			}

			int n_pop = 200;
			int n_pop_best = 6;
			int n_gen = 100;
			int pmax = 4;
			int max_fn_eval = 100000;
			int mutation_method = 1; // 1 = rand; 2 = best
			int check_freq = -1;
			double par_F = 0.8;
			double par_CR = 0.9;
			double par_F_l = 0.1;
			double par_F_u = 1.0;
			double par_tau_F = 0.1;
			double par_tau_CR = 0.1;
			TVector initial_lb; // this will default to -0.5
			TVector initial_ub; // this will default to  0.5
		};

		struct LBFGSB : public Solver {
			LBFGSB() : Solver("LBFGSB") {}
			LBFGSB(const int& verbosity, const int& n_restarts) :
				Solver(verbosity, n_restarts, "LBFGSB") {}
			LBFGSB(const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "LBFGSB") {}
			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.lbfgsb_settings.solver_iterations = solver_iterations;
				settings_.lbfgsb_settings.gradient_norm = gradient_norm;
				settings_.lbfgsb_settings.x_delta = x_delta;
				settings_.lbfgsb_settings.f_delta = f_delta;
				settings_.lbfgsb_settings.x_delta_violations = x_delta_violations;
				settings_.lbfgsb_settings.f_delta_violations = f_delta_violations;
				return settings_;
			}
			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return true;
				/*return optim::lbfgs(theta, objective, &optdata, settings);*/
			}
			
			int solver_iterations = 100;
			double gradient_norm = 1e-4;
			double x_delta = 1e-9;
			double f_delta = 1e-9;
			int x_delta_violations = 5;
			int f_delta_violations = 5;
		};
		
		struct GradientDescent : public Solver {
			GradientDescent(const int& method) : Solver("GD"), method(method) {}
			GradientDescent(const int& method, const int& verbosity, const int& n_restarts) :
				Solver(verbosity, n_restarts, "GD"), method(method) {}
			GradientDescent(const int& method, const int& verbosity, const int& n_restarts, const std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "GD"), method(method) {}

			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.conv_failure_switch = conv_failure_switch;
				settings_.iter_max = iter_max;
				settings_.grad_err_tol = err_tol;
				settings_.vals_bound = vals_bound;
				settings_.print_level = verbosity;
				settings_.gd_settings.method = method;
				settings_.gd_settings.par_step_size = step_size;
				settings_.gd_settings.step_decay = step_decay;
				settings_.gd_settings.step_decay_periods = step_decay_periods;
				settings_.gd_settings.step_decay_val = step_decay_val;
				settings_.gd_settings.par_momentum = momentum;
				settings_.gd_settings.par_ada_norm_term = ada_norm;
				settings_.gd_settings.ada_max = ada_max;
				settings_.gd_settings.par_adam_beta_1 = adam_beta_1;
				settings_.gd_settings.par_adam_beta_2 = adam_beta_2;
				return settings_;
			}
			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return optim::gd(theta, objective, &optdata, settings);
			}
			
			// GD method
			int method;
			// step size, or 'learning rate'
			double step_size = 0.1;
			// decay
			bool step_decay = false;
			optim::uint_t step_decay_periods = 10;
			double step_decay_val = 0.5;
			// momentum parameter
			double momentum = 0.9;
			// Ada parameters
			double ada_norm = 10e-08;
			double ada_rho = 0.9;
			bool ada_max = false;
			// Adam parameters
			double adam_beta_1 = 0.9;
			double adam_beta_2 = 0.999;

		};
		
		struct ConjugateGradient : public Solver {
			ConjugateGradient() : Solver("CG") {}
			ConjugateGradient(int& verbosity, int& n_restarts) :
				Solver(verbosity, n_restarts, "CG") {}
			ConjugateGradient(int& verbosity, int& n_restarts, std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "CG") {}

			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.conv_failure_switch = conv_failure_switch;
				settings_.iter_max = iter_max;
				settings_.grad_err_tol = err_tol;
				settings_.vals_bound = vals_bound;
				settings_.print_level = verbosity;
				settings_.cg_settings.restart_threshold = restart_threshold;
				return settings_;
			}
			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return optim::cg(theta, objective, &optdata, settings);
			}
			double restart_threshold = 0.1;
		};

		struct NelderMead : public Solver {
			NelderMead() : Solver("NM") {}
			NelderMead(int& verbosity, int& n_restarts) :
				Solver(verbosity, n_restarts, "NM") {}
			NelderMead(int& verbosity, int& n_restarts, std::string& sampling_method) :
				Solver(verbosity, n_restarts, sampling_method, "NM") {}

			SolverSettings settings() const override {
				SolverSettings settings_;
				settings_.conv_failure_switch = conv_failure_switch;
				settings_.iter_max = iter_max;
				settings_.grad_err_tol = err_tol;
				settings_.vals_bound = vals_bound;
				settings_.print_level = verbosity;
				settings_.nm_settings.adaptive_pars = adaptive_pars;
				settings_.nm_settings.par_alpha = par_alpha;
				settings_.nm_settings.par_beta = par_beta;
				settings_.nm_settings.par_gamma = par_gamma;
				settings_.nm_settings.par_delta = par_delta;
				return settings_;
			}

			bool solve
			(TVector& theta,
			std::function<double(const TVector& x, TVector* grad, void* optdata)> objective,
			OptData optdata, SolverSettings& settings) const
			{
				return optim::nm(theta, objective, &optdata, settings);
			}

			bool adaptive_pars = true;
			double par_alpha = 1.0; // reflection parameter
			double par_beta = 0.5; // contraction parameter
			double par_gamma = 2.0; // expansion parameter
			double par_delta = 0.5; // shrinkage parameter
		};
	}
	
	namespace models {
		using optimizer::Solver;
		using optimizer::LBFGSB;
		class Model {

		public:
			Model() = default;
			Model(const std::string& name) : name(name) {}
			Model(const std::string& name, const TMatrix& inputs, const TMatrix& outputs) : name(name), inputs(inputs), outputs(outputs) {}
			virtual void train() = 0;

		public:
			std::string name = "Model";
			TMatrix inputs;
			TMatrix outputs;
		};

		class GP : public Model {

		public:
			GP() : Model("GP") {
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
				solver = std::static_pointer_cast<Solver>(_solver);
				kernel = std::static_pointer_cast<Kernel>(_kernel);
			}
			GP(const GP& g) : Model(g) {
				likelihood_variance = g.likelihood_variance;
				kernel = g.kernel;
				solver = g.solver;
			}
			GP& operator=(const GP& g)
			{
				likelihood_variance = g.likelihood_variance;
				kernel = g.kernel;
				solver = g.solver;
				return *this;
			}		
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver) : Model("GP"), kernel(kernel), solver(solver) {}			
			GP(shared_ptr<Solver> solver, const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs), solver(solver) {
				shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
				kernel = std::static_pointer_cast<Kernel>(_kernel);
				if (kernel->length_scale.size() != inputs.cols() && kernel->length_scale.size() == 1)
				{   // Expand lengthscale dimensions
					kernel->length_scale = TVector::Constant(inputs.cols(), 1, kernel->length_scale.value()(0));
				}
			}
			GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs), kernel(kernel) {
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				solver = std::static_pointer_cast<Solver>(_solver);
			}									
			GP(const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs) {
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
				solver = std::static_pointer_cast<Solver>(_solver);
				kernel = std::static_pointer_cast<Kernel>(_kernel);
				if (kernel->length_scale.size() != inputs.cols() && kernel->length_scale.size() == 1)
				{   // Expand lengthscale dimensions
					kernel->length_scale = TVector::Constant(inputs.cols(), 1, kernel->length_scale.value()(0));
				}
			};			
			GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance) :
				Model("GP", inputs, outputs), kernel(kernel) {
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				solver = std::static_pointer_cast<Solver>(_solver);
				if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			GP(shared_ptr<Solver> solver, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance) :
				Model("GP", inputs, outputs), solver(solver) {
				shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
				kernel = std::static_pointer_cast<Kernel>(_kernel);
				if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver, const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs), kernel(kernel) {};			
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance) :
				Model("GP", inputs, outputs), kernel(kernel), solver(solver)
			{
				if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver, const TMatrix& inputs, const TMatrix& outputs, const Parameter<double>& likelihood_variance) :
				Model("GP", inputs, outputs), kernel(kernel), solver(solver)
			{
				if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			
			
			// GPNode Constructors
			GP(shared_ptr<Kernel> kernel, const double& likelihood_variance) : Model("GP"), kernel(kernel) {
				if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				solver = std::static_pointer_cast<Solver>(_solver);
			}
			GP(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance) : Model("GP"), kernel(kernel)
			{
				if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				solver = std::static_pointer_cast<Solver>(_solver);
			}
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver, const double& likelihood_variance) :
				Model("GP"), kernel(kernel), solver(solver) {
				if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			GP(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver, const Parameter<double>& likelihood_variance) :
				Model("GP"), kernel(kernel), solver(solver)
			{
				if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
				this->likelihood_variance = likelihood_variance;
			}
			GP(shared_ptr<Kernel> kernel) : Model("GP"), kernel(kernel) {
				shared_ptr<LBFGSB> _solver = make_shared<LBFGSB>();
				solver = std::static_pointer_cast<Solver>(_solver);
			}
			
			virtual void train() = 0;
			virtual TVector gradients() { TVector tmp; return tmp; }
			virtual double log_marginal_likelihood() { return 0.0; }
			virtual void set_params(const TVector& new_params) = 0; 
			virtual TVector get_params() { TVector tmp; return tmp; }
		public:
			Parameter<double> likelihood_variance = { "likelihood_variance ", 1e-8, "none" };
			shared_ptr<Kernel> kernel;
			shared_ptr<Solver> solver;
			TVector mean = TVector::Zero(1);
		};
	}

	namespace gaussian_process {
		using optimizer::Solver;
		using optimizer::PSO;
		using optimizer::SolverSettings;
		using optimizer::OptData;
		using models::GP;
		using namespace fdml::utilities::sobol;

		namespace LBFGSB {
			using Solver = cppoptlib::solver::Solver;
			using Function = cppoptlib::function::Function;
			using SolverState = cppoptlib::solver::SolverState;
			using FunctionState = cppoptlib::function::FunctionState;
			using Callback = cppoptlib::solver::Callback;
			using optimizer::SolverSettings;

			SolverState StoppingState(const SolverSettings& settings)
			{
				SolverState state;
				state.num_iterations = settings.lbfgsb_settings.solver_iterations;
				state.x_delta = settings.lbfgsb_settings.x_delta;
				state.x_delta_violations = settings.lbfgsb_settings.x_delta_violations;
				state.f_delta = settings.lbfgsb_settings.f_delta;
				state.f_delta_violations = settings.lbfgsb_settings.f_delta_violations;
				state.gradient_norm = settings.lbfgsb_settings.gradient_norm;
				state.condition_hessian = double{ 0 };
				state.status = cppoptlib::solver::SolverStatus::NotStarted;
				return state;
			}

			Callback Verbose0() { return [](const FunctionState& fstate, const SolverState& solstate, int iter) {}; }

			Callback Verbose1()
			{
				return
					[](const FunctionState& fstate, const SolverState& solstate, int iter)
				{
					if (iter == 0) {
						std::cout << std::setw(10) << std::left << "ITERATION"
							<< std::setw(17) << std::right << "FUNCTION VALUE"
							<< std::setw(17) << std::right << "DELTA_X"
							<< std::setw(17) << std::right << "DELTA_F" << std::endl;
					}
					else {
						if (std::isnan(fstate.value))
						{
							std::cout << std::setw(10) << std::left << solstate.num_iterations
								<< std::setw(17) << std::right << "NAN"
								<< std::setw(17) << std::right << "NAN"
								<< std::setw(17) << std::right << "NAN" << std::endl;
						}
						else {
							std::cout << std::setw(10) << std::left << solstate.num_iterations
								<< std::setw(17) << std::right << std::setprecision(4) << std::fixed << fstate.value
								<< std::setw(17) << std::right << std::setprecision(4) << std::fixed << solstate.x_delta
								<< std::setw(17) << std::right << std::setprecision(4) << std::fixed << solstate.f_delta << std::endl;
						}
					}
				};
			}

			Callback Verbose2()
			{
				return
					[](const FunctionState& fstate, const SolverState& solstate, int iter)
				{
					if (std::isnan(fstate.value))
					{
						std::cout << std::setw(10) << std::left << solstate.num_iterations
							<< std::setw(17) << std::right << "NAN"
							<< std::setw(17) << std::right << "NAN"
							<< std::setw(17) << std::right << "NAN" << std::endl;
					}
					else {
						std::cout << std::setw(10) << std::left << solstate.num_iterations + 1
							<< std::setw(19) << std::right << std::setprecision(4) << std::fixed << fstate.value
							<< std::setw(18) << std::right << std::setprecision(4) << std::fixed << solstate.x_delta
							<< std::setw(18) << std::right << std::setprecision(4) << std::fixed << solstate.f_delta << std::endl;
					}
				};

			}
		
			// Standard GPR Objective Function
			struct GPRObjective : public Function {
				EIGEN_MAKE_ALIGNED_OPERATOR_NEW
				GP* model;
				GPRObjective(GP* model) : model(model) {}
				double operator()(const TVector& x) const override {
					model->set_params(x);
					return -model->log_marginal_likelihood();
				}
				void gradient(const TVector& x, TVector* grad) const override { (*grad) = model->gradients() * -1.0; }
				FunctionState evaluate(const TVector& x, const int order = 1) const override {
					FunctionState state(static_cast<const int>(x.rows()), order);
					state.value = this->operator()(x);
					state.x = x;
					this->gradient(x, &state.gradient);
					return state;
				}
			};
		
		}

		
		class GPR : public GP {
		public:

			GPR(const TMatrix& inputs, const TMatrix& outputs) : GP(inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			GPR(const TMatrix& inputs, const TMatrix& outputs, shared_ptr<Solver> solver) : GP(solver, inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			//
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs) : GP(kernel, inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			};			
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, shared_ptr<Solver> solver) :
				GP(kernel, solver, inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			};			
			
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance_) : GP(kernel, inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
				likelihood_variance = likelihood_variance_;
			};			
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance, shared_ptr<Solver> solver) :
				GP(kernel, solver, inputs, outputs, likelihood_variance) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			};									
			//
			GPR(const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance_) : GP(inputs, outputs) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
				likelihood_variance = likelihood_variance_;
			};			
			GPR(const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance_, shared_ptr<Solver> solver) : GP(solver, inputs, outputs){
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
				likelihood_variance = likelihood_variance_;
			};									
			GPR(const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance_, const double& scale_) :
				GP(inputs, outputs) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
				likelihood_variance = likelihood_variance_;
			};			
			GPR(const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance_, const double& scale_, shared_ptr<Solver> solver) : GP(solver, inputs, outputs) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
				likelihood_variance = likelihood_variance_;
			};
			//
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs,
				const double& likelihood_variance, const double& scale_) :
				GP(kernel, inputs, outputs, likelihood_variance) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}		
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs,
				const double& likelihood_variance, const double& scale_, shared_ptr<Solver> solver) :
				GP(kernel, solver, inputs, outputs, likelihood_variance) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}			
			// Pickle
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs,
				const Parameter<double>& likelihood_variance, const Parameter<double>& scale, shared_ptr<Solver> solver) :
				GP(kernel, solver, inputs, outputs, likelihood_variance), scale(scale) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			double log_marginal_likelihood() {
				// Compute Log Likelihood [Rasmussen, Eq 2.30]
				double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
				double YKinvY = (outputs.transpose() * alpha)(0);
				double NLL = 0.0;
				if (*scale.is_fixed) { NLL = 0.5 * (logdet + YKinvY); }
				else { NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value()))); }
				NLL -= log_prior();
				// log_marginal_likelihood by default takes the objective function as LL
				// since this function computes NLL directly, we take the negative to output LL
				return -NLL;
			}
			double log_prior() {
				// Gamma Distribution
				// self.g = lambda x : (self.prior_coef[0] - 1) * np.log(x) - self.prior_coef[1] * x			
				const double shape = 1.6;
				const double rate = 0.3;
				double lp = 0.0;
				if (!(*kernel->length_scale.is_fixed)) {
					lp += (((shape - 1.0) * log(kernel->length_scale.value().array())) - (rate * kernel->length_scale.value().array())).sum();
				}
				if (!(*likelihood_variance.is_fixed)) {
					lp += ((shape - 1.0) * log(likelihood_variance.value())) - (rate * likelihood_variance.value());
				}
				return lp;
			}
			TVector log_prior_gradient() {
				// Gamma Distribution
				// self.gfod = lambda x : (self.prior_coef[0] - 1) - self.prior_coef[1] * x
				const double shape = 1.6;
				const double rate = 0.3;
				TVector lpg;
				if (!(*kernel->length_scale.is_fixed)) {
					lpg = (shape - 1.0) - (rate * kernel->length_scale.value().array()).array();
				}
				if (!(*likelihood_variance.is_fixed)) {
					lpg.conservativeResize(lpg.size() + 1);
					lpg.tail(1)(0) = (shape - 1.0) - (rate * likelihood_variance.value());
				}
				return lpg;
			}
			void set_params(const TVector& new_params) override
			{
				// Explicitly mention order? order = {StationaryKernel_lengthscale, StationaryKernel_variance, likelihood_variance}
				kernel->set_params(new_params);
				if (!(*likelihood_variance.is_fixed)) { likelihood_variance.transform_value(new_params.tail(1)(0)); }
				update_cholesky();
			}			
			void get_bounds(TVector& lower, TVector& upper, bool transformed = false) {
				kernel->get_bounds(lower, upper, transformed);

				if (!(*likelihood_variance.is_fixed)) {
					if (transformed) { likelihood_variance.transform_bounds(); }
					lower.conservativeResize(lower.rows() + 1);
					upper.conservativeResize(upper.rows() + 1);
					lower.tail(1)(0) = likelihood_variance.get_bounds().first;
					upper.tail(1)(0) = likelihood_variance.get_bounds().second;
				}
			}
			TVector get_params() override {
				TVector params = kernel->get_params();
				if (!(*likelihood_variance.is_fixed)) {
					likelihood_variance.transform_value(true);
					params.conservativeResize(params.rows() + 1);
					params.tail(1)(0) = likelihood_variance.value();
				}
				return params;
			}
			Eigen::Index params_size() {
				TVector param = get_params();
				return param.size();
			}
			TVector gradients() override {
				// dNLL = alpha*alpha^T - K^-1 [Rasmussen, Eq 5.9]
				if (alpha.size() == 0) { update_cholesky(); }
				TMatrix aaT = alpha * alpha.transpose().eval();
				TMatrix Kinv = chol.solve(TMatrix::Identity(inputs.rows(), inputs.rows()));
				TMatrix dNLL = 0.5 * (aaT - Kinv); // dL_dK

				std::vector<double> grad;
				// Get dK/dlengthscale and dK/dvariance -> {dK/dlengthscale, dK/dvariance}
				kernel->gradients(inputs, dNLL, D, K, grad);
				if (!(*likelihood_variance.is_fixed)) { grad.push_back(dNLL.diagonal().sum()); }
				TVector _grad = Eigen::Map<TVector>(grad.data(), grad.size());
				if (!(*scale.is_fixed)) { _grad.array() /= scale.value(); }
				// gamma log_prior derivative
				TVector lpg = log_prior_gradient();
				_grad -= lpg;
				return _grad;
			}
			const std::string model_type() const { return "GPR"; }
			MatrixVariant predict(const TMatrix& X, bool return_var = false)
			{
				update_cholesky();
				TMatrix Ks(inputs.rows(), X.rows());
				Ks.noalias() = kernel->K(inputs, X);
				TMatrix mu = Ks.transpose() * alpha;
				if (return_var) {
					TMatrix Kss = kernel->diag(X);
					TMatrix V = chol.solve(Ks);
					TMatrix var = abs((scale.value() * (Kss - (Ks.transpose() * V).diagonal()).array()));
					return std::make_pair(mu, var);
				}
				else { return mu; }
			}			
		
			
			void train() override {
				TVector lower_bound, upper_bound, theta;
				get_bounds(lower_bound, upper_bound, true);

				TMatrix X(solver->n_restarts, lower_bound.size());
				/* Try Uniform Distribution */
				if (solver->sampling_method == "uniform") {
					std::mt19937 generator(std::random_device{}());
					for (Eigen::Index c = 0; c < X.cols(); ++c) {
						std::uniform_real_distribution<> distribution(lower_bound[c], upper_bound[c]);
						auto uniform = [&](int, Eigen::Index) {return distribution(generator); };
						X.col(c) = TMatrix::NullaryExpr(solver->n_restarts, 1, uniform);
					}
				}
				else if (solver->sampling_method == "sobol") {
					X = generate_sobol(solver->n_restarts, lower_bound.size());
					scale_to_range(X, lower_bound, upper_bound);
					X.array() += 1e-6;
				}
				else { std::runtime_error("Unrecognized Sampling Method"); }

				if (solver->type == "LBFGSB") {

					std::vector<double> fhistory;
					std::vector<TVector> phistory;
					LBFGSB::GPRObjective objective(this);
					const LBFGSB::SolverState stopping_state = LBFGSB::StoppingState(solver->settings());
					cppoptlib::solver::LBFGSB::LBFGSB lbfgsb_solver(stopping_state);
					
					if (solver->verbosity == 0) { lbfgsb_solver.SetStepCallback(LBFGSB::Verbose0()); }
					else if (solver->verbosity == 1) { lbfgsb_solver.SetStepCallback(LBFGSB::Verbose1()); }
					else if (solver->verbosity == 2) { lbfgsb_solver.SetStepCallback(LBFGSB::Verbose2()); }

					lbfgsb_solver.SetLowerBound(lower_bound);
					lbfgsb_solver.SetUpperBound(upper_bound);
					objective_value = std::numeric_limits<double>::infinity();

					for (int i = 0; i < solver->n_restarts; ++i) {
						theta = X.row(i);
						auto [solution, solver_state] = lbfgsb_solver.minimize(objective, theta, i + 1, solver->verbosity);
						int local_min_idx = std::min_element(solver_state.f_history.begin(), solver_state.f_history.end()) - solver_state.f_history.begin();
						double local_min_f = *std::min_element(solver_state.f_history.begin(), solver_state.f_history.end());
						if (local_min_f < objective_value) {
							objective_value = local_min_f;
							theta = solver_state.x_history.at(local_min_idx);
							fhistory.push_back(local_min_f);
							phistory.push_back(theta);
						}
					}
					// Perform Checks on local history
					int min_idx = std::min_element(fhistory.begin(), fhistory.end()) - fhistory.begin();
					double min_f = *std::min_element(fhistory.begin(), fhistory.end());
					objective_value = min_f;
					theta = phistory[min_idx];
					// Fail Safe Optimizer -> PSO [ Add a set_second_optimizer() method? ]
					if (theta.array().isNaN().any() || theta.array().isInf().any()) {
						if (solver->verbosity > 0) { std::cout << "LBFGSB FAILED -> RUNNING PSO" << std::endl; }
						// Better way than to swap pointers?
						shared_ptr<PSO> _solver = make_shared<PSO>(solver->verbosity, solver->n_restarts, solver->sampling_method);
						shared_ptr<Solver> solver2 = std::static_pointer_cast<Solver>(_solver);
						solver.swap(solver2);
						from_optim_(theta, lower_bound, upper_bound, X);
					}
				}
				else {from_optim_(theta, lower_bound, upper_bound, X);}
			}

		private:
			void from_optim_(TVector& theta, const TVector& lower_bound, const TVector& upper_bound, const TMatrix& X) {
				std::vector<double> fhistory;
				std::vector<TVector> phistory;
				OptData optdata;
				auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
				{return objective_(x, grad, nullptr, opt_data); };
				SolverSettings settings = solver->settings();
				if (solver->vals_bound) {
					settings.lower_bounds = lower_bound.array();
					settings.upper_bounds = upper_bound.array();
				}
				bool success = false;
				for (int i = 0; i < solver->n_restarts; ++i) {
					theta = X.row(i);
					success = solver->solve(theta, objective, optdata, settings);
					fhistory.push_back(-log_marginal_likelihood());
					phistory.push_back(get_params());
				}
				int minElementIndex = std::min_element(fhistory.begin(), fhistory.end()) - fhistory.begin();
				objective_value = *std::min_element(fhistory.begin(), fhistory.end());
				set_params(phistory[minElementIndex]);
			}
			double objective_(const TVector& x, TVector* grad, TVector* hess, void* opt_data) {
				set_params(x);
				if (grad) { (*grad) = gradients() * -1.0; }
				return -log_marginal_likelihood();
			}
		protected:
			void update_cholesky() {
				TMatrix noise = TMatrix::Identity(inputs.rows(), outputs.rows());
				K = kernel->K(inputs, inputs, D);
				K += (noise * likelihood_variance.value());
				chol = K.llt();
				alpha = chol.solve(outputs);
				// scale is not considered a variable in optimization, it is directly linked to chol
				if (!(*scale.is_fixed)) {
					scale = (outputs.transpose() * alpha)(0) / outputs.rows();
				}
			}
		protected:
			TVector  alpha;
			TLLT	 chol;
			TMatrix	 K;
			TMatrix	 D;
		public:			
			Parameter<double> scale = { "scale", 1.0, "none" };
			double objective_value = 0.0;
			BoolVector missing;

		};

	}


}
#endif