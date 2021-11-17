#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include <fdml/base.h>
#include <optim/optim.hpp>

namespace fdml::optimizers {

	using SolverSettings = optim::algo_settings_t;
	struct OptData {};

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
			settings_.lbfgsb_settings.m = m;
			settings_.lbfgsb_settings.past = past;
			settings_.lbfgsb_settings.max_iterations = max_iterations;
			settings_.lbfgsb_settings.submin = submin;
			settings_.lbfgsb_settings.max_linesearch = max_linesearch;
			settings_.lbfgsb_settings.epsilon = epsilon;
			settings_.lbfgsb_settings.delta = delta;
			settings_.lbfgsb_settings.min_step = min_step;
			settings_.lbfgsb_settings.max_step = max_step;
			settings_.lbfgsb_settings.ftol = ftol;
			settings_.lbfgsb_settings.wolfe = wolfe;
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

		int m = 6;
		int past = 1;
		int max_iterations = 10;
		int submin = 20;
		int max_linesearch = 20;
		double epsilon= 1e-5;
		double delta = 1e-10;
		double min_step = 1e-20;
		double max_step = 1e20;
		double ftol = 1e-4;
		double wolfe = 0.9;
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
#endif