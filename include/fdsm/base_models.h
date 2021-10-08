#ifndef BASEMODELS_H
#define BASEMODELS_H
#include <sstream>
#include <iomanip>
#include <cppoptlibmod/function.h>
#include <cppoptlibmod/lbfgsb.h>
#include "./kernels.h"

namespace opt = cppoptlib;

namespace fdsm::base_models {
	using namespace fdsm::kernels;

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
			shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
			kernel = std::static_pointer_cast<Kernel>(_kernel);
		}
		GP(const GP& g) : Model(g) {
			likelihood_variance = g.likelihood_variance;
			kernel = g.kernel;
		}
		GP& operator=(const GP& g)
		{
			likelihood_variance = g.likelihood_variance;
			kernel = g.kernel;
			return *this;
		}
		GP(shared_ptr<Kernel> kernel) : Model("GP"), kernel(kernel) {}
		GP(const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs) {
			shared_ptr<SquaredExponential> _kernel = make_shared<SquaredExponential>(1.0, 1.0);
			kernel = std::static_pointer_cast<Kernel>(_kernel);
			if (kernel->length_scale.size() != inputs.cols() && kernel->length_scale.size() == 1)
			{   // Expand lengthscale dimensions
				kernel->length_scale = TVector::Constant(inputs.cols(), 1, kernel->length_scale.value()(0));
			}
		};
		GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs) : Model("GP", inputs, outputs), kernel(kernel) {};
		GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, double& likelihood_variance) : Model("GP", inputs, outputs), kernel(kernel) {}
		GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, Parameter<double>& likelihood_variance) : Model("GP", inputs, outputs), kernel(kernel)
		{
			if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
			this->likelihood_variance = likelihood_variance;
		}
		// GPNode Constructors
		GP(shared_ptr<Kernel> kernel, double& likelihood_variance) : Model("GP"), kernel(kernel) {}
		GP(shared_ptr<Kernel> kernel, Parameter<double>& likelihood_variance) : Model("GP"), kernel(kernel)
		{
			if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
			this->likelihood_variance = likelihood_variance;
		}

		virtual double objective_fxn() { return 0.0; }
		virtual void set_params(const TVector& new_params) = 0;
		virtual TVector get_params() { TVector tmp; return tmp; }
		virtual TVector gradients() { TVector tmp; return tmp; }


	public:
		Parameter<double> likelihood_variance = { "likelihood_variance ", 1e-8, "none" };
		shared_ptr<Kernel> kernel;
		TVector mean = TVector::Zero(1);
	};

}

namespace fdsm::base_models::optimizer {
	using Solver = opt::solver::Solver;
	using Function = opt::function::Function;
	using SolverState = opt::solver::SolverState;
	using FunctionState = opt::function::FunctionState;
	using Callback = opt::solver::Callback;

	struct SolverSettings {
		int verbosity;
		int n_restarts;
		int solver_iterations;
		double gradient_norm;
		double x_delta;
		double f_delta;
		int x_delta_violations;
		int f_delta_violations;
	};

	SolverState StoppingState(const SolverSettings& settings)
	{
		SolverState state;
		state.num_iterations = settings.solver_iterations;
		state.x_delta = settings.x_delta;
		state.x_delta_violations = settings.x_delta_violations;
		state.f_delta = settings.f_delta;
		state.f_delta_violations = settings.f_delta_violations;
		state.gradient_norm = settings.gradient_norm;
		state.condition_hessian = double{ 0 };
		state.status = opt::solver::SolverStatus::NotStarted;
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

	namespace objective_functions {
		// Standard GP Objective Function
		struct LogMarginalLikelihood : public Function {
			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
				GP* model;

			LogMarginalLikelihood(GP* model) : model(model) {}

			double operator()(const TVector& x) const override {
				model->set_params(x);
				return -model->objective_fxn();
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
}

namespace fdsm::base_models::gaussian_process {
	using namespace fdsm::base_models::optimizer;

	class GPR : public GP {

	private:
		using GPRSolver = opt::solver::LBFGSB::LBFGSB;


	public:
		GPR() : GP() {}
		GPR(shared_ptr<Kernel> kernel) : GP(kernel) {}
		GPR(const TMatrix& inputs, const TMatrix& outputs) : GP(inputs, outputs) {}
		GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs) : GP(kernel, inputs, outputs) {};
		GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, double& likelihood_variance) : GP(kernel, inputs, outputs, likelihood_variance) {}
		GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, Parameter<double>& likelihood_variance) : GP(kernel, inputs, outputs, likelihood_variance) {}

		double log_likelihood() {
			//TLLT cholesky(K);
			double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
			double quad = (outputs.array() * (K.llt().solve(outputs)).array()).sum();
			double lml = -0.5 * (logdet + quad);
			return lml;
		}

		double objective_fxn() override {
			// Compute Log Likelihood [Rasmussen, Eq 2.30]
			double LL = -0.5 * (outputs.transpose() * alpha)(0)
				- chol.matrixLLT().diagonal().array().log().sum()
				- (0.5 * inputs.rows()) * log(2.0 * PI);
			return LL;
		}

		void train() override {
			objective_functions::LogMarginalLikelihood obj_fxn(this);
			const SolverState stopping_state = StoppingState(solver_settings);
			GPRSolver solver(stopping_state);

			if (solver_settings.verbosity == 0) { solver.SetStepCallback(Verbose0()); }
			else if (solver_settings.verbosity == 1) { solver.SetStepCallback(Verbose1()); }
			else if (solver_settings.verbosity == 2) { solver.SetStepCallback(Verbose2()); }

			TVector lower_bound, upper_bound;
			get_bounds(lower_bound, upper_bound, true);
			// For GPR set lower and upper bound default for better convergence

			lower_bound.array() = lower_bound.array().unaryExpr([](double v) { if (!(std::isfinite(v)) || v == 0.0) { return 1e-3; } else { return v; } });
			upper_bound.array() = upper_bound.array().unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0; });
			solver.SetLowerBound(lower_bound);
			solver.SetUpperBound(upper_bound);

			// Try different initial points
			TVector theta0, theta1;
			double _NLL = std::numeric_limits<double>::infinity();
			/* Try Random Uniform */
			//Vmt19937_64 gen_eigen;
			//TMatrix X = uniformReal<TMatrix>(solver_settings.n_restarts, lower_bound.rows(), gen_eigen);
			std::mt19937 generator(std::random_device{}());
			std::uniform_real_distribution<> distribution;
			auto uniform = [&](int, Eigen::Index) {return distribution(generator); };
			TMatrix X = TMatrix::NullaryExpr(solver_settings.n_restarts, lower_bound.rows(), uniform);

			/* Try Latin Hypercube Sample [Currently Not Working] */
			//int seed = 1234;
			//double* samples = new double(lower_bound.rows()*solver_settings.n_restarts);
			//LHS::latin_center(lower_bound.rows(), solver_settings.n_restarts, &seed, samples);
			//TMatrix X(solver_settings.n_restarts, lower_bound.rows());
			//delete [] samples;

			for (int i = 0; i < solver_settings.n_restarts; ++i) {
				theta0 = X.row(i);
				auto [solution, solver_state] = solver.minimize(obj_fxn, theta0, i+1, solver_settings.verbosity);
				int argmin_f = static_cast<int>(std::min_element(solver_state.f_history.begin(), solver_state.f_history.end()) - solver_state.f_history.begin());
				double min_f = *std::min_element(solver_state.f_history.begin(), solver_state.f_history.end());
				if (min_f < _NLL) { _NLL = min_f;  theta1 = solver_state.x_history.at(argmin_f); }
			}
			if (solver_settings.verbosity != 0)
			{
				std::cout << "NLL = " << _NLL << std::endl;
				std::cout << "Best param = " << theta1.transpose() << std::endl;
			}
			// Recompute on best params
			set_params(theta1);
			if (store_parameters) { history.push_back(theta1); }
			objective_value = _NLL;
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
			TVector params;
			params = kernel->get_params();
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
			if (alpha.size() == 0) { objective_fxn(); }
			TMatrix aaT = alpha * alpha.transpose().eval();
			TMatrix Kinv = chol.solve(TMatrix::Identity(inputs.rows(), inputs.rows()));
			TMatrix dNLL = 0.5 * (aaT - Kinv); // dL_dK

			std::vector<double> grad;
			// Get dK/dlengthscale and dK/dvariance -> {dK/dlengthscale, dK/dvariance}
			kernel->gradients(inputs, dNLL, D, K, grad);
			if (!(*likelihood_variance.is_fixed)) { grad.push_back(dNLL.diagonal().sum()); }
			return Eigen::Map<TVector>(grad.data(), grad.size());
		}

		MatrixVariant predict(const TMatrix& X, bool return_var = false) const
		{
			TMatrix Ks(inputs.rows(), X.rows());
			Ks.noalias() = kernel->K(inputs, X);
			TMatrix mu = Ks.transpose() * alpha;
			if (return_var) {
				TMatrix Kss = kernel->diag(X);
				TMatrix V = chol.solve(Ks);
				TMatrix var = Kss - (Ks.transpose() * V).diagonal();
				return std::make_pair(mu, var);
			}
			else { return mu; }
		}

		SolverSettings solver_settings{ 0, 30, 10, 1e-4, 1e-9, 1e-9, 1, 1 };
		BoolVector missing;
		double objective_value = 0.0;
		
		// Functions used in SIDGP
		void inputs_changed() {
			// TODO: ADD MEAN FXN
			if (mean.size() == 0) { mean = TVector::Zero(inputs.rows()); }
			TMatrix noise = TMatrix::Identity(inputs.rows(), inputs.rows());
			K.noalias() = kernel->K(inputs, inputs, D);
			K.noalias() += (noise * likelihood_variance.value());
			chol = K.llt();
		}
		void outputs_changed() {alpha = chol.solve(outputs); }
		bool is_psd() {
			if (!(K.isApprox(K.transpose())) || chol.info() == Eigen::NumericalIssue) { return false; }
			else { return true; }
		}
		
	protected:
		void update_cholesky() {
			TMatrix noise = TMatrix::Identity(inputs.rows(), outputs.rows());
			K = kernel->K(inputs, inputs, D);
			K += (noise * likelihood_variance.value());
			chol = K.llt();			
			alpha = chol.solve(outputs);
		}
	
	protected:
		bool store_parameters = false;
		std::vector<TVector> history;
		TVector  alpha;
		TLLT	 chol;
		TMatrix	 K;
		TMatrix	 D;



	};

	class GPNode : public GP {
	private:
		using GPRSolver = opt::solver::LBFGSB::LBFGSB;
	public:
		GPNode(shared_ptr<Kernel> kernel) : GP(kernel) {
			if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
			if (!kernel->variance.fixed()) { kernel->variance.fix(); }
		}
		GPNode(shared_ptr<Kernel> kernel, double& likelihood_variance, double& scale_) : GP(kernel, likelihood_variance), scale("scale", scale_) {
			if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
			if (!kernel->variance.fixed()) { kernel->variance.fix(); }
		}
		GPNode(shared_ptr<Kernel> kernel, Parameter<double>& likelihood_variance, Parameter<double>& scale) : GP(kernel, likelihood_variance), scale(scale) {
			if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
			if (!kernel->variance.fixed()) { kernel->variance.fix(); }
		}
		
		double objective_fxn() override {
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
		double log_likelihood() {
			update_cholesky();
			TMatrix _K = K.array() * scale.value();
			TLLT _chol(_K);
			double logdet = 2 * _chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
			double quad = (outputs.array() * (_chol.solve(outputs)).array()).sum();
			double lml = -0.5 * (logdet + quad);
			return lml;
		}
		double log_prior() {
			// Gamma Distribution
			// self.g = lambda x : (self.prior_coef[0] - 1) * np.log(x) - self.prior_coef[1] * x			
			const double shape = 1.6;
			const double rate = 0.3;
			double lp = 0.0;
			if (!(*kernel->length_scale.is_fixed)){
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
		
		void train() override {
			objective_functions::LogMarginalLikelihood obj_fxn(this);
			const SolverState stopping_state = StoppingState(solver_settings);
			GPRSolver solver(stopping_state);
			if (solver_settings.verbosity == 0) { solver.SetStepCallback(Verbose0()); }
			else if (solver_settings.verbosity == 1) { solver.SetStepCallback(Verbose1()); }
			else if (solver_settings.verbosity == 2) { solver.SetStepCallback(Verbose2()); }

			TVector lower_bound, upper_bound;
			get_bounds(lower_bound, upper_bound, true);
			lower_bound.array() = lower_bound.array().unaryExpr([](double v) { if (!(std::isfinite(v)) || v == 0.0) { return 1e-3; } else { return v; } });
			upper_bound.array() = upper_bound.array().unaryExpr([](double v) { return std::isfinite(v) ? v : 1000.0; });
			solver.SetLowerBound(lower_bound);
			solver.SetUpperBound(upper_bound);

			// Try different initial points
			TVector theta0, theta1;
			double _NLL = std::numeric_limits<double>::infinity();
			std::mt19937 generator(std::random_device{}());
			std::uniform_real_distribution<> distribution(lower_bound.minCoeff(), upper_bound.maxCoeff());
			auto uniform = [&](int, Eigen::Index) {return distribution(generator); };
			TMatrix X = TMatrix::NullaryExpr(solver_settings.n_restarts, lower_bound.rows(), uniform);

			for (int i = 0; i < solver_settings.n_restarts; ++i) {
				theta0 = X.row(i);
				auto [solution, solver_state] = solver.minimize(obj_fxn, theta0, i + 1, solver_settings.verbosity);
				int argmin_f = static_cast<int>(std::min_element(solver_state.f_history.begin(), solver_state.f_history.end()) - solver_state.f_history.begin());
				double min_f = *std::min_element(solver_state.f_history.begin(), solver_state.f_history.end());
				if (min_f < _NLL) { _NLL = min_f;  theta1 = solver_state.x_history.at(argmin_f); }
			}
			if (solver_settings.verbosity != 0)
			{
				std::cout << "NLL = " << _NLL << std::endl;
				std::cout << "Best param = " << theta1.transpose() << std::endl;
			}
			// Recompute on best params
			set_params(theta1);
			if (store_parameters) { history.push_back(theta1); }
			objective_value = _NLL;


		}		
		TVector gradients() override {
			// dNLL = alpha*alpha^T - K^-1 [Rasmussen, Eq 5.9]
			if (alpha.size() == 0) { objective_fxn(); }
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
		void linked_prediction(TVector& latent_mu, TVector& latent_var, const TMatrix& linked_mu, const TMatrix& linked_var, const int& n_thread) {

			update_cholesky();
			kernel->expectations(linked_mu, linked_var);
			Eigen::initParallel();
			if (n_thread == 0 || n_thread == 1){
				for (Eigen::Index i = 0; i < linked_mu.rows(); ++i) {
					TMatrix I = TMatrix::Ones(inputs.rows(), 1);
					TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
					kernel->IJ(I, J, linked_mu.row(i), linked_var.row(i), inputs, i);
					double trace = (K.llt().solve(J)).trace();
					double Ialpha = (I.cwiseProduct(alpha)).array().sum();
					latent_mu[i] = (Ialpha);
					latent_var[i] =
					(abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum()
					- (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
				}
			}
			else {
				
				thread_pool pool;
				int split = int(linked_mu.rows() / n_thread);
				const int remainder = int(linked_mu.rows()) % n_thread;
				auto task = [=, &latent_mu, &latent_var](int begin, int end) 
				{
					for (Eigen::Index i = begin; i < end; ++i) {
						TMatrix I = TMatrix::Ones(inputs.rows(), 1);
						TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
						kernel->IJ(I, J, linked_mu.row(i), linked_var.row(i), inputs, i);
						double trace = (K.llt().solve(J)).trace();
						double Ialpha = (I.cwiseProduct(alpha)).array().sum();
						latent_mu[i] = (Ialpha);
						latent_var[i] =
						(abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum()
						- (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
					}
				};
				for (int s = 0; s < n_thread; ++s) {
					pool.push_task(task, int(s * split), int(s * split) + split);
				}
				pool.wait_for_tasks();
				if (remainder > 0) {
					task(linked_mu.rows() - remainder, linked_mu.rows());
				}
				pool.reset();
			}
		}

		void set_params(const TVector& new_params) override
		{
			// Explicitly mention order? order = {StationaryKernel_lengthscale, StationaryKernel_variance, likelihood_variance}
			kernel->set_params(new_params);
			if (!(*likelihood_variance.is_fixed)) { likelihood_variance.transform_value(new_params.tail(1)(0)); }
			update_cholesky();
		}
		Eigen::Index params_size() {
			TVector param = get_params();
			return param.size();
		}
		TMatrix get_parameter_history() {
			if (history.size() == 0) 
			{ throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
			Eigen::Index param_size = params_size();
			TMatrix _history(history.size(), param_size);
			for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
				_history.row(i) = history[i];
			}
			return _history;
		}
		
		// Setters
		void set_inputs(const TMatrix& input) { inputs = input; }
		void set_outputs(const TMatrix& output) { outputs = output; }
		// Getters
		const TMatrix get_inputs() { return inputs; }
		const TMatrix get_outputs() { return outputs; }

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
			TVector params;
			params = kernel->get_params();
			if (!(*likelihood_variance.is_fixed)) {
				likelihood_variance.transform_value(true);
				params.conservativeResize(params.rows() + 1);
				params.tail(1)(0) = likelihood_variance.value();
			}
			return params;
		}

	public:
		Parameter<double> scale = { "scale", 1.0, "none" };
		SolverSettings solver_settings{ 0, 15, 10, 1e-5, 1e-9, 1e-9, 1, 1 };
		BoolVector missing;
		double objective_value = 0.0;
		bool store_parameters = false;
	
	protected:
		std::vector<TVector> history;
		TVector  alpha;
		TLLT	 chol;
		TMatrix	 K;
		TMatrix	 D;
	};



}
#endif