#ifndef BASEMODELS_H
#define BASEMODELS_H
#include <sstream>
#include <iomanip>
#include <cppoptlibmod/function.h>
#include <cppoptlibmod/lbfgsb.h>
#include "./kernels.h"


// TODO: PLACE ALL NAMESPACES INSIDE BASE_MODELS

namespace opt = cppoptlib;

namespace fdml::base_models {
	using namespace fdml::kernels;

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
		GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance) : Model("GP", inputs, outputs), kernel(kernel) {
			if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
			this->likelihood_variance = likelihood_variance;
		}
		GP(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const Parameter<double>& likelihood_variance) : Model("GP", inputs, outputs), kernel(kernel)
		{
			if (likelihood_variance.value() < 0) { throw std::runtime_error("Noise Variance must be positive"); }
			this->likelihood_variance = likelihood_variance;
		}
		// GPNode Constructors
		GP(shared_ptr<Kernel> kernel, const double& likelihood_variance) : Model("GP"), kernel(kernel) {
			if (likelihood_variance < 0) { throw std::runtime_error("Noise Variance must be positive"); }
			this->likelihood_variance = likelihood_variance;
		}
		GP(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance) : Model("GP"), kernel(kernel)
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

	namespace optimizer {
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

	namespace gaussian_process {
		using namespace fdml::utilities::sobol;
		using namespace optimizer;

		class GPR : public GP {

		private:
			using GPRSolver = opt::solver::LBFGSB::LBFGSB;
		public:
			GPR() : GP() {}
			GPR(shared_ptr<Kernel> kernel) : GP(kernel) {}
			GPR(const TMatrix& inputs, const TMatrix& outputs) : GP(inputs, outputs) {}
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs) : GP(kernel, inputs, outputs) {};
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const double& likelihood_variance) : GP(kernel, inputs, outputs, likelihood_variance) {}
			GPR(shared_ptr<Kernel> kernel, const TMatrix& inputs, const TMatrix& outputs, const Parameter<double>& likelihood_variance) : GP(kernel, inputs, outputs, likelihood_variance) {}

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
				get_bounds(lower_bound, upper_bound, false);
				// For GPR set lower and upper bound default for better convergence
				//lower_bound.array() = lower_bound.array().unaryExpr([](double v) { if (!(std::isfinite(v)) || v == 0.0) { return 1e-3; } else { return v; } });
				//upper_bound.array() = upper_bound.array().unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0; });
				solver.SetLowerBound(lower_bound);
				solver.SetUpperBound(upper_bound);

				// Try different initial points
				TVector theta0, theta1;
				double _NLL = std::numeric_limits<double>::infinity();

				/* Try Uniform Distribution */
				TMatrix X(10000, lower_bound.size());
				std::mt19937 generator(std::random_device{}());
				for (Eigen::Index c = 0; c < X.cols(); ++c) {
					std::uniform_real_distribution<> distribution(lower_bound[c], upper_bound[c]);
					auto uniform = [&](int, Eigen::Index) {return distribution(generator); };
					X.col(c) = TMatrix::NullaryExpr(10000, 1, uniform);
				}

				/* Try Sobol Sample */
				//TMatrix X = generate_sobol(10000, lower_bound.rows());
				//scale(X, lower_bound, upper_bound);
				//X.array() += 1e-6;

				/* Try Normal Distribution*/
				//TMatrix X = gen_normal_matrix(10000, lower_bound.rows());

				int i = 0;
				std::vector<double> fhistory;
				std::vector<TVector> phistory;
				while(true) {
					TMatrix Xsub = X.block(i, 0, solver_settings.n_restarts, lower_bound.rows());
					for (int j = 0; j < solver_settings.n_restarts; ++j) {
						theta0 = Xsub.row(j);
						auto [solution, solver_state] = solver.minimize(obj_fxn, theta0, j + 1, solver_settings.verbosity);
						int argmin_f = static_cast<int>(std::min_element(solver_state.f_history.begin(), solver_state.f_history.end()) - solver_state.f_history.begin());
						double min_f = *std::min_element(solver_state.f_history.begin(), solver_state.f_history.end());
						if (min_f < _NLL) {
							_NLL = min_f;  theta1 = solver_state.x_history.at(argmin_f);
							fhistory.push_back(min_f); phistory.push_back(theta1);
						}
					}
					if (!std::isinf(_NLL)) { break; }
					i += solver_settings.n_restarts;
				}
				int minElementIndex = std::min_element(fhistory.begin(), fhistory.end()) - fhistory.begin();
				_NLL = *std::min_element(fhistory.begin(), fhistory.end());
				theta1 = phistory[minElementIndex];
				
				//std::vector<double> fhistory;
				//std::vector<TVector> phistory;
				//while (true) {
				//	for (int i = 0; i < X.rows(); ++i) {
				//		theta0 = X.row(i);
				//		auto [solution, solver_state] = solver.minimize(obj_fxn, theta0, i + 1, solver_settings.verbosity);
				//		int argmin_f = static_cast<int>(std::min_element(solver_state.f_history.begin(), solver_state.f_history.end()) - solver_state.f_history.begin());
				//		double min_f = *std::min_element(solver_state.f_history.begin(), solver_state.f_history.end());
				//		if (min_f < _NLL) { 
				//			_NLL = min_f;  theta1 = solver_state.x_history.at(argmin_f);
				//			fhistory.push_back(min_f); phistory.push_back(theta1);
				//			if (!std::isinf(_NLL)) { break; }
				//		}
				//	}
				//}
				//int minElementIndex = std::min_element(fhistory.begin(), fhistory.end()) - fhistory.begin();
				//_NLL = *std::min_element(fhistory.begin(), fhistory.end());
				//theta1 = phistory[minElementIndex];

				
				if (solver_settings.verbosity != 0)
				{
					std::cout << "NLL = " << _NLL << std::endl;
					std::cout << "Best param = " << theta1.transpose() << std::endl;
				}
				// Recompute on best params
				set_params(theta1);
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

			// Functions used in SIDGP
			void inputs_changed() {
				// TODO: ADD MEAN FXN
				if (mean.size() == 0) { mean = TVector::Zero(inputs.rows()); }
				TMatrix noise = TMatrix::Identity(inputs.rows(), inputs.rows());
				K.noalias() = kernel->K(inputs, inputs, D);
				K.noalias() += (noise * likelihood_variance.value());
				chol = K.llt();
			}
			void outputs_changed() { alpha = chol.solve(outputs); }
			bool is_psd() {
				if (!(K.isApprox(K.transpose())) || chol.info() == Eigen::NumericalIssue) { return false; }
				else { return true; }
			}

			const std::string model_type() const { return "GPR"; }
			
		protected:
			void update_cholesky() {
				TMatrix noise = TMatrix::Identity(inputs.rows(), outputs.rows());
				K = kernel->K(inputs, inputs, D);
				K += (noise * likelihood_variance.value());
				chol = K.llt();
				alpha = chol.solve(outputs);
			}
		protected:
			TVector  alpha;
			TLLT	 chol;
			TMatrix	 K;
			TMatrix	 D;
		public:
			SolverSettings solver_settings{ 0, 30, 100, 1e-9, 1e-9, 1e-9, 1, 1 };
			BoolVector missing;
			double objective_value = 0.0;

		};
	
	}


}

#endif