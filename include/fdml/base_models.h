#ifndef BASEMODELS_H
#define BASEMODELS_H
#include <sstream>
#include <iomanip>
#include <fdml/optimizers.h>
#include <lbfgsb_cpp/problem.h>
#include <lbfgsb_cpp/l_bfgs_b.h>

namespace fdml::base_models {
	using namespace fdml::kernels;
	namespace optimizer = fdml::optimizers;
	
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
			Parameter<double> likelihood_variance = { "likelihood_variance ", 1e-8, "logexp" };
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
		using namespace fdml::utilities;

		template<class T>
		struct Objective : public problem<T> {
			GP* model;
			Objective(GP* model, const int& dim) : model(model), problem<T>(dim) {}

			void gradient(const T& x, T& grad) {
				model->set_params(x);
				grad = model->gradients();
			}

			double operator()(const T& x){
				model->set_params(x);	
				return model->log_marginal_likelihood();
			}
		};

		/*
		template<class T>
		class matyas_function_base : public problem<T> {
		public:
			matyas_function_base() : problem<T>(2) {};

			virtual ~matyas_function_base() = default;

			double operator()(const T &x) {
				return 0.26 * (x[0] * x[0] + x[1] * x[1]) - 0.48 * x[0] * x[1];
			}


		};

		template<class T>
		class matyas_function : public matyas_function_base<T> {
		public:
			matyas_function() : matyas_function_base<T>() {}

			~matyas_function() = default;

			void gradient(const T &x, T &gr) {
				if (gr.size() != this->mInputDimension) {
					throw std::invalid_argument("gradient size does not match input dimension");
				}
				gr[0] = 0.26 * 2 * x[0] - 0.48 * x[1];
				gr[1] = 0.26 * 2 * x[1] - 0.48 * x[0];
			}
		};
		*/

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
				return NLL;
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
				if (kernel->ARD) { _grad -= lpg; }
				else { _grad.array() -= lpg.coeff(0); }
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
				get_bounds(lower_bound, upper_bound, false);

				TMatrix X(solver->n_restarts, lower_bound.size());
				/* Try Uniform Distribution */
				// if (solver->sampling_method == "uniform") {
				// 	std::mt19937 generator(std::random_device{}());
				// 	for (Eigen::Index c = 0; c < X.cols(); ++c) {
				// 		std::uniform_real_distribution<> distribution(lower_bound[c], upper_bound[c]);
				// 		auto uniform = [&](int, Eigen::Index) {return distribution(generator); };
				// 		X.col(c) = TMatrix::NullaryExpr(solver->n_restarts, 1, uniform);
				// 	}
				// }
				// else if (solver->sampling_method == "sobol") {
				// 	X = statistics::sobol::generate_sobol(solver->n_restarts, lower_bound.size());
				// 	metrics::scale_to_range(X, lower_bound, upper_bound);
				// 	X.array() += 1e-6;
				// }
				// else { std::runtime_error("Unrecognized Sampling Method"); }

				theta = TVector::Constant(1, -1.0);
				if (solver->type == "LBFGSB") {
					std::vector<double> fhistory;
					std::vector<TVector> phistory;
					Objective<TVector> objective(this, static_cast<int>(lower_bound.size()));
					objective.set_lower_bound(lower_bound);
					objective.set_upper_bound(upper_bound);
					l_bfgs_b<TVector> lbfgsb_solver;

					lbfgsb_solver.optimize(objective, theta);

					// for (int i = 0; i < solver->n_restarts; ++i) {
					// 	theta = X.row(i);
					// 	auto [fsol, xsol] = lbfgsb_solver.minimize(objective, theta, objective_value, lower_bound, upper_bound);
					// 	fhistory.push_back(fsol);
					// 	phistory.push_back(xsol);
					// }

					// // Perform Checks on local history
					// int min_idx = std::min_element(fhistory.begin(), fhistory.end()) - fhistory.begin();
					// double min_f = *std::min_element(fhistory.begin(), fhistory.end());
					// objective_value = min_f;
					// theta = phistory[min_idx];

					if (theta.array().isNaN().any() || theta.array().isInf().any()) {
						if (solver->verbosity > 0) { std::cout << "LBFGSB FAILED -> RUNNING PSO" << std::endl; }
						// Better way than to swap pointers?
						shared_ptr<PSO> _solver = make_shared<PSO>(solver->verbosity, solver->n_restarts, solver->sampling_method);
						shared_ptr<Solver> solver2 = std::static_pointer_cast<Solver>(_solver);
						solver.swap(solver2);
						from_optim_(theta, lower_bound, upper_bound, X);
					}
					// set_params(theta);
				}
				// else {from_optim_(theta, lower_bound, upper_bound, X);}
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
				return log_marginal_likelihood();
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
			Parameter<double> scale = { "scale", 1.0, "logexp" };
			double objective_value = 0.0;
			BoolVector missing;

		};

	}


}
#endif