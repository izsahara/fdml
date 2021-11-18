#ifndef DEEPMODELS_H
#define DEEPMODELS_H
// #pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <fdml/utilities.h>

#include <fdml/kernels.h>
#include <fdml/base_models.h>
#include <chrono>

namespace fdml::deep_models {

	namespace gaussian_process {
		using namespace fdml::kernels;
		using namespace fdml::utilities;
		using namespace fdml::base_models;
		using namespace fdml::base_models::gaussian_process;

		class Node : public GP {
		public:
			Node(shared_ptr<Kernel> kernel) : GP(kernel) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver) : GP(kernel, solver) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_) :
				GP(kernel, likelihood_variance), scale("scale", scale_) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_, shared_ptr<Solver> solver) :
				GP(kernel, solver, likelihood_variance), scale("scale", scale_) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale) :
				GP(kernel, likelihood_variance), scale(scale) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale_, shared_ptr<Solver> solver) :
				GP(kernel, solver, likelihood_variance) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}

			double log_marginal_likelihood() override {
				// Compute Log Likelihood [Rasmussen, Eq 2.30]
				double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
				double YKinvY = (outputs.transpose() * alpha)(0);
				double NLL = 0.0;
				if (*scale.is_fixed) { NLL = 0.5 * (logdet + YKinvY); }
				else { NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value()))); }
				NLL -= log_prior();
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
			
			void train() override {
				TVector lower_bound, upper_bound, theta;
				get_bounds(lower_bound, upper_bound, false);
				theta = TVector::Constant(lower_bound.size(), 0.0);

				if (solver->from_optim){
					auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
					{return objective_(x, grad, nullptr, opt_data); };
					opt::OptimData optdata;
					solver->solve(theta, objective, optdata);
				}
				else {
					// LBFGSB
					Objective objective(this, static_cast<int>(lower_bound.size()));
					objective.set_bounds(lower_bound, upper_bound);
					solver->solve(theta, objective);
				}
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
				if (kernel->ARD) {_grad -= lpg;}
				else { _grad.array() -= lpg.coeff(0); }
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

				if (n_thread == 0 || n_thread == 1) {
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
					Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
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
				{
					throw std::runtime_error("No Parameters Saved, set store_parameters = true");
				}
				Eigen::Index param_size = params_size();
				TMatrix _history(history.size(), param_size);
				for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
					_history.row(i) = history[i];
				}
				return _history;
			}

			// Setters (Python Interface)
			void set_inputs(const TMatrix& input) { inputs = input; }
			void set_outputs(const TMatrix& output) { outputs = output; }
			// Getters (Python Interface)
			const TMatrix get_inputs() { return inputs; }
			const TMatrix get_outputs() { return outputs; }

		private:
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
			BoolVector missing;
			double objective_value = 0.0;
			bool store_parameters = false;
			bool connected = false;
			std::vector<TVector> history;

		protected:
			TVector  alpha;
			TLLT	 chol;
			TMatrix	 K;
			TMatrix	 D;
		};

		class Layer {
		private:
			void check_nodes() {
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
				{
					if (node->inputs.size() == 0)
					{
						throw std::runtime_error("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
					}
					if (node->outputs.size() == 0)
					{
						throw std::runtime_error("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
					}
				}
			}
			void estimate_parameters(const Eigen::Index& n_burn) {
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TMatrix history = node->get_parameter_history();
					TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
					node->set_params(theta);
				}
			}
		public:

			Layer(const std::vector<Node>& nodes_, bool initialize = true) : nodes(nodes_) {
				if (initialize) {
					// Checks
					for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
						if (node->inputs.size() > 0)
						{
							throw std::runtime_error("Node has inputs, required empty");
						}
						if (node->outputs.size() > 0)
						{
							throw std::runtime_error("Node has outputs, required empty");
						}
						if (node->kernel == nullptr)
						{
							throw std::runtime_error("Node has no kernel, kernel required");
						}
					}
				}
			}
			Layer& operator()(Layer& layer) {
				// Initialize			[ CurrentLayer(NextLayer) ]
				if (state == 0) {
					layer.index = index + 1;
					if (layer.nodes.size() == nodes.size())
					{
						layer.set_inputs(o_output);
						if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
						{
							layer.set_outputs(o_output, true);
						}
					}
					else if (layer.nodes.size() < nodes.size())
					{
						if (layer.last_layer) { layer.set_inputs(o_output); }
						else {
							// Apply Dimensionality Reduction (Kernel PCA)
							layer.set_inputs(o_output);
							kernelpca::KernelPCA pca(layer.nodes.size(), "sigmoid");
							TMatrix input_transformed = pca.transform(o_output);
							if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
							{
								layer.set_outputs(input_transformed, true);
							}
						}
					}
					else
					{
						//layer.set_inputs(o_output);
						//if (!layer.last_layer) {/* Dimension Expansion*/ }

					}
				}
				// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
				if (state == 2) {
					TMatrix linked_mu = layer.latent_output.first;
					TMatrix linked_var = layer.latent_output.second;
					TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
					TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
					Eigen::Index column = 0;
					for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
						TVector latent_mu = TVector::Zero(linked_mu.rows());
						TVector latent_var = TVector::Zero(linked_mu.rows());
						node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var, n_thread);
						output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
						output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
						column++;
					}
					latent_output.first = output_mean;
					latent_output.second = output_variance;
				}
				return *this;
			}
			std::vector<TMatrix> get_parameter_history() {
				std::vector<TMatrix> history;
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					history.push_back(node->get_parameter_history());
				}
				return history;
			}
			void train() {
				if (state == 0 || state == 2) { check_nodes(); state = 1; }
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					if (!node->store_parameters) { node->store_parameters = true; }
					node->train();
				}
			}
			void predict(const TMatrix& X) {
				/*
				* All nodes (GPR) predict X and output a pair of N X M matrices
				* Where N = number of rows X ; M = number of nodes in layer
				* The pair is the mean and variance.
				*/
				TMatrix node_mu(X.rows(), nodes.size());
				TMatrix node_var(X.rows(), nodes.size());
				for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
				{
					MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
					node_mu.block(0, i, X.rows(), 1) = pred.first;
					node_var.block(0, i, X.rows(), 1) = pred.second;
				}

				latent_output = std::make_pair(node_mu, node_var);
				//latent_output = pred;
			}
			void connect_observed_inputs() { 
				connected = true;
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					node->connected = true;
				}
			}
			
			// Setters
			void set_inputs(const TMatrix& inputs) {
				o_input = inputs;
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					node->inputs = inputs;
				}
			}
			void set_outputs(const TMatrix& outputs, bool latent = false) {
				BoolVector missing;
				if (latent) { missing = BoolVector::Ones(outputs.rows()); }
				else { missing = BoolVector::Zero(outputs.rows()); }
				if ((outputs.array().isNaN()).any())
				{
					missing = operations::get_missing_index<BoolVector>(outputs);
				}

				if (nodes.size())
					for (std::size_t c = 0; c < nodes.size(); ++c) {
						nodes[c].outputs = outputs.col(c);
						nodes[c].missing = missing;
					}
				o_output = outputs;

			}
			// Getters
			TMatrix get_inputs() { return o_input; }
			TMatrix get_outputs() { return o_output; }
			void reconstruct_observed(const TMatrix& inputs, const TMatrix& outputs) {
				o_input = inputs;
				o_output = outputs;
			}
			friend class SIDGP;
		public:
			std::vector<Node> nodes;
			bool last_layer = false;
			bool connected = false;
			int n_thread = 0;
			int state = 0;
			int index = 0;
		private:
			TMatrix o_input;
			TMatrix o_output;
			MatrixPair latent_output;

		};

		class Node2 : public GP {
		public:
			Node2(shared_ptr<Kernel> kernel) : GP(kernel) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node2(shared_ptr<Kernel> kernel, shared_ptr<Solver> solver) : GP(kernel, solver) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node2(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_) :
				GP(kernel, likelihood_variance), scale("scale", scale_) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node2(shared_ptr<Kernel> kernel, const double& likelihood_variance, const double& scale_, shared_ptr<Solver> solver) :
				GP(kernel, solver, likelihood_variance), scale("scale", scale_) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node2(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale) :
				GP(kernel, likelihood_variance), scale(scale) {
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}
			Node2(shared_ptr<Kernel> kernel, const Parameter<double>& likelihood_variance, const Parameter<double>& scale_, shared_ptr<Solver> solver) :
				GP(kernel, solver, likelihood_variance) {
				scale = scale_;
				if (kernel->variance.value() != 1.0) { kernel->variance = 1.0; }
				if (!kernel->variance.fixed()) { kernel->variance.fix(); }
			}

			double log_marginal_likelihood() override {
				// Compute Log Likelihood [Rasmussen, Eq 2.30]
				double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
				double YKinvY = (outputs.transpose() * alpha)(0);
				double NLL = 0.0;
				if (*scale.is_fixed) { NLL = 0.5 * (logdet + YKinvY); }
				else { NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value()))); }
				NLL -= log_prior();
				return NLL;
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

			
			void train() override {
				TVector lower_bound, upper_bound, theta;
				get_bounds(lower_bound, upper_bound, false);
				theta = TVector::Constant(lower_bound.size(), 0.0);

				if (solver->from_optim){
					auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
					{return objective_(x, grad, nullptr, opt_data); };
					opt::OptimData optdata;
					solver->solve(theta, objective, optdata);
				}
				else {
					// LBFGSB
					Objective objective(this, static_cast<int>(lower_bound.size()));
					objective.set_bounds(lower_bound, upper_bound);
					solver->solve(theta, objective);
				}
			}
			TVector gradients() override {
				std::vector<TMatrix> grad_;
				if (alpha.size() == 0) { update_cholesky(); }
				// Kernel Derivatives
				kernel->fod(inputs, grad_);
				TVector grad(grad_.size());
				for (int i = 0; i < grad_.size(); ++i) {
					TMatrix KKT = chol.solve(grad_[i]);
					double trace = KKT.trace();
					double YKKT = (outputs.transpose() * KKT * alpha).coeff(0);
					double P1 = -0.5 * trace;
					double P2 = 0.5 * YKKT;
					if (!(*scale.is_fixed)) {
						grad[i] = -P1 - (P2 / scale.value());
					}
					else {
						grad[i] = -P1 - P2;
					}
				}
				grad -= log_prior_gradient();
				return grad;
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

				if (n_thread == 0 || n_thread == 1) {
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
					Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
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
				{
					throw std::runtime_error("No Parameters Saved, set store_parameters = true");
				}
				Eigen::Index param_size = params_size();
				TMatrix _history(history.size(), param_size);
				for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
					_history.row(i) = history[i];
				}
				return _history;
			}

			// Setters (Python Interface)
			void set_inputs(const TMatrix& input) { inputs = input; }
			void set_outputs(const TMatrix& output) { outputs = output; }
			// Getters (Python Interface)
			const TMatrix get_inputs() { return inputs; }
			const TMatrix get_outputs() { return outputs; }

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

		private:
			double objective_(const TVector& x, TVector* grad, TVector* hess, void* opt_data) {
				set_params(x);
				if (grad) { (*grad) = gradients() * -1.0; }
				return -log_marginal_likelihood();
			}

		protected:
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
			BoolVector missing;
			double objective_value = 0.0;
			bool store_parameters = false;
			bool connected = false;
			std::vector<TVector> history;

		protected:
			TVector  alpha;
			TLLT	 chol;
			TMatrix	 K;
			TMatrix	 D;
		};

		class Layer2 {
		private:
			void check_nodes() {
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node)
				{
					if (node->inputs.size() == 0)
					{
						throw std::runtime_error("A Node in the Layer2 has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
					}
					if (node->outputs.size() == 0)
					{
						throw std::runtime_error("A Node in the Layer2 has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
					}
				}
			}
			void estimate_parameters(const Eigen::Index& n_burn) {
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TMatrix history = node->get_parameter_history();
					TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
					node->set_params(theta);
				}
			}
		public:

			Layer2(const std::vector<Node2>& nodes_, bool initialize = true) : nodes(nodes_) {
				if (initialize) {
					// Checks
					for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
						if (node->inputs.size() > 0)
						{
							throw std::runtime_error("Node has inputs, required empty");
						}
						if (node->outputs.size() > 0)
						{
							throw std::runtime_error("Node has outputs, required empty");
						}
						if (node->kernel == nullptr)
						{
							throw std::runtime_error("Node has no kernel, kernel required");
						}
					}
				}
			}
			Layer2& operator()(Layer2& layer) {
				// Initialize			[ CurrentLayer(NextLayer) ]
				if (state == 0) {
					layer.index = index + 1;
					if (layer.nodes.size() == nodes.size())
					{
						layer.set_inputs(o_output);
						if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
						{
							layer.set_outputs(o_output, true);
						}
					}
					else if (layer.nodes.size() < nodes.size())
					{
						if (layer.last_layer) { layer.set_inputs(o_output); }
						else {
							// Apply Dimensionality Reduction (Kernel PCA)
							layer.set_inputs(o_output);
							kernelpca::KernelPCA pca(layer.nodes.size(), "sigmoid");
							TMatrix input_transformed = pca.transform(o_output);
							if (layer.o_output.size() == 0 || layer.o_output.rows() != o_output.rows())
							{
								layer.set_outputs(input_transformed, true);
							}
						}
					}
					else
					{
						//layer.set_inputs(o_output);
						//if (!layer.last_layer) {/* Dimension Expansion*/ }

					}
				}
				// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
				if (state == 2) {
					TMatrix linked_mu = layer.latent_output.first;
					TMatrix linked_var = layer.latent_output.second;
					TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
					TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
					Eigen::Index column = 0;
					for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
						TVector latent_mu = TVector::Zero(linked_mu.rows());
						TVector latent_var = TVector::Zero(linked_mu.rows());
						node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var, n_thread);
						output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
						output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
						column++;
					}
					latent_output.first = output_mean;
					latent_output.second = output_variance;
				}
				return *this;
			}
			std::vector<TMatrix> get_parameter_history() {
				std::vector<TMatrix> history;
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					history.push_back(node->get_parameter_history());
				}
				return history;
			}
			void train() {
				if (state == 0 || state == 2) { check_nodes(); state = 1; }
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					if (!node->store_parameters) { node->store_parameters = true; }
					node->train();
				}
			}
			void predict(const TMatrix& X) {
				/*
				* All nodes (GPR) predict X and output a pair of N X M matrices
				* Where N = number of rows X ; M = number of nodes in layer
				* The pair is the mean and variance.
				*/
				TMatrix node_mu(X.rows(), nodes.size());
				TMatrix node_var(X.rows(), nodes.size());
				for (std::vector<Node2>::size_type i = 0; i != nodes.size(); ++i)
				{
					MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
					node_mu.block(0, i, X.rows(), 1) = pred.first;
					node_var.block(0, i, X.rows(), 1) = pred.second;
				}

				latent_output = std::make_pair(node_mu, node_var);
				//latent_output = pred;
			}
			void connect_observed_inputs() {
				connected = true;
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					node->connected = true;
				}
			}

			// Setters
			void set_inputs(const TMatrix& inputs) {
				o_input = inputs;
				for (std::vector<Node2>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					node->inputs = inputs;
				}
			}
			void set_outputs(const TMatrix& outputs, bool latent = false) {
				BoolVector missing;
				if (latent) { missing = BoolVector::Ones(outputs.rows()); }
				else { missing = BoolVector::Zero(outputs.rows()); }
				if ((outputs.array().isNaN()).any())
				{
					missing = operations::get_missing_index<BoolVector>(outputs);
				}

				if (nodes.size())
					for (std::size_t c = 0; c < nodes.size(); ++c) {
						nodes[c].outputs = outputs.col(c);
						nodes[c].missing = missing;
					}
				o_output = outputs;

			}
			// Getters
			TMatrix get_inputs() { return o_input; }
			TMatrix get_outputs() { return o_output; }
			void reconstruct_observed(const TMatrix& inputs, const TMatrix& outputs) {
				o_input = inputs;
				o_output = outputs;
			}
			friend class SIDGP2;
		public:
			std::vector<Node2> nodes;
			bool last_layer = false;
			bool connected = false;
			int n_thread = 0;
			int state = 0;
			int index = 0;
		private:
			TMatrix o_input;
			TMatrix o_output;
			MatrixPair latent_output;
			int verbosity = 1;

		};



		class Imputer {
			TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
				return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
			}
			TMatrix sample_mvn(const TMatrix& K) {
				statistics::MVN sampler(K);
				return sampler();
			}
			double log_likelihood_(const TMatrix& K, const TMatrix& outputs) {
				TLLT chol(K);
				double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
				double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
				double lml = -0.5 * (logdet + quad);
				return lml;
			}
			void conditional_mvn(Node& node, TMatrix& mu, TMatrix& var) {
				TMatrix X1 = operations::mask_matrix(node.inputs, node.missing, false, 0);
				TMatrix W1 = operations::mask_matrix(node.inputs, node.missing, true, 0);
				TMatrix W2 = operations::mask_matrix(node.outputs, node.missing, true, 0);
				TMatrix R = node.kernel->K(W1, W1);
				TMatrix c = node.kernel->K(X1, X1, node.likelihood_variance.value());
				TMatrix r = node.kernel->K(W1, X1);
				TLLT chol = R.llt();
				TMatrix alpha = chol.solve(r); // Rinv_r = np.linalg.solve(R, r)
				TMatrix beta = (r.transpose() * alpha); // r_Rinv_r = r.T @ Rinv_r
				TMatrix tmp(alpha.rows(), alpha.cols());
				operations::visit_lambda(alpha, [&tmp, &W2](double v, int i, int j) { tmp(i, j) = W2(i) * v; });
				mu.resize(alpha.rows(), 1);
				mu = tmp.colwise().sum().transpose();
				var = (node.kernel->variance.value() * (c - beta)).cwiseAbs();
			}
		public:
			void ess_update(Node& target, Layer& linked, const std::size_t& node_idx) {
				/* Elliptical Slice Sampling Update (Algorithm 1)
				* Nishihara, R., Murray, I. & Adams, R. P. (2014), �Parallel MCMC with generalized elliptical slice sampling�,
				* The Journal of Machine Learning Research 15(1), 2087�2112.
				*/

				auto log_likelihood = [](const TMatrix& K, const TMatrix& outputs)
				{
					TLLT chol(K);
					double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
					double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
					double lml = -0.5 * (logdet + quad);
					return lml;
				};

				if (target.missing.count() == 0) { return; }
				TMatrix K = target.kernel->K(target.inputs, target.inputs, target.likelihood_variance.value());
				K *= target.scale.value();
				TMatrix nu = sample_mvn(K);
				double log_y = 0.0;

				for (std::vector<Node>::iterator node = linked.nodes.begin(); node != linked.nodes.end(); ++node) {
					const TMatrix W = node->inputs;
					const TMatrix Y = node->outputs;
					TMatrix Kw = node->kernel->K(W, W, node->likelihood_variance.value());
					Kw *= node->scale.value();
					log_y += log_likelihood(Kw, Y);
				}
				log_y += log(statistics::random_uniform());
				double theta = statistics::random_uniform(0.0, 2.0 * PI);
				double theta_min = theta - 2.0 * PI;
				double theta_max = theta;

				TVector mean = TVector::Zero(target.inputs.rows());
				while (true) {
					TMatrix fp = update_f(target.outputs, nu, mean, theta);
					double log_yp = 0.0;

					for (std::size_t n = 0; n < linked.nodes.size(); ++n) {
						linked.nodes[n].inputs.col(node_idx) = fp;
						const TMatrix W2 = linked.nodes[n].inputs;
						const TMatrix Y2 = linked.nodes[n].outputs;
						TMatrix Kw2 = linked.nodes[n].kernel->K(W2, W2, linked.nodes[n].likelihood_variance.value());
						Kw2 *= linked.nodes[n].scale.value();
						log_yp += log_likelihood(Kw2, Y2);
					}
					// DEBUG
					//std::cout << "log_yp = " << log_yp << " " << "log_y" << log_y << std::endl;
					//
					if (log_yp > log_y) { target.outputs = fp; return; }
					else {

						if (theta < 0) { theta_min = theta; }
						else { theta_max = theta; }
						theta = statistics::random_uniform(theta_min, theta_max);
					}
				}
			}
			void ess_update(Node2& target, Layer2& linked, const std::size_t& node_idx) {
				/* Elliptical Slice Sampling Update (Algorithm 1)
				* Nishihara, R., Murray, I. & Adams, R. P. (2014), �Parallel MCMC with generalized elliptical slice sampling�,
				* The Journal of Machine Learning Research 15(1), 2087�2112.
				*/

				auto log_likelihood = [](const TMatrix& K, const TMatrix& outputs)
				{
					TLLT chol(K);
					double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
					double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
					double lml = -0.5 * (logdet + quad);
					return lml;
				};

				if (target.missing.count() == 0) { return; }
				TMatrix K = target.kernel->K(target.inputs, target.inputs, target.likelihood_variance.value());
				K *= target.scale.value();
				TMatrix nu = sample_mvn(K);
				double log_y = 0.0;

				for (std::vector<Node2>::iterator node = linked.nodes.begin(); node != linked.nodes.end(); ++node) {
					const TMatrix W = node->inputs;
					const TMatrix Y = node->outputs;
					TMatrix Kw = node->kernel->K(W, W, node->likelihood_variance.value());
					Kw *= node->scale.value();
					log_y += log_likelihood(Kw, Y);
				}
				log_y += log(statistics::random_uniform());
				double theta = statistics::random_uniform(0.0, 2.0 * PI);
				double theta_min = theta - 2.0 * PI;
				double theta_max = theta;

				TVector mean = TVector::Zero(target.inputs.rows());
				while (true) {
					TMatrix fp = update_f(target.outputs, nu, mean, theta);
					double log_yp = 0.0;

					for (std::size_t n = 0; n < linked.nodes.size(); ++n) {
						linked.nodes[n].inputs.col(node_idx) = fp;
						const TMatrix W2 = linked.nodes[n].inputs;
						const TMatrix Y2 = linked.nodes[n].outputs;
						TMatrix Kw2 = linked.nodes[n].kernel->K(W2, W2, linked.nodes[n].likelihood_variance.value());
						Kw2 *= linked.nodes[n].scale.value();
						log_yp += log_likelihood(Kw2, Y2);
					}
					// DEBUG
					//std::cout << "log_yp = " << log_yp << " " << "log_y" << log_y << std::endl;
					//
					if (log_yp > log_y) { target.outputs = fp; return; }
					else {

						if (theta < 0) { theta_min = theta; }
						else { theta_max = theta; }
						theta = statistics::random_uniform(theta_min, theta_max);
					}
				}
			}
						
		};

		class SIDGP : public Imputer {

		private:
			void initialize_layers() {
				if (layers.front().o_input.size() == 0) { throw std::runtime_error("First Layer Requires Observed Inputs"); }
				if (layers.back().o_output.size() == 0) { throw std::runtime_error("Last Layer Requires Observed Outputs"); }
				//if (layers.back().nodes.size() != 1) { throw std::runtime_error("Last Layer Must Only have 1 Node for a Single Output"); }
				layers.front().index = 1;
				layers.back().last_layer = true;
				// Propagate First Layer
				TMatrix X = layers.front().get_inputs();
				layers.front().set_outputs(X, true);

				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
					layer->state = 0;
					(*layer)(*std::next(layer));
				}
			}
			void sample(int n_burn = 10) {
				for (int i = 0; i < n_burn; ++i) {
					// DEBUG
					//std::cout << "iter = " << i << std::endl;
					//
					for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
						for (std::size_t n = 0; n < layer->nodes.size(); ++n) {
							ess_update(layer->nodes[n], *std::next(layer), n);
						}
					}
				}
			}
		public:
			SIDGP(const std::vector<Layer>& layers, bool initialize = true) : layers(layers) {
				if (initialize) {
					initialize_layers();
					sample(10);
				}
			}

			void train(int n_iter = 50, int ess_burn = 10) {
				auto train_start = std::chrono::system_clock::now();
				std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);				
				auto verbose = [this, &train_start_t, &n_iter](const int& n, const int& i, const double& p) {
					if (verbosity == 1) {
						if (n == 0 && i == 1) {
							std::cout << "TRAIN START: " <<
							std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
						}
						std::cout << std::setw(3) << std::left <<
							std::setprecision(1) << std::fixed << p <<
							std::setw(5) << std::left << " % |";
						std::cout << std::setw(7) <<
							std::left << " LAYER " << std::setw(3) <<
							std::left << i << "\r" << std::flush;
						if (n == n_iter - 1 && i == layers.size()) {
							auto train_end = std::chrono::system_clock::now();
							std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
							std::cout << "TRAIN END: " <<
							std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
						}
					}
				};
				n_iter_ = n_iter;
				for (int i = 0; i < n_iter; ++i) {
					double progress = double(i+1) * 100.0 / double(n_iter);
					// I-step
					sample(ess_burn);
					// M-step
					for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
						layer->train();
						verbose(i, layer->index, progress);
					}
				}
				//std::system("cls");
				std::cout << std::endl;
			}
			void estimate(Eigen::Index n_burn = 0) {
				if (n_burn == 0) { n_burn = std::size_t(0.75 * n_iter_); }
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
					layer->estimate_parameters(n_burn);
				}
			}
			MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 1) {
				sample(50);
				const std::size_t n_layers = layers.size();
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				auto verbose = [this, &pred_start_t, &n_impute](const int& i, const double& p) {
					if (verbosity == 1) {
						if (i == 0) {
							std::cout << "PREDICTION START: " <<
							std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
						}
						std::cout << std::setw(3) << std::left <<
							std::setprecision(1) << std::fixed << p << std::setw(5) << std::left << " % |";
						std::cout << std::setw(7) <<
							std::left << "N_IMPUTE" << std::setw(1) << std::left << "" << i << "\r" << std::flush;
						if (i == n_impute - 1) {
							auto pred_end = std::chrono::system_clock::now();
							std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
							std::cout << "PREDICTION END: " <<
							std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
						}
					}
				};
				for (int i = 0; i < n_impute; ++i) {
					double progress = double(i+1) * 100.0 / double(n_impute);
					sample();
					layers.front().predict(X);
					std::size_t j = 1;
					for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
						if (layer->state != 2) { layer->state = 2; }
						layer->n_thread = n_thread;
						(*layer)(*std::prev(layer));
						j++;
					}
					if (i == 0) {
						mean = layers.back().latent_output.first;
						variance = square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second;
					}
					else {
						mean.noalias() += layers.back().latent_output.first;
						variance.noalias() += (square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second);
					}	
					verbose(i, progress);
				}
				std::cout << std::endl;
				mean.array() /= double(n_impute);
				variance.array() /= double(n_impute);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}
			
			void set_observed(const TMatrix& X, const TMatrix& Z) {				
				if (X.size() != layers.front().o_input.size()) {
					layers.front().set_inputs(X);
					layers.back().set_outputs(Z);
					initialize_layers();
				}
				else {
					layers.front().set_inputs(X);
					layers.back().set_outputs(Z);
				}
			}

			const std::vector<Layer> get_layers() const { return layers; }

			void set_n_iter(const int& n) { n_iter_ = n; }
			const int n_iterations() const { return n_iter_; }
			const std::string model_type() const { return "SIDGP"; }

		private:
			int n_iter_ = 0;
			std::vector<Layer> layers;
		public:
			int verbosity = 1;
		};

		class SIDGP2 : public Imputer {

		private:
			void initialize_layers() {
				if (layers.front().o_input.size() == 0) { throw std::runtime_error("First Layer Requires Observed Inputs"); }
				if (layers.back().o_output.size() == 0) { throw std::runtime_error("Last Layer Requires Observed Outputs"); }
				//if (layers.back().nodes.size() != 1) { throw std::runtime_error("Last Layer Must Only have 1 Node for a Single Output"); }
				layers.front().index = 1;
				layers.back().last_layer = true;
				// Propagate First Layer
				TMatrix X = layers.front().get_inputs();
				layers.front().set_outputs(X, true);

				for (std::vector<Layer2>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
					layer->state = 0;
					(*layer)(*std::next(layer));
				}
			}
			void sample(int n_burn = 10) {
				for (int i = 0; i < n_burn; ++i) {
					// DEBUG
					//std::cout << "iter = " << i << std::endl;
					//
					for (std::vector<Layer2>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
						for (std::size_t n = 0; n < layer->nodes.size(); ++n) {
							ess_update(layer->nodes[n], *std::next(layer), n);
						}
					}
				}
			}
		public:
			SIDGP2(const std::vector<Layer2>& layers, bool initialize = true) : layers(layers) {
				if (initialize) {
					initialize_layers();
					sample(10);
				}
			}

			void train(int n_iter = 50, int ess_burn = 10) {
				auto train_start = std::chrono::system_clock::now();
				std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);
				auto verbose = [this, &train_start_t, &n_iter](const int& n, const int& i, const double& p) {
					if (verbosity == 1) {
						if (n == 0 && i == 1) {
							std::cout << "TRAIN START: " <<
								std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
						}
						std::cout << std::setw(3) << std::left <<
							std::setprecision(1) << std::fixed << p <<
							std::setw(5) << std::left << " % |";
						std::cout << std::setw(7) <<
							std::left << " LAYER " << std::setw(3) <<
							std::left << i << "\r" << std::flush;
						if (n == n_iter - 1 && i == layers.size()) {
							auto train_end = std::chrono::system_clock::now();
							std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
							std::cout << "TRAIN END: " <<
								std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
						}
					}
				};
				n_iter_ = n_iter;
				for (int i = 0; i < n_iter; ++i) {
					double progress = double(i + 1) * 100.0 / double(n_iter);
					// I-step
					sample(ess_burn);
					// M-step
					for (std::vector<Layer2>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
						layer->train();
						verbose(i, layer->index, progress);
					}
				}
				//std::system("cls");
				std::cout << std::endl;
			}
			void estimate(Eigen::Index n_burn = 0) {
				if (n_burn == 0) { n_burn = std::size_t(0.75 * n_iter_); }
				for (std::vector<Layer2>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
					layer->estimate_parameters(n_burn);
				}
			}
			MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 1) {
				sample(50);
				const std::size_t n_layers = layers.size();
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				auto verbose = [this, &pred_start_t, &n_impute](const int& i, const double& p) {
					if (verbosity == 1) {
						if (i == 0) {
							std::cout << "PREDICTION START: " <<
								std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
						}
						std::cout << std::setw(3) << std::left <<
							std::setprecision(1) << std::fixed << p << std::setw(5) << std::left << " % |";
						std::cout << std::setw(7) <<
							std::left << "N_IMPUTE" << std::setw(1) << std::left << "" << i << "\r" << std::flush;
						if (i == n_impute - 1) {
							auto pred_end = std::chrono::system_clock::now();
							std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
							std::cout << "PREDICTION END: " <<
								std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
						}
					}
				};
				for (int i = 0; i < n_impute; ++i) {
					double progress = double(i) * 100.0 / double(n_impute);
					sample();
					layers.front().predict(X);
					std::size_t j = 1;
					for (std::vector<Layer2>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
						if (layer->state != 2) { layer->state = 2; }
						layer->n_thread = n_thread;
						(*layer)(*std::prev(layer));
						j++;
					}
					if (i == 0) {
						mean = layers.back().latent_output.first;
						variance = square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second;
					}
					else {
						mean.noalias() += layers.back().latent_output.first;
						variance.noalias() += (square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second);
					}
					verbose(i, progress);
				}
				std::cout << std::endl;
				mean.array() /= double(n_impute);
				variance.array() /= double(n_impute);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}

			MatrixPair predict(const TMatrix& X, TMatrix& Yref, int n_impute = 50, int n_thread = 1) {
				sample(50);
				const std::size_t n_layers = layers.size();
				TMatrix mean = TMatrix::Zero(X.rows(), 1);
				TMatrix variance = TMatrix::Zero(X.rows(), 1);
				std::vector<MatrixPair> predictions;

				auto pred_start = std::chrono::system_clock::now();
				std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
				auto verbose = [this, &pred_start_t, &n_impute, &Yref](const int& i, const double& p, const TVector& mu) {
					if (verbosity == 1) {
						if (i == 0) {
							std::cout << "PREDICTION START: " <<
								std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
						}
						std::cout << std::setw(3) << std::left <<
							std::setprecision(1) << std::fixed << p << std::setw(5) << std::left << " % |";
						std::cout << std::setw(7) <<
							std::left << "N_IMPUTE" << std::setw(1) << std::left << "" << i << "  " <<
							std::setw(7) << std::left << "NRMSE = " << 
							std::setw(3) << std::left << std::setprecision(5) << std::fixed <<
							metrics::rmse(Yref, mu) / (Yref.maxCoeff() - Yref.minCoeff()) << "\r" << std::flush;
						if (i == n_impute - 1) {
							auto pred_end = std::chrono::system_clock::now();
							std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
							std::cout << "PREDICTION END: " <<
								std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
						}
					}
				};
				for (int i = 0; i < n_impute; ++i) {
					double progress = double(i) * 100.0 / double(n_impute);
					sample();
					layers.front().predict(X);
					std::size_t j = 1;
					for (std::vector<Layer2>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
						if (layer->state != 2) { layer->state = 2; }
						layer->n_thread = n_thread;
						(*layer)(*std::prev(layer));
						j++;
					}
					if (i == 0) {
						mean = layers.back().latent_output.first;
						variance = square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second;
					}
					else {
						mean.noalias() += layers.back().latent_output.first;
						variance.noalias() += (square(layers.back().latent_output.first.array()).matrix() + layers.back().latent_output.second);
					}
					TVector tmp_mu = mean.array() / double(i);
					verbose(i, progress, tmp_mu);
				}
				std::cout << std::endl;
				mean.array() /= double(n_impute);
				variance.array() /= double(n_impute);
				variance.array() -= square(mean.array());

				return std::make_pair(mean, variance);
			}			

			void set_observed(const TMatrix& X, const TMatrix& Z) {
				if (X.size() != layers.front().o_input.size()) {
					layers.front().set_inputs(X);
					layers.back().set_outputs(Z);
					initialize_layers();
				}
				else {
					layers.front().set_inputs(X);
					layers.back().set_outputs(Z);
				}
			}

			const std::vector<Layer2> get_layers() const { return layers; }

			void set_n_iter(const int& n) { n_iter_ = n; }
			const int n_iterations() const { return n_iter_; }
			const std::string model_type() const { return "SIDGP"; }

		private:
			int n_iter_ = 0;
			std::vector<Layer2> layers;
		public:
			int verbosity = 1;
		};




	}



}
#endif