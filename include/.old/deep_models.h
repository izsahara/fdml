#ifndef DEEPMODELS_H
#define DEEPMODELS_H

#include "./utilities.h"
#include "./kernels.h"
#include "./base_models.h"
#include <filesystem>
#include <highfive/H5Easy.hpp>

namespace SMh::deep_models {}

namespace SMh::deep_models::gaussian_process2 {
	using namespace SMh::kernels;
	using namespace SMh::utilities;
	using namespace SMh::base_models::gaussian_process;

	class Layer {
	private:
		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
		}
		void check_nodes() {
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				if (node->inputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
				}
				if (node->outputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
				}
			}
		}
	public:

		Layer(std::vector<GPNode>& nodes_) {
			// Checks
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				if (node->inputs.size() > 0) 
				{ throw std::exception("Node has inputs, required empty"); }
				if (node->outputs.size() > 0) 
				{ throw std::exception("Node has outputs, required empty"); }
				if (node->kernel == nullptr) 
				{ throw std::exception("Node has no kernel, kernel required"); }
			}
			nodes = nodes_;
		}
		// Operator() overload
		Layer& operator()(Layer& layer) {
			// Initialize			[ CurrentLayer(NextLayer) ]
			if (state_ == 0) {
				layer.index = index + 1;
				// Inputs
				if (layer.nodes.size() == nodes.size())
				{
					layer.set_inputs(o_output);
				}
				else if (layer.nodes.size() < nodes.size())
				{
					// Apply Dimensionality Reduction (Kernel PCA)
				}
				else
				{
					// Dimension Expansion
				}
				// Outputs
				if (layer.o_output.size() == 0)
				{
					layer.set_outputs(o_output, true);
				}
			}
			// Sample				[ CurrentLayer(NextLayer) ]
			if (state_ == 1) {
				double log_y = layer.log_likelihood();
				log_y += log(random_uniform(0.0, 1.0));
				TVector mean = TVector::Zero(o_input.rows());
				for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					if (!(node->missing.count() > 0)) { continue; }
					TMatrix nu = node->sample_mvn();
					double theta = random_uniform(0.0, 2.0 * PI);
					double theta_min = theta - 2.0 * PI;
					double theta_max = theta;

					while (true) {
						TMatrix fp = update_f(node->outputs, nu, mean, theta);
						layer.set_inputs(fp);
						double log_yp = layer.log_likelihood();
						if (log_yp > log_y) { node->outputs = fp; break; }
						else {
							if (theta < 0) { theta_min = theta; }
							else { theta_max = theta; }
							theta = random_uniform(theta_min, theta_max);
						}
					}
				}
			}
			// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
			if (state_ == 2) {
				TMatrix linked_mu = layer.latent_output.first;
				TMatrix linked_var = layer.latent_output.second;
				TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
				TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
				int column = 0;
				for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TVector latent_mu = TVector::Zero(linked_mu.rows());
					TVector latent_var = TVector::Zero(linked_mu.rows());
					node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var);
					output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
					output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
					column++;
				}
				latent_output.first = output_mean;
				latent_output.second = output_variance;
			}
			return *this;
		}

		// Main Functions
		double log_likelihood() {
			double ll = 0;
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				ll += node->log_likelihood();
			}
			return ll;
		}
		void estimate_parameters(Eigen::Index n_burn) {
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				TMatrix history = node->get_parameter_history();
				TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
				node->set_params(theta);
			}
		}
		void train() {
			if (state_ == 0 || state_ == 2) { check_nodes(); state_ = 1; }
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
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
			//TMatrix node_mu(X.rows(), nodes.size());
			//TMatrix node_var(X.rows(), nodes.size());
			MatrixPair pred;
			for (std::vector<GPNode>::size_type i = 0; i != nodes.size(); ++i)
			{
				pred = std::get<MatrixPair>(nodes[i].predict(X, true));
				//node_mu.block(0, i, X.rows(), 1) = pred.first;
				//node_var.block(0, i, X.rows(), 1) = pred.second;
			}

			//latent_output = std::make_pair(node_mu, node_var);
			latent_output = pred;
		}
		
		// Setters
		void set_inputs(const TMatrix& inputs) {
			o_input = inputs;
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->inputs = inputs;
			}
		}
		void set_outputs(const TMatrix& outputs, bool latent = false) {
			BoolVector missing;
			if (latent) { missing = BoolVector::Ones(outputs.rows()); }
			else { missing = BoolVector::Zero(outputs.rows()); }
			if ((outputs.array().isNaN()).any())
			{
				missing = get_missing_index<BoolVector>(outputs);
			}
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->outputs = outputs;
				node->missing = missing;
			}
			o_output = outputs;

		}
		// Getters
		TMatrix get_inputs() { return o_input; }
		TMatrix get_outputs() { return o_output; }

		// Read Only
		const std::string state() const {
			if (state_ == 0) { return "initialize"; }
			if (state_ == 1) { return "training"; }
			if (state_ == 2) { return "predict"; }
		}
		// Expose to Python
		const MatrixPair latent_output_() const { return latent_output; }

		friend class SIDGP;

	public:
		std::vector<GPNode> nodes;
		int index = 0;
	private:
		TMatrix o_input;
		TMatrix o_output;
		MatrixPair latent_output;
		int state_ = 0;

	};

	class SIDGP {

	private:
		#pragma region PRINT_UTILITIES
		void print_utility(int arg, int idx) {
			if (arg == 1) {
				std::cout << std::setw(3) << std::left << idx + 1 << std::setw(2) << std::right << " : ";
			}
		}
		void print_utility(std::size_t idx) {
			if (idx == 0) { std::cout << "LAYER " << idx + 1 << " -> " << "LAYER " << idx + 2 << " -> "; }
			else if (idx == layers.size() - 2) { std::cout << "LAYER " << layers.size() << "\r" << std::flush; }
			else { std::cout << "LAYER " << idx + 2 << " -> " << "\r" << std::flush;}
		}
		void print_utility(int idx, double& progress, double obj_fxn) {
			std::cout << std::setw(3) << std::left << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER "
				<< std::setw(3) << std::left << idx
				<< std::setw(10) << std::right << std::setprecision(4) << std::fixed << obj_fxn << "\r" << std::flush;
		}
		void print_utility(int idx, double& progress) {
			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER " << std::setw(3) << std::left << idx << "\r" << std::flush;
		}
		#pragma endregion
		void initialize_layers() {
			if (layers.front().o_input.size() == 0) { throw std::exception("First Layer Requires Observed Inputs"); }
			if (layers.back().o_output.size() == 0) { throw std::exception("Last Layer Requires Observed Outputs"); }
			layers.front().index = 1;
			// Propagate First Layer
			TMatrix X = layers.front().get_inputs();
			layers.front().set_outputs(X, true);

			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
				layer->state_ = 0;
				(*layer)(*std::next(layer));
			}
		}
		void sample(int n_burn = 10) {
			for (int i = 0; i < n_burn; ++i) {
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
					if (layer->state_ != 1) { layer->state_ = 1; }
					(*layer)(*std::next(layer));
				}
				clear_cholesky();
			}
		}
		void clear_cholesky() {
			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer)
			{
				for (std::vector<GPNode>::iterator node = layer->nodes.begin(); node != layer->nodes.end(); ++node) {
					node->clear_cholesky();
				}
			}
		}
	
	public:

		SIDGP(std::vector<Layer>& layers_) : layers(layers_) {
			//layers.insert(layers.end(),
			//	std::make_move_iterator(layers_.begin()),
			//	std::make_move_iterator(layers_.end()));
			//layers_.erase(layers_.begin(), layers_.end());
			initialize_layers();
			sample(10);
		}

		void train(int n_iter = 500, int ess_burn = 10) {
			train_iter = n_iter;
			for (int i = 0; i < n_iter + 1; ++i) {
				double progress = double(i) * 100.0 / double(n_iter);
				// I-step
				sample(ess_burn);
				// M-step
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
					layer->train();
					print_utility(layer->index, progress);
				}
			}
			// Estimate Point Parameters and Assign to Nodes
			//for (std::vector<Layer>::size_type j = 0; j != layers.size(); ++j) {
			//	layers[j].estimate_parameters();
			//}
		}

		void estimate(Eigen::Index n_burn = 0) {
			if (n_burn == 0) { n_burn = std::size_t(0.75 * train_iter); }
			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
				layer->estimate_parameters(n_burn);
			}
		}

		MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
			// Clear K, chol and alpha in all layer/nodes
			sample(50);
			const std::size_t n_layers = layers.size();
			TMatrix mean = TMatrix::Zero(X.rows(), 1);
			TMatrix variance = TMatrix::Zero(X.rows(), 1);
			std::vector<MatrixPair> predictions;
			for (int i = 0; i < n_impute; ++i) {
				print_utility(1, i);
				sample();
				layers.front().predict(X);
				std::size_t j = 1;
				for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
					print_utility(j);
					if (layer->state_ != 2) { layer->state_ = 2; }
					(*layer)(*std::prev(layer));
					j++;
				}
				//std::system("cls");
				predictions.push_back(layers.back().latent_output);
			}
			std::cout << std::endl;

			TMatrix output_mean = TMatrix::Zero(X.rows(), n_impute);
			TMatrix output_var = TMatrix::Zero(X.rows(), n_impute);
			for (int k = 0; k < predictions.size(); ++k)
			{
				output_mean.block(0, k, X.rows(), 1) = predictions[k].first;
				output_var.block(0, k, X.rows(), 1) = predictions[k].second;
			}

			//std::string path = "E:/23620029-Faiz/C/SMh/cpp_interface/tests/output_mean.dat";
			//save_data(path, output_mean);

			mean = output_mean.rowwise().mean();
			variance = (square((output_mean).array()).matrix() + (output_var)).rowwise().mean();
			variance.array() -= square(mean.array());
			std::system("cls");
			return std::make_pair(mean, variance);
		}

		const int train_iter_() const { return train_iter; }
		const std::vector<Layer> layers_() const { return layers; }
		std::vector<Layer> layers;

	private:
		int train_iter = 0;
	};


}

namespace SMh::deep_models::gaussian_process {
	using namespace SMh::kernels;
	using namespace SMh::utilities;
	using namespace SMh::base_models::gaussian_process;

	class Layer {
	private:
		void check_nodes() {
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				if (node->inputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
				}
				if (node->outputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
				}
			}
		}	
		void estimate_parameters(const Eigen::Index& n_burn, const bool& partitioned) {
			if (partitioned) {
				for (std::size_t p = 0; p < partitioned_history.size(); ++p)
				{
					TMatrix history = partitioned_history[p];
					TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
					nodes[p].set_params(theta);
				}
			}
			else {
				for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TMatrix history = node->get_parameter_history();
					TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
					node->set_params(theta);
				}
			}
		}
	public:

		Layer(std::vector<GPNode>& nodes_) {
			// Checks
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				if (node->inputs.size() > 0)
				{
					throw std::exception("Node has inputs, required empty");
				}
				if (node->outputs.size() > 0)
				{
					throw std::exception("Node has outputs, required empty");
				}
				if (node->kernel == nullptr)
				{
					throw std::exception("Node has no kernel, kernel required");
				}
			}
			nodes = nodes_;
		}
		// Operator() overload
		Layer& operator()(Layer& layer) {
			// Initialize			[ CurrentLayer(NextLayer) ]
			if (state_ == 0) {
				layer.index = index + 1;
				// Inputs
				if (layer.nodes.size() == nodes.size())
				{
					layer.set_inputs(o_output);
				}
				else if (layer.nodes.size() < nodes.size())
				{
					// Apply Dimensionality Reduction (Kernel PCA)
				}
				else
				{
					// Dimension Expansion
				}
				// Outputs
				if (layer.o_output.size() == 0)
				{
					layer.set_outputs(o_output, true);
				}
			}
			// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
			if (state_ == 2) {
				TMatrix linked_mu = layer.latent_output.first;
				TMatrix linked_var = layer.latent_output.second;
				TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
				TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
				int column = 0;
				for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TVector latent_mu = TVector::Zero(linked_mu.rows());
					TVector latent_var = TVector::Zero(linked_mu.rows());
					node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var);
					output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
					output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
					column++;
				}
				latent_output.first = output_mean;
				latent_output.second = output_variance;
			}
			return *this;
		}

		// Main Functions
		void set_parameter_history(std::vector<TMatrix>& history) {
			partitioned_history = history;
		}
		std::vector<TMatrix> get_parameter_history() {
			std::vector<TMatrix> history;
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				history.push_back(node->get_parameter_history());
			}
			return history;
		}

		void train() {
			if (state_ == 0 || state_ == 2) { check_nodes(); state_ = 1; }
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
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
			for (std::vector<GPNode>::size_type i = 0; i != nodes.size(); ++i)
			{
				MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
				node_mu.block(0, i, X.rows(), 1) = pred.first;
				node_var.block(0, i, X.rows(), 1) = pred.second;
			}

			latent_output = std::make_pair(node_mu, node_var);
			//latent_output = pred;
		}

		// Setters
		void set_inputs(const TMatrix& inputs) {
			o_input = inputs;
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->inputs = inputs;
			}
		}
		void set_outputs(const TMatrix& outputs, bool latent = false) {
			BoolVector missing;
			if (latent) { missing = BoolVector::Ones(outputs.rows()); }
			else { missing = BoolVector::Zero(outputs.rows()); }
			if ((outputs.array().isNaN()).any())
			{
				missing = get_missing_index<BoolVector>(outputs);
			}
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->outputs = outputs;
				node->missing = missing;
			}
			o_output = outputs;

		}
		// Getters
		TMatrix get_inputs() { return o_input; }
		TMatrix get_outputs() { return o_output; }

		// Read Only
		const std::string state() const {
			if (state_ == 0) { return "initialize"; }
			if (state_ == 1) { return "training"; }
			if (state_ == 2) { return "predict"; }
		}
		// Expose to Python
		const MatrixPair latent_output_() const { return latent_output; }
		friend class SIDGP;
	public:
		std::vector<GPNode> nodes;
		int index = 0;
	private:
		TMatrix o_input;
		TMatrix o_output;
		MatrixPair latent_output;
		std::vector<TMatrix> partitioned_history;
		int state_ = 0;

	};

#pragma region SAMPLING_ROUTINES
	TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
		return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	}

	TMatrix sample_mvn(const TMatrix& K) {
		MVN sampler(K);
		return sampler();
	}

	double log_likelihood(const TMatrix& K, const TMatrix& outputs) {
		TLLT chol(K);
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
		double lml = -0.5 * (logdet + quad);
		return lml;
	}

	void one_sample(GPNode& target, Layer& linked) {
		const TMatrix X = target.inputs;
		const TMatrix f = target.outputs;

		if (target.missing.count() == target.missing.size()) { return; }
		TMatrix K = target.kernel->K(X, X, target.likelihood_variance.value());
		K *= target.scale.value();
		TMatrix nu = sample_mvn(K);
		double log_y = 0.0;
		for (std::vector<GPNode>::iterator node = linked.nodes.begin(); node != linked.nodes.end(); ++node) {
			const TMatrix W = node->inputs;
			const TMatrix Y = node->outputs;
			TMatrix Kw = node->kernel->K(W, W, node->likelihood_variance.value());
			Kw *= node->scale.value();
			log_y += log_likelihood(Kw, Y);
		}
		log_y += log(random_uniform());
		double theta = random_uniform(0.0, 2.0 * PI);
		double theta_min = theta - 2.0 * PI;
		double theta_max = theta;

		TVector mean = TVector::Zero(X.rows());
		while (true) {
			TMatrix fp = update_f(f, nu, mean, theta);
			double log_yp = 0.0;
			for (std::vector<GPNode>::iterator node2 = linked.nodes.begin(); node2 != linked.nodes.end(); ++node2) {
				node2->inputs = fp;
				const TMatrix W2 = node2->inputs;
				const TMatrix Y2 = node2->outputs;
				TMatrix Kw2 = node2->kernel->K(W2, W2, node2->likelihood_variance.value());
				Kw2 *= node2->scale.value();
				log_yp += log_likelihood(Kw2, Y2);
			}
			if (log_yp > log_y) { target.outputs = fp; return; }
			else {
			
				if (theta < 0) { theta_min = theta; }
				else { theta_max = theta; }
				theta = random_uniform(theta_min, theta_max);		
			}
		}
	}
#pragma endregion

	class SIDGP {

	private:
		#pragma region PRINT_UTILITIES
		void print_utility(int arg, int idx) {
			if (arg == 1) {
				std::cout << std::setw(3) << std::left << idx + 1 << std::setw(2) << std::right << " : ";
			}
		}
		void print_utility(std::size_t idx) {
			if (idx == 0) { std::cout << "LAYER " << idx + 1 << " -> " << "LAYER " << idx + 2 << " -> "; }
			else if (idx == layers.size() - 2) { std::cout << "LAYER " << layers.size() << "\r" << std::flush; }
			else { std::cout << "LAYER " << idx + 2 << " -> " << "\r" << std::flush; }
		}
		void print_utility(int idx, double& progress, double obj_fxn) {
			std::cout << std::setw(3) << std::left << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER "
				<< std::setw(3) << std::left << idx
				<< std::setw(10) << std::right << std::setprecision(4) << std::fixed << obj_fxn << "\r" << std::flush;
		}
		void print_utility(int idx, double& progress) {
			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER " << std::setw(3) << std::left << idx << "\r" << std::flush;
		}
#pragma endregion
		void initialize_layers() {
			if (layers.front().o_input.size() == 0) { throw std::exception("First Layer Requires Observed Inputs"); }
			if (layers.back().o_output.size() == 0) { throw std::exception("Last Layer Requires Observed Outputs"); }
			layers.front().index = 1;
			// Propagate First Layer
			TMatrix X = layers.front().get_inputs();
			layers.front().set_outputs(X, true);

			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
				layer->state_ = 0;
				(*layer)(*std::next(layer));
			}
		}
		void sample(int n_burn = 10) {
			for (int i = 0; i < n_burn; ++i) {
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
					for (std::vector<GPNode>::iterator node = layer->nodes.begin(); node != layer->nodes.end() - 1; ++node){
						one_sample(*node, *std::next(layer));
					}
				}
			}
		}
		void partitioned_sample(std::vector<Layer>& local_layers, int n_burn = 10) {
			for (int i = 0; i < n_burn; ++i) {
				for (std::vector<Layer>::iterator layer = local_layers.begin(); layer != local_layers.end() - 1; ++layer) {
					for (std::vector<GPNode>::iterator node = layer->nodes.begin(); node != layer->nodes.end() - 1; ++node) {
						one_sample(*node, *std::next(layer));
					}
				}
			}
		}
		void partitioned_train(int& n_iter, int& ess_burn, int& n_thread) {
			p_train = true;
			// Make (n_thread) copies of layers
			//std::vector<std::vector<Layer>> layer_threads(n_thread, layers);
			std::vector<std::vector<Layer>> layer_threads;
			for (int t = 0; t < n_thread; ++t)
			{layer_threads.push_back(layers);}
			// Diving total number of iterations by number of threads
			int iter_split = n_iter / n_thread;
			int remainder = n_iter % n_thread;
			// Setup thread task
			thread_pool pool;
			auto train_task = [=, &ess_burn](std::vector<Layer> local_layers, int split)
			{
				for (int i = 0; i < split; ++i) {
					partitioned_sample(local_layers, ess_burn);
					for (std::vector<Layer>::iterator layer = local_layers.begin(); layer != local_layers.end(); ++layer) {
						layer->train();
					}
				}
			};
			// Execute training for each thread
			for (std::size_t lt = 0; lt < layer_threads.size(); ++lt) {
				if (lt == 0){ pool.push_task(train_task, layer_threads[lt], iter_split + remainder); }
				else { pool.push_task(train_task, layer_threads[lt], iter_split); }
			}
			pool.wait_for_tasks();

			std::vector<std::vector<TMatrix>> final_params;
			// For each thread
			for (std::vector<std::vector<Layer>>::size_type thread = 0; thread != layer_threads.size(); ++thread){
				// For each layer in thread
				for (std::vector<Layer>::size_type ll = 0; ll != layer_threads[thread].size(); ++ll) {
					// If first thread, initialize final params vector: 
					// Each TMatrix represents parameter history from each node
					if (thread == 0) {final_params.push_back(layer_threads[thread][ll].get_parameter_history());}
					// For other threads
					else {
						std::vector<TMatrix> params_vector = layer_threads[thread][ll].get_parameter_history();
						// For each node, concatenate params with previous thread params
						for (std::size_t p = 0; p < params_vector.size(); ++p) {
							TMatrix param_tmp(final_params[ll][p].rows() + params_vector[p].rows(), params_vector[p].cols());
							param_tmp << final_params[ll][p], params_vector[p];
							final_params[ll][p] = param_tmp;
						}
					}
				}
			}

			// Assign parameter history to each layer in SIDGP model
			for (int fl = 0; fl < layers.size(); ++fl) {
				layers[fl].set_parameter_history(final_params[fl]);
			}
		}

	public:

		SIDGP(std::vector<Layer>& layers) : layers(layers) {
			initialize_layers();
			sample(10);
		}

		void train(int n_iter = 50, int ess_burn = 10, int n_thread = 1) {
			n_iter_ = n_iter;
			if (n_thread == 0 || n_thread == 1) {
				for (int i = 0; i < n_iter + 1; ++i) {
					double progress = double(i) * 100.0 / double(n_iter);
					// I-step
					sample(ess_burn);
					// M-step
					for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
						layer->train();
						print_utility(layer->index, progress);
					}
				}
			}
			else {partitioned_train(n_iter, ess_burn, n_thread);}
		}

		void estimate(Eigen::Index n_burn = 0) {
			if (n_burn == 0) { n_burn = std::size_t(0.75 * n_iter_); }
			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
				layer->estimate_parameters(n_burn, p_train);
			}
		}

		MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
			sample(50);
			const std::size_t n_layers = layers.size();
			TMatrix mean = TMatrix::Zero(X.rows(), 1);
			TMatrix variance = TMatrix::Zero(X.rows(), 1);
			std::vector<MatrixPair> predictions;
			for (int i = 0; i < n_impute; ++i) {
				print_utility(1, i);
				sample();
				layers.front().predict(X);
				std::size_t j = 1;
				for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
					print_utility(j);
					if (layer->state_ != 2) { layer->state_ = 2; }
					(*layer)(*std::prev(layer));
					j++;
				}
				predictions.push_back(layers.back().latent_output);
			}
			std::cout << std::endl;

			TMatrix output_mean = TMatrix::Zero(X.rows(), n_impute);
			TMatrix output_var = TMatrix::Zero(X.rows(), n_impute);
			for (int k = 0; k < predictions.size(); ++k)
			{
				output_mean.block(0, k, X.rows(), 1) = predictions[k].first;
				output_var.block(0, k, X.rows(), 1) = predictions[k].second;
			}

			mean = output_mean.rowwise().mean();
			variance = (square((output_mean).array()).matrix() + (output_var)).rowwise().mean();
			variance.array() -= square(mean.array());
			std::system("cls");
			return std::make_pair(mean, variance);
		}

		const int n_iterations() const { return n_iter_; }
		const std::vector<Layer> layers_() const { return layers; }
		std::vector<Layer> layers;

	private:
		int n_iter_ = 0;
		bool p_train = false;
	};




}


#ifdef OLD_DEEP_MODELS
namespace SMh::deep_models::gaussian_process_py {
	using namespace SMh::kernels;
	using namespace SMh::utilities;
	using namespace SMh::base_models;
	using namespace SMh::base_models::gaussian_process;

	TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
		return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	}

	class Node : public GPR {

	public:

		Node(shared_ptr<Kernel> kernel) : GPR(kernel) {}

		void train() {
			GPR::train();
			_NLL = objective_value;
			if (store_parameters) { _history.push_back(get_params()); }
		}

		TMatrix get_parameter_history() {
			if (_history.size() == 0) { throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
			Eigen::Index param_size = params_size();
			TMatrix history(_history.size(), param_size);
			for (std::vector<TVector>::size_type i = 0; i != _history.size(); ++i) {
				history.row(i) = _history[i];
			}
			return history;
		}
		double log_likelihood() {
			TMatrix K_ = compute_K();
			TLLT chol_(K_);
			double logdet = 2 * chol_.matrixL().toDenseMatrix().diagonal().array().log().sum();
			double quad = (outputs.array() * (K_.llt().solve(outputs)).array()).sum();
			double lml = -0.5 * (logdet + quad);
			return lml;
		}

		TMatrix compute_K() {
			TMatrix noise = TMatrix::Identity(inputs.rows(), inputs.rows());
			TMatrix K_ = kernel->K(inputs, inputs);
			K_ += (noise * likelihood_variance.value());
			return K_;
		}

		// Setters
		void set_inputs(const TMatrix& input) { inputs = input; inputs_changed(); }
		void set_outputs(const TMatrix& output) { outputs = output; outputs_changed(); }
		void set_missing(const BoolVector& missing_) { missing = missing_; }

		// Getters
		const TMatrix get_inputs() const { return inputs; }
		const TMatrix get_outputs() const { return outputs; }
		const BoolVector get_missing() const { return missing; }

		// Read Only Property
		double NLL() const { return _NLL; }
		bool store_parameters = false;
	private:
		double _NLL = std::numeric_limits<double>::infinity();
		std::vector<TVector> _history;
	};


	class Layer {

	public:
		Layer(std::vector<Node>& nodes) : nodes(nodes) {}
		double log_likelihood() {
			double nll = 0;
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				nll += node->log_likelihood();
			}
			return nll;
		}
		void estimate_parameters() {
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				TMatrix history = node->get_parameter_history();
				TVector theta = history.colwise().mean();
				node->set_params(theta);
			}
		}
		void train() {
			if (!nodes_checked) { check_nodes(); }
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				if (!node->store_parameters) { node->store_parameters = true; }
				node->train();
			}
		}
		MatrixPair predict(const TMatrix& X) {
			/*
			* All nodes predict X and output a pair of N X M matrices
			* Where N = number of rows X ; M = number of nodes in layer
			* The pair is the mean and variance.
			*/
			TMatrix node_mu(X.rows(), nodes.size());
			TMatrix node_var(X.rows(), nodes.size());
			for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
			{
				MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
				node_mu.block(0, i, X.rows(), 1).noalias() = pred.first;
				node_var.block(0, i, X.rows(), 1).noalias() = pred.second;
			}

			MatrixPair pred = std::make_pair(node_mu, node_var);
			return pred;
		}

		// Setters
		void set_observed_inputs(const TMatrix& inputs) {
			o_input = inputs;
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->set_inputs(inputs);
			}
		}
		void set_observed_outputs(const TMatrix& outputs) {
			o_output = outputs;
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->set_outputs(outputs);
			}

		}

		// Getters
		TMatrix get_observed_inputs() { return o_input; }
		TMatrix get_observed_outputs() { return o_output; }

		// Read Only Property
		std::vector<Node> nodes;

		// Read/Write Properties
		int index = 0;
	private:
		void check_nodes() {
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				if (node->inputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
				}
				if (node->outputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
				}
			}
			nodes_checked = true;
		}

	private:
		TMatrix o_input;
		TMatrix o_output;
		MatrixPair latent_input;
		bool nodes_checked = false;
	};


	//class SIDGP {

	//public:

	//	SIDGP(std::vector<Layer>& layers) : layers(layers) {}

	//	virtual void train(int n_iter, int ess_burn) = 0;

	//public:
	//	std::vector<Layer> layers;
	//};

}

namespace SMh::deep_models::gaussian_process_c1 {
	//	using namespace SMh::kernels;
	//	using namespace SMh::utilities;
	//	using namespace SMh::base_models::gaussian_process;
	//
	//	class Node {
	//
	//	public:
	//
	//		Node(shared_ptr<Kernel> kernel) { model.kernel = std::move(kernel); }
	//
	//		void train() {
	//			model.train();
	//			NLL_ = model.objective_value;
	//			if (store_parameters) { _history.push_back(model.get_params()); }
	//		}
	//		MatrixVariant predict(const TMatrix& X, bool return_var = false) const { return model.predict(X, return_var); }
	//
	//		// Functions Used  In Layer 
	//		double log_likelihood() { return model.log_likelihood(); }
	//		void set_params(const TVector& new_params) { model.set_params(new_params); }
	//		bool is_psd() { return model.is_psd(); }
	//		void kernel_expectations(const TMatrix& mean, const TMatrix& variance) { model.kernel->expectations(mean, variance); }
	//		void kernel_IJ(TMatrix& I, TMatrix& J, const TRVector& mean, const TMatrix& X, const Eigen::Index& idx) { model.kernel->IJ(I, J, mean, X, idx); }
	//		TMatrix sample_mvn() {
	//			MVN sampler(model.K);
	//			return sampler();
	//		}
	//
	//		// Setters
	//		void set_inputs(const TMatrix& input) { model.inputs = input; model.inputs_changed(); }
	//		void set_outputs(const TMatrix& output) { model.outputs = output; }
	//		void set_missing(const BoolVector& missing) { model.missing = missing; }
	//
	//		// Getters
	//		const TMatrix get_inputs() const { return model.inputs; }
	//		const TMatrix get_outputs() const { return model.outputs; }
	//		const BoolVector get_missing() const { return model.missing; }
	//		TMatrix get_parameter_history() {
	//			if (_history.size() == 0) { throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
	//			Eigen::Index param_size = model.params_size();
	//			TMatrix history(_history.size(), param_size);
	//			for (std::vector<TVector>::size_type i = 0; i != _history.size(); ++i) {
	//				history.row(i) = _history[i];
	//			}
	//			return history;
	//		}
	//		Parameter<double> likelihood_variance() const { return model.likelihood_variance; }
	//		Parameter<double> kernel_variance() const { return model.kernel->variance; }
	//		TMatrix K() const { return model.K; }
	//		TVector alpha() const { return model.alpha; }
	//
	//		// Read Only Property
	//		double NLL() const { return NLL_; }
	//		double parameters_stored() const { return store_parameters; }
	//
	//		friend class Layer;
	//	private:
	//		GPR model;
	//	private:
	//		bool store_parameters = false;
	//		double NLL_ = std::numeric_limits<double>::infinity();
	//		std::vector<TVector> _history;
	//	};
	//
	//	class Layer {
	//	private:
	//		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
	//			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	//		}
	//	public:
	//		Layer(std::vector<Node>& nodes_) {
	//			// Move elements in nodes_ to nodes
	//			nodes.insert(nodes.end(),
	//				std::make_move_iterator(nodes_.begin()),
	//				std::make_move_iterator(nodes_.end()));
	//			nodes_.erase(nodes_.begin(), nodes_.end());
	//		}
	//
	//		double log_likelihood() {
	//			double ll = 0;
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
	//			{
	//				ll += node->log_likelihood();
	//			}
	//			return ll;
	//		}
	//		void estimate_parameters() {
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//				TMatrix history = node->get_parameter_history();
	//				TVector theta = history.colwise().mean();
	//				node->set_params(theta);
	//			}
	//		}
	//		void train() {
	//			if (state_ == 0 || state_ == 2) { check_nodes(); state_ = 1; }
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//				if (!node->store_parameters) { node->store_parameters = true; }
	//				node->train();
	//			}
	//		}
	//		void predict(const TMatrix& X) {
	//			/*
	//			* All nodes predict X and output a pair of N X M matrices
	//			* Where N = number of rows X ; M = number of nodes in layer
	//			* The pair is the mean and variance.
	//			*/
	//			TMatrix node_mu(X.rows(), nodes.size());
	//			TMatrix node_var(X.rows(), nodes.size());
	//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
	//			{
	//				MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
	//				node_mu.block(0, i, X.rows(), 1).noalias() = pred.first;
	//				node_var.block(0, i, X.rows(), 1).noalias() = pred.second;
	//			}
	//
	//			latent_output = std::make_pair(node_mu, node_var);
	//		}
	//
	//		// Setters
	//		void set_inputs(const TMatrix& inputs) {
	//			o_input = inputs;
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//				node->set_inputs(inputs);
	//			}
	//		}
	//		void set_outputs(const TMatrix& outputs, bool latent = false) {
	//			BoolVector missing;
	//			if (latent) { missing = BoolVector::Ones(outputs.rows()); }
	//			else { missing = BoolVector::Zero(outputs.rows()); }
	//			if ((outputs.array().isNaN()).any())
	//			{
	//				missing = get_missing_index<BoolVector>(outputs);
	//			}
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//				node->set_outputs(outputs);
	//				node->set_missing(missing);
	//			}
	//			o_output = outputs;
	//
	//		}
	//		// Getters
	//		TMatrix get_inputs() { return o_input; }
	//		TMatrix get_outputs() { return o_output; }
	//
	//		// Read Only
	//		const std::string state() const {
	//			if (state_ == 0) { return "initialize"; }
	//			if (state_ == 1) { return "training"; }
	//			if (state_ == 2) { return "predict"; }
	//		}
	//		// Read/Write Properties
	//		int index = 0;
	//
	//		// operator() overload
	//		Layer& operator()(Layer& layer) {
	//			// Initialize
	//			if (state_ == 0) {
	//				layer.index = index + 1;
	//				// Inputs
	//				if (layer.nodes.size() == nodes.size())
	//				{
	//					layer.set_inputs(o_output);
	//				}
	//				else if (layer.nodes.size() < nodes.size())
	//				{
	//					// Apply Dimensionality Reduction (Kernel PCA)
	//				}
	//				else
	//				{
	//					// Dimension Expansion
	//				}
	//				// Outputs
	//				if (layer.o_output.size() == 0)
	//				{
	//					layer.set_outputs(o_output, true);
	//				}
	//			}
	//			// Sample
	//			if (state_ == 1) {
	//				double log_y = layer.log_likelihood();
	//				log_y += log(random_uniform(0.0, 1.0));
	//				TVector mean = TVector::Zero(o_input.rows());
	//				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//					BoolVector missing = node->get_missing();
	//					if (!(missing.count() > 0)) { continue; }
	//					TMatrix nu = node->sample_mvn();
	//					double theta = random_uniform(0.0, 2 * PI);
	//					double theta_min = theta - 2 * PI;
	//					double theta_max = theta;
	//
	//					while (true) {
	//						TMatrix fp = update_f(node->get_outputs(), nu, mean, theta);
	//						layer.set_inputs(fp);
	//						double log_yp = layer.log_likelihood();
	//						if (log_yp > log_y) { node->set_outputs(fp); break; }
	//						else {
	//							if (theta < 0) { theta_min = theta; }
	//							else { theta_max = theta; }
	//							theta = random_uniform(theta_min, theta_max);
	//						}
	//					}
	//				}
	//			}
	//			// Linked Prediction
	//			if (state_ == 2) {
	//				TMatrix mean = layer.latent_output.first;
	//				TMatrix variance = layer.latent_output.second;
	//				TMatrix output_mean = TMatrix::Zero(mean.rows(), nodes.size());
	//				TMatrix output_variance = TMatrix::Zero(variance.rows(), nodes.size());
	//				int column = 0;
	//				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
	//					TVector pred_mu = TVector::Zero(mean.rows());
	//					TVector pred_var = TVector::Zero(mean.rows());
	//					node->kernel_expectations(mean, variance);
	//					double trace;
	//					for (Eigen::Index i = 0; i < mean.rows(); ++i) {
	//						TMatrix inputs = node->get_inputs();
	//						TMatrix I = TMatrix::Ones(inputs.rows(), 1);
	//						TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
	//						node->kernel_IJ(I, J, static_cast<TRVector>(mean.row(i)), inputs, i);
	//						if (node->is_psd()) { trace = (node->K().llt().solve(J)).trace(); }
	//						else { trace = (node->K().colPivHouseholderQr().solve(J)).trace(); }
	//						double Ialpha = (I.cwiseProduct(node->alpha())).array().sum();
	//						pred_mu[i] = (Ialpha);
	//						pred_var[i] = (abs(((node->alpha().transpose() * J).cwiseProduct(node->alpha().transpose()).array().sum() - (pow(Ialpha, 2)) + node->kernel_variance().value * (1 + 1e-8 - trace))));
	//					}
	//					output_mean.block(0, column, mean.rows(), 1) = pred_mu;
	//					output_variance.block(0, column, variance.rows(), 1) = pred_var;
	//					column++;
	//				}
	//				latent_output.first = output_mean;
	//				latent_output.second = output_variance;
	//			}
	//			return *this;
	//		}
	//
	//		std::vector<Node> nodes;
	//		MatrixPair latent_output;
	//
	//		friend class SIDGP;
	//
	//	private:
	//		void check_nodes() {
	//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
	//			{
	//				if (node->get_inputs().size() == 0)
	//				{
	//					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
	//				}
	//				if (node->get_outputs().size() == 0)
	//				{
	//					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
	//				}
	//			}
	//		}
	//
	//	private:
	//		TMatrix o_input;
	//		TMatrix o_output;
	//		int state_ = 0;
	//	};
	//
	//	class SIDGP {
	//
	//	private:
	//
	//	#pragma region PRINT_UTILITIES
	//		void print_utility(int arg, int idx) {
	//			if (arg == 1) {
	//				std::cout << std::setw(3) << std::left << idx + 1 << std::setw(2) << std::right << " : ";
	//			}
	//		}
	//		void print_utility(std::size_t idx) {
	//			if (idx == 0) { std::cout << "LAYER " << idx + 1 << " -> " << "LAYER " << idx + 2 << " -> "; }
	//			else if (idx == layers.size() - 2) { std::cout << "LAYER " << layers.size() << "\r" << std::flush; }
	//			else { std::cout << "LAYER " << idx + 2 << " -> "; }
	//		}
	//		void print_utility(int idx, double& progress, double obj_fxn) {
	//			std::cout << std::setw(3) << std::left << progress << std::setw(5) << std::left << " % |";
	//			std::cout << std::setw(7) << std::left << " LAYER "
	//				<< std::setw(3) << std::left << idx
	//				<< std::setw(10) << std::right << std::setprecision(4) << std::fixed << obj_fxn << "\r" << std::flush;
	//		}
	//		void print_utility(int idx, double& progress) {
	//			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
	//			std::cout << std::setw(7) << std::left << " LAYER " << std::setw(3) << std::left << idx << "\r" << std::flush;
	//		}
	//#pragma endregion
	//
	//		void initialize_layers() {
	//			if (layers.front().o_input.size() == 0) { throw std::exception("First Layer Requires Observed Inputs"); }
	//			if (layers.back().o_output.size() == 0) { throw std::exception("Last Layer Requires Observed Outputs"); }
	//			layers.front().index = 1;
	//			// Propagate First Layer
	//			TMatrix X = layers.front().get_inputs();
	//			layers.front().set_outputs(X, true);
	//
	//			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
	//				layer->state_ = 0;
	//				(*layer)(*std::next(layer));
	//			}
	//
	//		}
	//		void sample(int n_burn = 10) {
	//			for (int i = 0; i < n_burn; ++i) {
	//				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
	//					if (layer->state_ != 1) { layer->state_ = 1; }
	//					(*layer)(*std::next(layer));
	//				}
	//			}
	//		}
	//
	//	public:
	//
	//		SIDGP(std::vector<Layer>& layers_) {
	//			layers.insert(layers.end(),
	//				std::make_move_iterator(layers_.begin()),
	//				std::make_move_iterator(layers_.end()));
	//			layers_.erase(layers_.begin(), layers_.end());
	//			initialize_layers();
	//			sample(10);
	//		}
	//
	//		void train(int n_iter = 500, int ess_burn = 10) {
	//			for (int i = 0; i < n_iter; ++i) {
	//				double progress = double(i) * 100.0 / double(n_iter);
	//				// I-step
	//				sample(ess_burn);
	//				// M-step
	//				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
	//					layer->train();
	//					print_utility(layer->index, progress);
	//				}
	//			}
	//			// Estimate Point Parameters and Assign to Nodes
	//			for (std::vector<Layer>::size_type j = 0; j != layers.size(); ++j) {
	//				layers[j].estimate_parameters();
	//			}
	//		}
	//
	//		MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
	//			sample(50);
	//			const std::size_t n_layers = layers.size();
	//			TMatrix mean = TMatrix::Zero(X.rows(), 1);
	//			TMatrix variance = TMatrix::Zero(X.rows(), 1);
	//			std::vector<MatrixPair> predictions;
	//			for (int i = 0; i < n_impute; ++i) {
	//				print_utility(1, i);
	//				sample();
	//				layers.front().predict(X);
	//				for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
	//					if (layer->state_ != 2) { layer->state_ = 2; }
	//					(*layer)(*std::prev(layer));
	//				}
	//				//predictions.push_back(pred);
	//			}
	//
	//			TMatrix output_mean = TMatrix::Zero(X.rows(), n_impute);
	//			TMatrix output_var = TMatrix::Zero(X.rows(), n_impute);
	//			for (int k = 0; k < predictions.size(); ++k)
	//			{
	//				output_mean.block(0, k, X.rows(), 1) = predictions[k].first;
	//				output_var.block(0, k, X.rows(), 1) = predictions[k].second;
	//			}
	//			mean = output_mean.rowwise().mean();
	//			variance = (square((output_mean).array()).matrix() + (output_var)).rowwise().mean();
	//
	//			//mean.noalias() += pred.first;
	//			//variance.noalias() += (square((pred.first).array()).matrix() + (pred.second));
	//
	//			std::system("cls");
	//			//mean.array() /= n_impute;
	//			//variance.array() /= n_impute;
	//			variance.array() -= square(mean.array());
	//			return std::make_pair(mean, variance);
	//		}
	//
	//	public:
	//		std::vector<Layer> layers;
	//	};

}

namespace SMh::deep_models::gaussian_process_c2 {
	using namespace SMh::kernels;
	using namespace SMh::utilities;
	using namespace SMh::base_models::gaussian_process;

	class Node : private GPR {

	private:
		void linked_prediction(TVector& latent_mu, TVector& latent_var, const TMatrix& linked_mu, const TMatrix& linked_var) {
			kernel->expectations(linked_mu, linked_var);
			double trace = 0.0;
			TMatrix K_unscaled = K.array() / kernel->variance.value();
			TVector alpha_unscaled = K_unscaled.llt().solve(outputs);
			// Apply Multi-threading
			for (Eigen::Index i = 0; i < linked_mu.rows(); ++i) {
				TMatrix I = TMatrix::Ones(inputs.rows(), 1);
				TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
				kernel->IJ(I, J, static_cast<TRVector>(linked_mu.row(i)), inputs, i);
				if (is_psd()) { trace = (K_unscaled.llt().solve(J)).trace(); }
				else { trace = (K_unscaled.colPivHouseholderQr().solve(J)).trace(); }
				double Ialpha = (I.cwiseProduct(alpha_unscaled)).array().sum();
				latent_mu[i] = (Ialpha);
				latent_var[i] =
					(abs((((alpha_unscaled.transpose() * J).cwiseProduct(alpha_unscaled.transpose()).array().sum()
						- (pow(Ialpha, 2))) + kernel->variance.value() * ((1.0 + likelihood_variance.value()) - trace))));
			}
		}

	public:
		Node(shared_ptr<Kernel> kernel) : GPR(std::move(kernel)) {}
		// Python Constructor
		Node(shared_ptr<Kernel> kernel, Parameter<double> likelihood_variance_) : GPR(std::move(kernel)) {
			likelihood_variance = std::move(likelihood_variance_);
		}
		// Expose GPR attributes for user
		using GPR::kernel;
		using GPR::likelihood_variance;

		// Main Functions
		TMatrix get_parameter_history() {
			if (history.size() == 0) { throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
			Eigen::Index param_size = params_size();
			TMatrix _history(history.size(), param_size);
			for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
				_history.row(i) = history[i];
			}
			return _history;
		}
		TMatrix sample_mvn() {
			MVN sampler(this->K);
			return sampler();
		}

		// Setters
		void set_inputs(const TMatrix& input) { inputs = input; inputs_changed(); }
		void set_outputs(const TMatrix& output) { outputs = output; }
		void set_missing(const BoolVector& missing_) { missing = missing_; }

		// Getters
		const TMatrix get_inputs() const { return inputs; }
		const TMatrix get_outputs() const { return outputs; }
		const BoolVector get_missing() const { return missing; }

		// Read Only Property
		const bool parameters_stored() const { return store_parameters; }
		// Expose To Python
		const Parameter<double> likelihood_variance_() const { return likelihood_variance; }
		const shared_ptr<Kernel> kernel_() const { return kernel; }

		friend class Layer;
	};

	class Layer {
	private:
		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
		}
		void check_nodes() {
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				if (node->inputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
				}
				if (node->outputs.size() == 0)
				{
					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
				}
			}
		}

	public:
		friend class SIDGP;
		Layer(std::vector<Node>& nodes_) {
			// Move elements in nodes_ to nodes
			nodes.insert(nodes.end(),
				std::make_move_iterator(nodes_.begin()),
				std::make_move_iterator(nodes_.end()));
			nodes_.erase(nodes_.begin(), nodes_.end());
		}

		// Operator() overload
		Layer& operator()(Layer& layer) {
			// Initialize			[ CurrentLayer(NextLayer) ]
			if (state_ == 0) {
				layer.index = index + 1;
				// Inputs
				if (layer.nodes.size() == nodes.size())
				{
					layer.set_inputs(o_output);
				}
				else if (layer.nodes.size() < nodes.size())
				{
					// Apply Dimensionality Reduction (Kernel PCA)
				}
				else
				{
					// Dimension Expansion
				}
				// Outputs
				if (layer.o_output.size() == 0)
				{
					layer.set_outputs(o_output, true);
				}
			}
			// Sample				[ CurrentLayer(NextLayer) ]
			if (state_ == 1) {
				double log_y = layer.log_likelihood();
				log_y += log(random_uniform(0.0, 1.0));
				TVector mean = TVector::Zero(o_input.rows());
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					if (!(node->missing.count() > 0)) { continue; }
					TMatrix nu = node->sample_mvn();
					double theta = random_uniform(0.0, 2.0 * PI);
					double theta_min = theta - 2.0 * PI;
					double theta_max = theta;

					while (true) {
						TMatrix fp = update_f(node->outputs, nu, mean, theta);
						layer.set_inputs(fp);
						double log_yp = layer.log_likelihood();
						if (log_yp > log_y) { node->outputs = fp; node->outputs_changed(); break; }
						else {
							if (theta < 0) { theta_min = theta; }
							else { theta_max = theta; }
							theta = random_uniform(theta_min, theta_max);
						}
					}
				}
			}
			// Linked Prediction	[ CurrentLayer(PreviousLayer) ]
			if (state_ == 2) {
				TMatrix linked_mu = layer.latent_output.first;
				TMatrix linked_var = layer.latent_output.second;
				TMatrix output_mean = TMatrix::Zero(linked_mu.rows(), nodes.size());
				TMatrix output_variance = TMatrix::Zero(linked_var.rows(), nodes.size());
				int column = 0;
				for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
					TVector latent_mu = TVector::Zero(linked_mu.rows());
					TVector latent_var = TVector::Zero(linked_mu.rows());
					node->linked_prediction(latent_mu, latent_var, linked_mu, linked_var);
					output_mean.block(0, column, linked_mu.rows(), 1) = latent_mu;
					output_variance.block(0, column, linked_var.rows(), 1) = latent_var;
					column++;
				}
				latent_output.first = output_mean;
				latent_output.second = output_variance;
			}
			return *this;
		}

		// Main Functions
		double log_likelihood() {
			double ll = 0;
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
			{
				ll += node->log_likelihood();
			}
			return ll;
		}
		void estimate_parameters(Eigen::Index n_burn) {
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				TMatrix history = node->get_parameter_history();
				TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
				node->set_params(theta);
			}
		}
		void train() {
			if (state_ == 0 || state_ == 2) { check_nodes(); state_ = 1; }
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
				node_mu.block(0, i, X.rows(), 1).noalias() = pred.first;
				node_var.block(0, i, X.rows(), 1).noalias() = pred.second;
			}

			latent_output = std::make_pair(node_mu, node_var);
		}

		// Setters
		void set_inputs(const TMatrix& inputs) {
			o_input = inputs;
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->set_inputs(inputs);
			}
		}
		void set_outputs(const TMatrix& outputs, bool latent = false) {
			BoolVector missing;
			if (latent) { missing = BoolVector::Ones(outputs.rows()); }
			else { missing = BoolVector::Zero(outputs.rows()); }
			if ((outputs.array().isNaN()).any())
			{
				missing = get_missing_index<BoolVector>(outputs);
			}
			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->outputs = outputs;
				node->missing = missing;
			}
			o_output = outputs;

		}
		// Getters
		TMatrix get_inputs() { return o_input; }
		TMatrix get_outputs() { return o_output; }

		// Read Only
		const std::string state() const {
			if (state_ == 0) { return "initialize"; }
			if (state_ == 1) { return "training"; }
			if (state_ == 2) { return "predict"; }
		}
		// Expose to Python
		const MatrixPair latent_output_() const { return latent_output; }

	public:
		std::vector<Node> nodes;
		int index = 0;
	private:
		TMatrix o_input;
		TMatrix o_output;
		MatrixPair latent_output;
		int state_ = 0;
	};

	class SIDGP {

	private:
#pragma region PRINT_UTILITIES
		void print_utility(int arg, int idx) {
			if (arg == 1) {
				std::cout << std::setw(3) << std::left << idx + 1 << std::setw(2) << std::right << " : ";
			}
		}
		void print_utility(std::size_t idx) {
			if (idx == 0) { std::cout << "LAYER " << idx + 1 << " -> " << "LAYER " << idx + 2 << " -> "; }
			else if (idx == layers.size() - 2) { std::cout << "LAYER " << layers.size() << "\r" << std::flush; }
			else { std::cout << "LAYER " << idx + 2 << " -> "; }
		}
		void print_utility(int idx, double& progress, double obj_fxn) {
			std::cout << std::setw(3) << std::left << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER "
				<< std::setw(3) << std::left << idx
				<< std::setw(10) << std::right << std::setprecision(4) << std::fixed << obj_fxn << "\r" << std::flush;
		}
		void print_utility(int idx, double& progress) {
			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << " LAYER " << std::setw(3) << std::left << idx << "\r" << std::flush;
		}
#pragma endregion
		void initialize_layers() {
			if (layers.front().o_input.size() == 0) { throw std::exception("First Layer Requires Observed Inputs"); }
			if (layers.back().o_output.size() == 0) { throw std::exception("Last Layer Requires Observed Outputs"); }
			layers.front().index = 1;
			// Propagate First Layer
			TMatrix X = layers.front().get_inputs();
			layers.front().set_outputs(X, true);

			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
				layer->state_ = 0;
				(*layer)(*std::next(layer));
			}
		}
		void sample(int n_burn = 10) {
			for (int i = 0; i < n_burn; ++i) {
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
					if (layer->state_ != 1) { layer->state_ = 1; }
					(*layer)(*std::next(layer));
				}
			}
		}
	public:

		SIDGP(std::vector<Layer>& layers_) : layers(layers_) {
			//layers.insert(layers.end(),
			//	std::make_move_iterator(layers_.begin()),
			//	std::make_move_iterator(layers_.end()));
			//layers_.erase(layers_.begin(), layers_.end());
			initialize_layers();
			sample(10);
		}

		void train(int n_iter = 500, int ess_burn = 10) {
			train_iter = n_iter;
			for (int i = 0; i < n_iter + 1; ++i) {
				double progress = double(i) * 100.0 / double(n_iter);
				// I-step
				sample(ess_burn);
				// M-step
				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
					layer->train();
					print_utility(layer->index, progress);
				}
			}
			// Estimate Point Parameters and Assign to Nodes
			//for (std::vector<Layer>::size_type j = 0; j != layers.size(); ++j) {
			//	layers[j].estimate_parameters();
			//}
		}

		void estimate(Eigen::Index n_burn = 0) {
			if (n_burn == 0) { n_burn = std::size_t(0.75 * train_iter); }
			for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
				layer->estimate_parameters(n_burn);
			}
		}

		MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
			sample(50);
			const std::size_t n_layers = layers.size();
			TMatrix mean = TMatrix::Zero(X.rows(), 1);
			TMatrix variance = TMatrix::Zero(X.rows(), 1);
			std::vector<MatrixPair> predictions;
			for (int i = 0; i < n_impute; ++i) {
				print_utility(1, i);
				sample();
				layers.front().predict(X);
				std::size_t j = 1;
				for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
					print_utility(j);
					if (layer->state_ != 2) { layer->state_ = 2; }
					(*layer)(*std::prev(layer));
					j++;
				}
				predictions.push_back(layers.back().latent_output);
			}

			TMatrix output_mean = TMatrix::Zero(X.rows(), n_impute);
			TMatrix output_var = TMatrix::Zero(X.rows(), n_impute);
			for (int k = 0; k < predictions.size(); ++k)
			{
				output_mean.block(0, k, X.rows(), 1) = predictions[k].first;
				output_var.block(0, k, X.rows(), 1) = predictions[k].second;
			}

			std::string path = "E:/23620029-Faiz/C/SMh/cpp_interface/tests/output_mean.dat";
			//save_data(path, output_mean);

			mean = output_mean.rowwise().mean();
			variance = (square((output_mean).array()).matrix() + (output_var)).rowwise().mean();
			variance.array() -= square(mean.array());
			std::system("cls");
			return std::make_pair(mean, variance);
		}

		const int train_iter_() const { return train_iter; }
		const std::vector<Layer> layers_() const { return layers; }

	private:
		int train_iter = 0;
		std::vector<Layer> layers;
	};

}

//namespace SMh::deep_models::gaussian_process {
//	using namespace SMh::kernels;
//	using namespace SMh::utilities;
//	using namespace SMh::base_models::gaussian_process;
//
//	// update_cholesky() called on input/output change and set_params
//
//
//	class Node : public GPR {
//
//	private:
//		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
//			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
//		}
//
//	public:
//		#pragma region MAKE_PRIVATE
//		using GPR::K;
//		using GPR::chol;
//		using GPR::alpha;
//		using GPR::update_cholesky;
//		#pragma endregion
//		
//		Node(shared_ptr<Kernel> kernel) : GPR(kernel) {}
//		void train() {
//			GPR::train();
//			NLL = objective_value;
//			if (store_parameters) { _history.push_back(get_params()); }
//		}
//		TMatrix get_parameter_history() {
//			if (_history.size() == 0) { throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
//			Eigen::Index param_size = params_size();
//			TMatrix history(_history.size(), param_size);
//			for (std::vector<TVector>::size_type i = 0; i != _history.size(); ++i) {
//				history.row(i) = _history[i];
//			}
//			return history;
//		}		
//		double log_marginal_likelihood() {
//			// Add Jitter
//			//K.array() += (TMatrix::Identity(inputs.rows(), outputs.rows()).array()*1e-6);
//			double lml = objective_fxn();
//			//double logdet = log(K.determinant());
//			//double quad = (outputs.array() * (K.llt().solve(outputs)).array()).sum();
//			//double lml = -0.5 * (logdet + quad);
//			return lml;
//		}
//
//		void sample(std::vector<Node>& next_layer) {
//			TMatrix nu;
//			if (!(missing.count() > 0)) { return; }
//			TMatrix mean = TMatrix::Zero(inputs.rows(), 1);
//			MVN sampler{ mean, K };
//			nu = sampler();
//			//nu.array() /= 10;
//			double log_y = 0;
//			for (std::vector<Node>::iterator it1 = next_layer.begin(); it1 != next_layer.end(); ++it1)
//			{log_y += it1->log_marginal_likelihood();}
//			log_y += log(random_uniform(0.0, 1.0));
//			double params = random_uniform(0.0, 2 * PI);
//			double params_min = params - 2 * PI;
//			double params_max = params;
//			while (true) {
//				TMatrix fp = update_f(outputs, nu, mean, params);
//				double log_yp = 0;
//				for (std::vector<Node>::iterator it2 = next_layer.begin(); it2 != next_layer.end(); ++it2)
//				{
//					it2->inputs = fp;
//					it2->update_cholesky();
//					log_yp += it2->log_marginal_likelihood();
//				}
//				if (log_yp > log_y) { outputs = fp; update_cholesky(); return; }
//				else {
//					if (params < 0) { params_min = params; }
//					else { params_max = params; }
//					params = random_uniform(params_min, params_max);
//				}
//			}
//		}
//		bool store_parameters = false;
//		double NLL = std::numeric_limits<double>::infinity();
//	private:
//		std::vector<TVector> _history;
//	};
//	
//	class Layer {
//
//	private:
//		#pragma region DELETE
//		void conditional_mvn(GPR& node, TMatrix& mu, TMatrix& var) {
//			TMatrix X1 = mask_matrix(node.inputs, node.missing, false, 0);
//			TMatrix W1 = mask_matrix(node.inputs, node.missing, true, 0);
//			TMatrix W2 = mask_matrix(node.outputs, node.missing, true, 0);
//			TMatrix R = node.kernel->K(W1, W1);
//			TMatrix c = node.kernel->K(X1, X1, node.likelihood_variance.value());
//			TMatrix r = node.kernel->K(W1, X1);
//			TLLT chol = R.llt();
//			TMatrix alpha = chol.solve(r); // Rinv_r = np.linalg.solve(R, r)
//			TMatrix beta = (r.transpose() * alpha); // r_Rinv_r = r.T @ Rinv_r
//			TMatrix tmp(alpha.rows(), alpha.cols());
//			visit_lambda(alpha, [&tmp, &W2](double v, int i, int j) { tmp(i, j) = W2(i) * v; });
//			mu.resize(alpha.rows(), 1);
//			mu = tmp.colwise().sum().transpose();
//			var = (node.kernel->variance.value() * (c - beta)).cwiseAbs();
//
//		}
//		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
//			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
//		}
//		void sample(Layer& next_layer) {
//			//TMatrix mean, K, nu;
//			TMatrix nu;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{
//				if (!(node->missing.count() > 0)) { continue; }
//				//if (node->missing.count() == node->missing.size()) {
//				//	mean = TVector::Zero(node->inputs.rows());
//				//	K = node->kernel->K(node->inputs, node->inputs, node->likelihood_variance.value());
//				//}
//				//else { conditional_mvn(*node, mean, K); }
//				TMatrix mean = TMatrix::Zero(node->inputs.rows(), 1);
//				//nu = node->sample_m_();
//				//nu.array() /= 10.0;
//				double log_y = next_layer.log_marginal_likelihood();
//				log_y += log(random_uniform(0.0, 1.0));
//				double params = random_uniform(0.0, 2 * PI);
//				double params_min = params - 2 * PI;
//				double params_max = params;
//
//				while (true) {
//					TMatrix fp = update_f(node->outputs, nu, mean, params);
//					next_layer.observed_inputs(fp);
//					double log_yp = next_layer.log_marginal_likelihood();
//
//					if (log_yp > log_y) { node->outputs = fp; return; }
//					else {
//						if (params < 0) { params_min = params; }
//						else { params_max = params; }
//						params = random_uniform(params_min, params_max);
//					}
//				}
//			}
//		}
//		void linked_prediction(MatrixPair& output, std::size_t n_nodes) {
//			/*
//			*	TODO: 
//			*	Dimension Reduction
//			*	Move Expectations to Kernel Class
//			*	Thread the loop instead
//			*/
//			TMatrix mean = latent_input.first;
//			TMatrix variance = latent_input.second;
//			TMatrix output_mean(mean.rows(), n_nodes);
//			TMatrix output_variance(variance.rows(), n_nodes);
//			int column = 0;
//			
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				Kernel& kernel = *(node->kernel);
//				TMatrix& X = node->inputs;
//				TMatrix& Y = node->outputs;
//				TMatrix K = kernel.K(X, X);
//				TLLT    llt(K);
//				bool is_psd = true;
//				TVector alpha(Y.rows());
//				if (!(K.isApprox(K.transpose())) || llt.info() == Eigen::NumericalIssue) {
//					alpha = K.colPivHouseholderQr().solve(observed_output);
//					is_psd = false;
//				}
//				else { alpha = K.ldlt().solve(Y); }
//				TVector pred_mu(mean.rows());
//				TVector pred_var(mean.rows());
//
//				TRVector sqrd_ls = square(static_cast<TRVector>(kernel.length_scale.value()).array());
//				TMatrix xi_term1 = 1 / sqrt(1 + ((2 * variance.array()).rowwise() / sqrd_ls.array()));
//				TMatrix xi_term2 = (2 * variance.array()).rowwise() + sqrd_ls.array();
//				TMatrix zeta0 = 1 / sqrt(1 + ((4 * variance.array()).rowwise() / sqrd_ls.array()));
//				TMatrix QD1 = (8 * variance.array()).rowwise() + (2 * sqrd_ls.array());
//				TVector QD2 = 2 * square(kernel.length_scale.value().array()).array();
//				TMatrix xi;
//
//				const Eigen::Index n_rows = mean.rows();
//				const Eigen::Index n_cols = X.cols();
//				int split = int(mean.rows() / n_thread);
//				const int remainder = mean.rows() % n_thread;
//				//synced_stream sync_out;
//				thread_pool pool;
//				auto task =
//				[&X, &K, &is_psd, &kernel, &mean, &variance, &xi_term1, &xi_term2, &zeta0, &QD1, &QD2, &alpha, &pred_mu, &pred_var]
//				(int begin, int end, int s)
//				{
//					//sync_out.println("Task no. ", s, " executing.");
//					TMatrix xi;
//					for (int i = begin; i < end; ++i) {
//						TMatrix Xz = ((X.rowwise() - static_cast<TRVector>(mean.row(i))));
//						// Compute I
//						TMatrix I(X.rows(), 1);
//						xi.noalias() =
//						(exp(((-1 * square(Xz.array())).array().rowwise()
//						/ static_cast<TRVector>(xi_term2.row(i)).array())).matrix())
//						* (xi_term1.row(i).asDiagonal());
//						I = xi.rowwise().prod();
//						// Compute J
//						TMatrix J = TMatrix::Ones(X.rows(), X.rows());
//						for (int j = 0; j < Xz.cols(); ++j) {
//							TVector L = static_cast<TVector>(Xz.col(j).array().square().matrix()).array();
//							TMatrix LR = L.replicate(1, Xz.rows()).transpose();
//							TMatrix CL = (2 * (Xz.col(j) * Xz.col(j).transpose()).array()).matrix();
//							J.array() *= (zeta0.row(i))(j) * exp(((-((LR + CL).colwise() + L) / (QD1.row(i))(j)) - (((LR - CL).colwise() + L) / QD2(j))).array()).array();
//						}
//						double trace;
//						if (is_psd) { trace = (K.ldlt().solve(J)).trace(); }
//						else { trace = (K.colPivHouseholderQr().solve(J)).trace(); }
//						double Ialpha = (I.cwiseProduct(alpha)).array().sum();
//						pred_mu[i] = (Ialpha);
//						pred_var[i] = (abs(((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2)) + kernel.variance.value() * (1 + 1e-8 - trace))));
//					}
//				};
//				for (int s = 0; s < n_thread; ++s) { pool.push_task(task, int(s * split), int(s * split) + split, s); }														
//				pool.wait_for_tasks();
//				if (remainder > 0) { task(n_rows - remainder, n_rows, -1); }
//				output_mean.block(0, column, mean.rows(), 1) = pred_mu;
//				output_variance.block(0, column, variance.rows(), 1) = pred_var;
//				pool.reset();
//			}
//			output = std::make_pair(output_mean, output_variance);
//		}
//		#pragma endregion		
//		
//		void check_nodes() {
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{
//				if (node->inputs.size() == 0)
//				{
//					throw std::exception("A Node in the Layer has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
//				}
//				if (node->outputs.size() == 0)
//				{
//					throw std::exception("A Node in the Layer has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
//				}
//			}
//			nodes_checked = true;
//		}				
//
//	public:
//		Layer() = default;
//		Layer(std::vector<Node> nodes) : nodes(nodes) {}
//
//		void observed_inputs(const TMatrix& X) {
//			observed_input = X;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->inputs = X;
//			}
//		}
//		void observed_outputs(const TMatrix& Y, bool set_all_missing = false) {
//			BoolVector missing;
//			if (set_all_missing) {missing = BoolVector::Ones(Y.rows());}
//			else {
//				if ((Y.array().isNaN()).any())
//				{
//					missing = get_missing_index<BoolVector>(Y);
//				}
//				else { missing = BoolVector::Zero(Y.rows()); }
//			}
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->outputs = Y;
//				node->missing = missing;
//			}
//			observed_output = Y;
//		}
//		
//		double log_marginal_likelihood() {
//			double nll = 0;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{nll += node->log_marginal_likelihood();}
//			return nll;
//		}	
//		void estimate_parameters() {
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); i++) {
//				TMatrix history = nodes[i].get_parameter_history();
//				TVector theta = history.colwise().mean();
//				nodes[i].set_params(theta);
//			}
//		}
//		void update_cholesky() {
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{node->update_cholesky();}
//		}
//		std::vector<TVector> get_params() {
//			std::vector<TVector> params;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				params.push_back(node->get_params());
//			}
//			return params;
//		}		
//		
//		void train() {
//			if (!nodes_checked) { check_nodes(); }
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); i++) {
//				if (!nodes[i].store_parameters) { nodes[i].store_parameters = true; }
//				nodes[i].train();
//			}
//		}
//		
//		MatrixPair predict(const TMatrix& X) {
//			/*
//			* All nodes predict X and output a pair of N X M matrices
//			* Where N = number of rows X ; M = number of nodes in layer
//			* The pair is the mean and variance.
//			*/
//			TMatrix node_mu(X.rows(), nodes.size());
//			TMatrix node_var(X.rows(), nodes.size());
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
//			{
//				MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
//				node_mu.block(0, i, X.rows(), 1).noalias() = pred.first;
//				node_var.block(0, i, X.rows(), 1).noalias() = pred.second;
//			}
//
//			MatrixPair pred = std::make_pair(node_mu, node_var);
//			return pred;
//		}
//		void propagate(MatrixPair& prediction, const std::size_t& n_nodes) {
//			TMatrix mean = prediction.first;
//			TMatrix variance = prediction.second;
//			TMatrix output_mean = TMatrix::Zero(mean.rows(), n_nodes);
//			TMatrix output_variance = TMatrix::Zero(variance.rows(), n_nodes);
//			int column = 0;
//			bool is_psd = true;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				if (!(node->K.isApprox(node->K.transpose())) || node->chol.info() == Eigen::NumericalIssue){is_psd = false;}				
//				TVector pred_mu(mean.rows());
//				TVector pred_var(mean.rows());
//				node->kernel->expectations(mean, variance);
//				double trace;
//				for (Eigen::Index i = 0; i < mean.rows(); ++i) {
//					TMatrix I = TMatrix::Ones(node->inputs.rows(), 1);
//					TMatrix J = TMatrix::Ones(node->inputs.rows(), node->inputs.rows());
//					node->kernel->IJ(I, J, static_cast<TRVector>(mean.row(i)), node->inputs, i);
//					if (is_psd) { trace = (node->K.llt().solve(J)).trace(); }
//					else { trace = (node->K.colPivHouseholderQr().solve(J)).trace(); }
//					double Ialpha = (I.cwiseProduct(node->alpha)).array().sum();
//					pred_mu[i] = (Ialpha);
//					pred_var[i] = (abs(((node->alpha.transpose() * J).cwiseProduct(node->alpha.transpose()).array().sum() - (pow(Ialpha, 2)) + node->kernel->variance.value() * (1 + 1e-8 - trace))));
//				}
//				output_mean.block(0, column, mean.rows(), 1) = pred_mu;
//				output_variance.block(0, column, variance.rows(), 1) = pred_var;
//				column++;
//			}
//			prediction.first = output_mean;
//			prediction.second = output_variance;
//		}
//		std::vector<Node> nodes_() const { return nodes; }
//		friend class SIDGP;
//
//	private:
//		std::vector<Node> nodes;
//		TMatrix observed_input; 
//		TMatrix observed_output;
//		MatrixPair latent_input;
//		int n_thread = 1;
//		bool nodes_checked = false;
//	public:
//		int layer_number = 0;
//	};
//
//	class SIDGP {
//
//	private:
//		void initialize_layers() {
//
//			// Check First and last layer contains observed input/output
//			if (layers.front().observed_input.size() == 0) { throw std::exception("First Layer requires Observed Inputs"); }
//			if (layers.back().observed_output.size() == 0) { throw std::exception("First Layer requires Observed Outputs"); }
//			// Fill first layer outputs
//			if (layers.front().observed_output.size() == 0) {
//				if (layers.front().observed_input.cols() == layers.front().nodes.size())
//				{layers.front().observed_outputs(layers.front().observed_input, true);}
//				// Add Kernel PCA
//				// Add Dimension Expansion
//			}
//			layers.front().layer_number = 1;
//			layers.front().update_cholesky();
//			for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
//				layer->layer_number = std::prev(layer)->layer_number + 1;
//				if (std::prev(layer)->nodes.size() == layer->nodes.size()) 
//				{ layer->observed_inputs(std::prev(layer)->observed_output);}
//				// Add Kernel PCA
//				// Add Dimension Expansion
//				if (layer->observed_output.size() > 0)
//				{
//					const Eigen::Index missing_count = layer->nodes.at(0).missing.count();
//					const std::size_t missing_size = layer->nodes.at(0).missing.size();
//					// Some but not all observed outputs are NAN; Fill with CMVN
//					//if (missing_count > 0 && missing_count < missing_size)
//				}
//				else { layer->observed_outputs(layer->observed_input, true); }	
//				layer->update_cholesky();
//			}
//		}
//		void sample_layers(int burn_in = 1)
//		{
//			for (int i = 0; i < burn_in; ++i) {
//				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
//					for (std::vector<Node>::iterator node = layer->nodes.begin(); node != layer->nodes.end(); ++node) {
//						node->sample(std::next(layer)->nodes);
//					}
//				}
//			}
//		}
//
//		#pragma region PRINT_UTILITIES
//		void print_utility(int arg, int idx) {
//			if (arg == 1) {
//				std::cout << std::setw(3) << std::left << idx + 1 << std::setw(2) << std::right << " : ";
//			}
//		}
//		void print_utility(std::size_t idx) {
//			if (idx == 0) { std::cout << "LAYER " << idx + 1 << " -> " << "LAYER " << idx + 2 << " -> "; }
//			else if (idx == layers.size() - 2) { std::cout << "LAYER " << layers.size() << "\r" << std::flush; }
//			else { std::cout << "LAYER " << idx + 2 << " -> "; }
//		}
//		void print_utility(int idx, double& progress, double obj_fxn) {
//			std::cout << std::setw(3) << std::left << progress << std::setw(5) << std::left << " % |";
//			std::cout << std::setw(7) << std::left << " LAYER "
//					<< std::setw(3) << std::left << idx
//					<< std::setw(10) << std::right << std::setprecision(4) << std::fixed << obj_fxn << "\r" << std::flush;
//		}
//		void print_utility(int idx, double& progress) {
//			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
//			std::cout << std::setw(7) << std::left << " LAYER " << std::setw(3) << std::left << idx << "\r" << std::flush;
//		}
//		#pragma endregion
//	
//	public:
//
//		SIDGP(std::vector<Layer>& layers) : layers(layers)
//		{
//			initialize_layers();
//			sample_layers(10);
//		}
//
//		void train(int iter = 500, int ess_burn = 10) {
//			train_iter = iter;
//			for (int i = 0; i < train_iter; ++i) {
//				double progress = double(i) * 100.0 / double(iter);
//				// I-step
//				sample_layers(ess_burn);
//				// M-step
//				for (std::vector<Layer>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
//					layer->train();
//					print_utility(layer->layer_number,  progress);
//				}
//			}
//			// Estimate Point Parameters and Assign to Nodes
//			for (std::vector<Layer>::size_type j = 0; j != layers.size(); ++j) {
//				layers[j].estimate_parameters();
//			}
//		}
//
//		MatrixPair predict(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
//			sample_layers(50);
//			const std::size_t n_layers = layers.size();
//			TMatrix mean = TMatrix::Zero(X.rows(), 1);
//			TMatrix variance = TMatrix::Zero(X.rows(), 1);
//			MatrixPair pred;
//			std::vector<MatrixPair> predictions;
//			for (int i = 0; i < n_impute; ++i) {
//				print_utility(1, i);
//				sample_layers();
//				for (std::vector<Layer>::size_type j = 0; j != layers.size() - 1; ++j) {
//					print_utility(j);
//					if (j == 0) { pred = layers[j].predict(X); }
//					else
//					{
//						layers[j].n_thread = n_thread;
//						layers[j].propagate(pred, layers[j+1].nodes.size());
//					}
//				}
//				predictions.push_back(pred);
//			}
//
//			TMatrix output_mean = TMatrix::Zero(X.rows(), n_impute);
//			TMatrix output_var = TMatrix::Zero(X.rows(), n_impute);
//			for (int k = 0; k < predictions.size(); ++k)
//			{
//				output_mean.block(0, k, X.rows(), 1) = predictions[k].first;
//				output_var.block(0, k, X.rows(), 1) = predictions[k].second;
//			}
//			mean = output_mean.rowwise().mean();
//			variance = (square((output_mean).array()).matrix() + (output_var)).rowwise().mean();
//
//			//mean.noalias() += pred.first;
//			//variance.noalias() += (square((pred.first).array()).matrix() + (pred.second));
//
//			std::system("cls");
//			//mean.array() /= n_impute;
//			//variance.array() /= n_impute;
//			variance.array() -= square(mean.array());
//			return std::make_pair(mean, variance);
//		}
//
//		//MatrixPair predict_(const TMatrix& X, int n_impute = 50, int n_thread = 2) {
//		//	sample_layers(50);
//		//	const std::size_t n_layers = layers.size();
//		//	TMatrix mean = TMatrix::Zero(X.rows(), 1);
//		//	TMatrix variance = TMatrix::Zero(X.rows(), 1);
//		//	for (int i = 0; i < n_impute; ++i) {
//		//		print_utility(1, i);
//		//		sample_layers();
//		//		for (std::vector<Layer>::size_type j = 0; j != layers.size()-1; ++j) {
//		//			print_utility(j);
//		//			if (j == 0) { layers[j+1].latent_input = layers[j].predict(X); }
//		//			else 
//		//			{ 
//		//				layers[j]._mode = 1;
//		//				layers[j].n_thread = n_thread;
//		//				layers[j](layers[j + 1]);						
//		//			}
//		//			if (j == layers.size() - 2) {
//		//				mean.noalias() += layers[j + 1].latent_input.first;
//		//				variance.noalias() += ( square((layers[j + 1].latent_input.first).array()).matrix() + (layers[j + 1].latent_input.second) );
//		//			}
//		//		}				
//		//	}
//		//	std::system("cls");
//		//	mean.array() /= n_impute;
//		//	variance.array() /= n_impute;
//		//	variance.array() -= square(mean.array());
//		//	return std::make_pair(mean, variance);
//		//}
//
//		// Python Interface
//		std::vector<Layer> layers_() const { return layers; }
//
//	private:
//		std::vector<Layer> layers;
//
//	public:
//		int train_iter = 500;
//
//
//	};
//
//}
//namespace SMh::deep_models::dump {
//
//	using namespace SMh::kernels;
//	using namespace SMh::utilities;
//	using namespace SMh::base_models::gaussian_process;
//
//	class Node : public GPR {
//
//	public:
//		Node(shared_ptr<Kernel> kernel) : GPR(kernel) {}
//		void train() {
//			GPR::train();
//			NLL = -objective_fxn();
//			if (store_parameters) { _history.push_back(get_params()); }
//		}
//		TMatrix get_parameter_history() {
//			if (_history.size() == 0) { throw std::runtime_error("No Parameters Saved, set store_parameters = true"); }
//			Eigen::Index param_size = params_size();
//			TMatrix history(_history.size(), param_size);
//			for (std::vector<TVector>::size_type i = 0; i != _history.size(); ++i) {
//				history.row(i) = _history[i];
//			}
//			return history;
//		}
//
//		bool store_parameters = false;
//		double NLL = std::numeric_limits<double>::infinity();
//	private:
//		std::vector<TVector> _history;
//	};
//
//	class OldL {
//
//	private:
//		void conditional_mvn(GPR& node, TMatrix& mu, TMatrix& var) {
//			TMatrix X1 = mask_matrix(node.inputs, node.missing, false, 0);
//			TMatrix W1 = mask_matrix(node.inputs, node.missing, true, 0);
//			TMatrix W2 = mask_matrix(node.outputs, node.missing, true, 0);
//			TMatrix R = node.kernel->K(W1, W1);
//			TMatrix c = node.kernel->K(X1, X1, node.likelihood_variance.value());
//			TMatrix r = node.kernel->K(W1, X1);
//			TLLT chol = R.llt();
//			TMatrix alpha = chol.solve(r); // Rinv_r = np.linalg.solve(R, r)
//			TMatrix beta = (r.transpose() * alpha); // r_Rinv_r = r.T @ Rinv_r
//			TMatrix tmp(alpha.rows(), alpha.cols());
//			visit_lambda(alpha, [&tmp, &W2](double v, int i, int j) { tmp(i, j) = W2(i) * v; });
//			mu.resize(alpha.rows(), 1);
//			mu = tmp.colwise().sum().transpose();
//			var = (node.kernel->variance.value() * (c - beta)).cwiseAbs();
//
//		}
//		TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
//			return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
//		}
//
//		void sample(OldL& next_layer) {
//			TMatrix mean, K, nu;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{
//				if (!(node->missing.count() > 0)) { continue; }
//				if (node->missing.count() == node->missing.size()) {
//					mean = TVector::Zero(node->inputs.rows());
//					K = node->kernel->K(node->inputs, node->inputs, node->likelihood_variance.value());
//				}
//				else { conditional_mvn(*node, mean, K); }
//				nu = sample_mvn(mean, K);
//				double log_y = next_layer.objective_fxn(true);
//				log_y += log(random_uniform(0.0, 1.0));
//				double params = random_uniform(0.0, 2 * PI);
//				double params_min = params - 2 * PI;
//				double params_max = params;
//
//				while (true) {
//					TMatrix fp = update_f(node->outputs, nu, mean, params);
//					next_layer.set_inputs(fp);
//					double log_yp = next_layer.objective_fxn(true);
//
//					if (log_yp > log_y) { node->outputs = fp; return; }
//					else {
//						if (params < 0) { params_min = params; }
//						else { params_max = params; }
//						params = random_uniform(params_min, params_max);
//					}
//				}
//			}
//		}
//		void set_inputs(const TMatrix& X) {
//			if (_inputs.first.size() > 0) { _inputs.first = X; }
//			else { _inputs.second = X; }
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->inputs = X;
//			}
//		}
//		void set_outputs(const TMatrix& Y) {
//			if (_outputs.first.size() > 0) { _outputs.first = Y; }
//			else { _outputs.second = Y; }
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->outputs = Y;
//			}
//		}
//		void latent_inputs(const TMatrix& X) {
//			// Applies X to all Nodes
//			_inputs.second = X;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->inputs = X;
//			}
//		}
//		void latent_outputs(const TMatrix& Y) {
//			// Applies Y to all nodes
//			BoolVector missing;
//			// Initialize All Missing If Latent
//			if (_outputs.second.size() == 0)
//			{
//				missing = BoolVector::Ones(Y.rows());
//			}
//			// If Any NAN in latent outputs
//			else if (_outputs.second.size() > 0 && (Y.array().isNaN()).any())
//			{
//				missing = get_missing_index<BoolVector>(Y);
//			}
//			else { missing = BoolVector::Zero(Y.rows()); }
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->outputs = Y;
//				node->missing = missing;
//			}
//			_outputs.second = Y;
//		}
//		void check_nodes() {
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{
//				if (node->inputs.size() == 0)
//				{
//					throw std::exception("A Node in the OldL has no Inputs. Either provide Observed Inputs, or pass through a Model for Latent Inputs"); break;
//				}
//				if (node->outputs.size() == 0)
//				{
//					throw std::exception("A Node in the OldL has no Outputs. Either provide Observed Outputs, or pass through a Model for Latent Outputs"); break;
//				}
//			}
//			nodes_checked = true;
//		}
//
//	public:
//		OldL() = default;
//		OldL(std::vector<Node> nodes) : nodes(nodes) {}
//
//		void observed_inputs(const TMatrix& X) {
//			_inputs.first = X;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->inputs = X;
//			}
//		}
//		void observed_outputs(const TMatrix& Y) {
//			BoolVector missing;
//			if ((Y.array().isNaN()).any())
//			{
//				missing = get_missing_index<BoolVector>(Y);
//			}
//			else { missing = BoolVector::Zero(Y.rows()); }
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				node->outputs = Y;
//				node->missing = missing;
//			}
//			_outputs.first = Y;
//		}
//		std::vector<TVector> get_params() {
//			std::vector<TVector> params;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
//				params.push_back(node->get_params());
//			}
//			return params;
//		}
//		double objective_fxn(bool compute = false) {
//			double nll = 0;
//			for (std::vector<Node>::iterator node = nodes.begin(); node != nodes.end(); ++node)
//			{
//				if (compute) { nll += node->objective_fxn(); }
//				else { nll += node->NLL; }
//			}
//			return nll;
//		}
//		void estimate_parameters() {
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); i++) {
//				TMatrix history = nodes[i].get_parameter_history();
//				TVector theta = history.colwise().mean();
//				nodes[i].set_params(theta);
//				double nll = -nodes[i].objective_fxn();
//			}
//		}
//
//		void train() {
//			if (!nodes_checked) { check_nodes(); }
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); i++) {
//				if (!nodes[i].store_parameters) { nodes[i].store_parameters = true; }
//				nodes[i].train();
//			}
//		}
//		MatrixPair predict(const TMatrix& X) {
//			/*
//			* All nodes predict X and output a pair of N X M matrices
//			* Where N = number of rows X ; M = number of nodes in layer
//			* The pair is the mean and variance.
//			*/
//			TMatrix node_mu(X.rows(), nodes.size());
//			TMatrix node_var(X.rows(), nodes.size());
//			for (std::vector<Node>::size_type i = 0; i != nodes.size(); ++i)
//			{
//				MatrixPair pred = std::get<MatrixPair>(nodes[i].predict(X, true));
//				node_mu.block(0, i, X.rows(), 1).noalias() = pred.first;
//				node_var.block(0, i, X.rows(), 1).noalias() = pred.second;
//			}
//
//			MatrixPair pred = std::make_pair(node_mu, node_var);
//			return pred;
//		}
//
//		// Python Interface
//		std::vector<Node> nodes() const { return nodes; }
//		MatrixPair inputs() const { return _inputs; }
//		MatrixPair outputs() const { return _outputs; }
//
//		friend class SIDGP;
//
//	private:
//		std::vector<Node> nodes;
//		MatrixPair _inputs;  // (Observed, Latent)
//		MatrixPair _outputs; // (Observed, Latent)
//		MatrixPair latent_input;
//		MatrixPair latent_output;
//
//		bool nodes_checked = false;
//	public:
//		int layer_number = 0;
//	};
//
//	// TODO: Consider Connect Feature, Possible Simulataneous (Observed, Latent) Existence
//	class OldSI {
//
//	private:
//		void initialize_layers() {
//			TMatrix tmp_out;
//			TMatrix tmp_in = observed_inputs;
//
//			layers.front().observed_inputs(observed_inputs);
//			layers.back().observed_outputs(observed_outputs);
//			int layer_number = 1;
//			for (std::vector<OldL>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
//				layer->layer_number = layer_number;
//				if (observed_inputs.cols() == layer->nodes.size()) { tmp_out = tmp_in; }
//				// Add Kernel PCA
//				// Add Dimension Expansion
//
//				// ======================= Inputs ======================= //
//				if (layer == layers.begin()) {}
//				else { layer->latent_inputs(tmp_in); }
//
//				// ====================== Outputs ====================== //
//				// Observed Outputs Given
//				if (layer->_outputs.first.size() > 0) {
//					const Eigen::Index missing_count = layer->nodes.at(0).missing.count();
//					const std::size_t missing_size = layer->nodes.at(0).missing.size();
//					// If all NAN (i.e. missing) Set to Latent Output
//					if (missing_count == missing_size)
//					{
//						layer->_outputs.first.resize(0, 0);
//						layer->latent_outputs(tmp_out);
//					}
//					// Some but not all observed outputs are NAN; Fill with CMVN
//					else if (missing_count > 0 && missing_count < missing_size)
//					{
//						/*
//						*
//						if l==self.n_layer-1:
//							m,v=cmvn(kernel.last_layer_input, kernel.last_layer_global_input, self.Y[l][:,[k]], kernel.scale,kernel.length,kernel.nugget,kernel.name,kernel.missingness)
//						else:
//							m,v=cmvn(kernel.input,kernel.global_input,self.Y[l][:,[k]],kernel.scale,kernel.length,kernel.nugget,kernel.name,kernel.missingness)
//						samp=copy.deepcopy(self.Y[l][:,[k]])
//						samp[kernel.missingness,0]=np.random.default_rng().multivariate_normal(mean=m,cov=v,check_valid='ignore')
//						if l==self.n_layer-1:
//							kernel.output=copy.deepcopy(samp[~kernel.missingness,:])
//						else:
//							kernel.output=copy.deepcopy(samp)
//						Out[:,[k]]=samp
//						*/
//
//					}
//					else { tmp_out = layer->_outputs.first; }
//				}
//				else { layer->latent_outputs(tmp_out); }
//				tmp_in = tmp_out;
//				layer_number++;
//			}
//		}
//
//		void sample_layers(int burn_in = 1)
//		{
//			for (int i = 0; i < burn_in; ++i) {
//				bool last_hidden_layer = false; // TODO
//				for (std::vector<OldL>::iterator layer = layers.begin(); layer != layers.end() - 1; ++layer) {
//					layer->sample(*std::next(layer));
//				}
//			}
//		}
//
//		void print_utility(int arg, int index = -1, int layer_number = -1, double layer_obj = -1.0) {
//			if (arg == 0) {
//				std::string hd(60, '=');
//				position_print(Justified::CENTRE, "DEEP GAUSSIAN PROCESS: STOCHASTIC IMPUTATION", 60);
//				std::cout << hd << std::endl;
//				std::cout << std::endl;
//			}
//			else if (arg == 1) {
//				std::string hd(30, '-');
//				position_print(Justified::CENTRE, "IMPUTATION STEP", 30, index);
//				std::cout << hd << std::endl;
//			}
//			else if (arg == 2) {
//				if (layer_number == 1) {
//					std::cout
//						<< std::setw(10) << std::left << "LAYER"
//						<< std::setw(15) << std::right << "NLL"
//						<< std::endl;
//				}
//				std::cout
//					<< std::setw(10) << std::left << layer_number
//					<< std::setw(15) << std::right << layer_obj
//					<< std::endl;
//			}
//			else if (arg == -1) {
//				std::system("cls");
//			}
//			else {}
//
//		}
//
//	public:
//
//		OldSI(std::vector<OldL>& layers, const TMatrix& inputs, const TMatrix& outputs) : layers(layers), observed_inputs(inputs), observed_outputs(outputs)
//		{
//			initialize_layers();
//			sample_layers(10);
//		}
//
//		void train(int iter = 500, int ess_burn = 10) {
//			train_iter = iter;
//			for (int i = 0; i < train_iter; ++i) {
//				print_utility(1, i + 1);
//				// I-step
//				sample_layers(ess_burn);
//				// M-step
//				for (std::vector<OldL>::iterator layer = layers.begin(); layer != layers.end(); ++layer) {
//					layer->train();
//					print_utility(2, -1, layer->layer_number, layer->objective_fxn());
//				}
//				print_utility(-1);
//			}
//			// Estimate Point Parameters and Assign to Nodes
//			for (std::vector<OldL>::size_type j = 0; j != layers.size(); ++j) {
//				layers[j].estimate_parameters();
//			}
//		}
//
//		void predict(const TMatrix& X, int n_impute = 50, bool return_var = false) {
//			sample_layers(50);
//			const std::size_t n_layers = layers.size();
//			for (int i = 0; i < n_impute; ++i) {
//				sample_layers();
//				std::vector<MatrixPair> layer_pred(n_layers);
//				//layer_pred.push_back(layers.at(0).predict(X, true));
//				for (std::vector<OldL>::size_type j = 1; j != layers.size(); ++j) {
//					MatrixPair latent_output = layer_pred.at(j - 1);
//					//layers.at(j).propagate(latent_output);
//					//layer_pred.push_back(std::get<MatrixPair>(layers.at(j).predict(latent_input.first, latent_input.second)));
//				}
//			}
//		}
//
//		// Python Interface
//		std::vector<OldL> layers() const { return layers; }
//
//	private:
//		std::vector<OldL> layers;
//
//	public:
//		TMatrix observed_inputs;
//		TMatrix observed_outputs;
//		int train_iter = 500;
//
//
//	};
//
//}
#endif

#endif