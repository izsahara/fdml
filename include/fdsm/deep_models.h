#ifndef DEEPMODELS_H
#define DEEPMODELS_H

#include "./utilities.h"
#include "./kernels.h"
#include "./base_models.h"
#include <filesystem>

namespace fdsm::deep_models {}

namespace fdsm::deep_models::gaussian_process {
	using namespace fdsm::kernels;
	using namespace fdsm::utilities;
	using namespace fdsm::base_models::gaussian_process;

	class Layer {
	private:
		void check_nodes() {
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node)
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
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				TMatrix history = node->get_parameter_history();
				TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
				node->set_params(theta);
			}
		}
	public:

		Layer(std::vector<GPNode>& nodes_) {
			// Checks
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
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
			nodes = nodes_;
		}
		// Operator() overload
		Layer& operator()(Layer& layer) {
			// Initialize			[ CurrentLayer(NextLayer) ]
			if (state_ == 0) {
				layer.index = index + 1;
				if (layer.nodes.size() == nodes.size())
				{
					layer.set_inputs(o_output);
					if (layer.o_output.size() == 0)
					{
						layer.set_outputs(o_output, true);
					}
				}
				else if (layer.nodes.size() < nodes.size())
				{
					if (layer.last_layer) {layer.set_inputs(o_output);}
					else {
						// Apply Dimensionality Reduction (Kernel PCA)
						kernel_pca::KernelPCA pca(layer.nodes.size(), "sigmoid");
						TMatrix input_transformed = pca.transform(o_output);
						layer.set_inputs(input_transformed);
						if (layer.o_output.size() == 0)
						{
							layer.set_outputs(input_transformed, true);
						}
					}
				}
				else
				{
					// Dimension Expansion
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

		// Main Functions
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

		void reconstruct_observed(const TMatrix& inputs, const TMatrix& outputs) {
			o_input = inputs;
			o_output = outputs;
		}


		// Setters
		void set_inputs(const TMatrix& inputs) {
			o_input = inputs;
			for (std::vector<GPNode>::iterator node = nodes.begin(); node != nodes.end(); ++node) {
				node->inputs = inputs;
			}
		}
		void set_outputs(const TMatrix& outputs, bool latent = false) {
			// NOTE: RECONSTRUCT ARG IS ONLY USED IN PYTHON INTERFACE
			// TODO: HIDE RECONSTRUCT ARG FROM USER

			BoolVector missing;
			if (latent) { missing = BoolVector::Ones(outputs.rows()); }
			else { missing = BoolVector::Zero(outputs.rows()); }
			if ((outputs.array().isNaN()).any())
			{
				missing = get_missing_index<BoolVector>(outputs);
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
		bool last_layer = false;
		int state_ = 0;
		int n_thread = 0;

	};


	TMatrix update_f(const TMatrix& f, const TMatrix& nu, const TMatrix& mean, const double& params) {
		return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	}

	TMatrix sample_mvn_(const TMatrix& K) {
		MVN sampler(K);
		return sampler();
		//Vmt19937_64 gen_eigen;
		//TVector mean = TVector::Zero(K.rows());
		//MvNormalGen sampler = makeMvNormalGen(mean, K);
		//return sampler.generate(gen_eigen);
	}

	double log_likelihood(const TMatrix& K, const TMatrix& outputs) {
		TLLT chol(K);
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
		double lml = -0.5 * (logdet + quad);
		return lml;
	}

	void one_sample(GPNode& target, Layer& linked, const std::size_t& node_idx) {
		const TMatrix X = target.inputs;
		const TMatrix f = target.outputs;

		if (target.missing.count() == 0) { return; }
		TMatrix K = target.kernel->K(X, X, target.likelihood_variance.value());
		K *= target.scale.value();
		TMatrix nu = sample_mvn_(K);
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

			for (std::size_t n = 0; n < linked.nodes.size(); ++n) {
				linked.nodes[n].inputs.col(node_idx) = fp;
				const TMatrix W2 = linked.nodes[n].inputs;
				const TMatrix Y2 = linked.nodes[n].outputs;
				TMatrix Kw2 = linked.nodes[n].kernel->K(W2, W2, linked.nodes[n].likelihood_variance.value());
				Kw2 *= linked.nodes[n].scale.value();
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


	class SIDGP {

	private:

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
		void print_utility(int idx, double& progress, std::string message, double& time) {
			std::cout << std::setw(3) << std::left << std::setprecision(1) << std::fixed << progress << std::setw(5) << std::left << " % |";
			std::cout << std::setw(7) << std::left << message << std::setw(3) << std::left << idx << std::setw(3) << std::left << time << std::endl;
			//std::cout << std::setw(7) << std::left << message << std::setw(3) << std::left << idx << std::setw(3) << std::left << time << "\r" << std::flush;
		}

		void initialize_layers() {
			if (layers.front().o_input.size() == 0) { throw std::runtime_error("First Layer Requires Observed Inputs"); }
			if (layers.back().o_output.size() == 0) { throw std::runtime_error("Last Layer Requires Observed Outputs"); }
			if (layers.back().nodes.size() != 1) { throw std::runtime_error("Last Layer Must Only have 1 Node for a Single Output"); }
			layers.front().index = 1;
			layers.back().last_layer = true;
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
					for (std::size_t n = 0; n < layer->nodes.size(); ++n) {
						one_sample(layer->nodes[n], *std::next(layer), n);	
					}

					//for (std::vector<GPNode>::iterator node = layer->nodes.begin(); node != layer->nodes.end(); ++node){
					//	one_sample(*node, *std::next(layer));
					//}
				}
			}
		}
	public:
		SIDGP(std::vector<Layer>& layers, bool initialize = true) : layers(layers) {
			if (initialize) {
				initialize_layers();
				sample(10);
			}
		}

		void train(int n_iter = 50, int ess_burn = 10) {
			n_iter_ = n_iter;
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
			std::system("clear");
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
			double time = 0.0;
			for (int i = 0; i < n_impute + 1; ++i) {
				clock_t start = clock();
				double progress = double(i) * 100.0 / double(n_impute);
				print_utility(i, progress, " N_IMPUTE ", time);
				sample();
				layers.front().predict(X);
				std::size_t j = 1;
				for (std::vector<Layer>::iterator layer = layers.begin() + 1; layer != layers.end(); ++layer) {
					if (layer->state_ != 2) { layer->state_ = 2; }
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
				clock_t end = clock();
				time = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
			}
			std::cout << std::endl;
			mean.array() /= double(n_impute);
			variance.array() /= double(n_impute);
			variance.array() -= square(mean.array());

			std::system("clear");
			return std::make_pair(mean, variance);
		}


		void set_layers(std::vector<Layer>& layers_) { layers = std::move(layers_); }
		const std::vector<Layer> get_layers() const { return layers; }

		const int n_iterations() const { return n_iter_; }

	private:
		int n_iter_ = 0;
		std::vector<Layer> layers;
	};

}


#endif