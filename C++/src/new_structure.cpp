// #pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#include <fdml/utilities.h>
#include <fdml/kernels.h>
#include <fdml/base_models.h>
#include <fdml/optimizers.h>
#include <chrono>
#include <rapidcsv.h>

using namespace fdml::kernels;
using namespace fdml::utilities;
using namespace fdml::base_models;
using namespace fdml::base_models::gaussian_process;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, "\t", "\n");
template <typename Derived>
void write_data(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

TMatrix read_data(std::string filename) {

	rapidcsv::Document doc(filename, rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams('\t'));
	int nrows = doc.GetRowCount();
	int ncols = doc.GetColumnCount();

	TMatrix data(nrows, ncols);
	for (std::size_t i = 0; i < nrows; ++i) {
		std::vector<double> row = doc.GetRow<double>(i);
		for (std::vector<double>::size_type j = 0; j != row.size(); j++) {
			data(i, j) = row[j];
		}
	}
	return data;
}

// Move to kernels.h
enum TKernel { TSquaredExponential, TMatern52 };
// Move to optimizers.h
enum TSolver { TLBFGSB, TPSO };

enum TLayer  { TInput, TInputConnected, THidden, TObserved, TUnchanged };
enum Task { Init, Train, LinkedPredict };
using KernelPtr = std::shared_ptr<Kernel>;
using SolverPtr = std::shared_ptr<Solver>;

using fdml::optimizers::LBFGSB;
using fdml::optimizers::PSO;
static std::mt19937_64 rng(std::random_device{}());

class ProgressBar
{
	static const auto overhead = sizeof " [100%]";

	std::ostream& os;
	const std::size_t bar_width;
	std::string message;
	const std::string full_bar;

public:
	ProgressBar(std::ostream& os, std::size_t line_width,
		std::string message_, const char symbol = '|')
		: os{ os },
		bar_width{ line_width - overhead },
		message{ std::move(message_) },
		full_bar{ std::string(bar_width, symbol) + std::string(bar_width, ' ') }
	{
		if (message.size() + 1 >= bar_width || message.find('\n') != message.npos) {
			os << message << '\n';
			message.clear();
		}
		else {
			message += ' ';
		}
		write(0.0);
	}

	// not copyable
	ProgressBar(const ProgressBar&) = delete;
	ProgressBar& operator=(const ProgressBar&) = delete;

	~ProgressBar()
	{
		write(1.0);
		os << '\n';
	}

	void write(double fraction) {
		// clamp fraction to valid range [0,1]
		if (fraction < 0)
			fraction = 0;
		else if (fraction > 1)
			fraction = 1;

		auto width = bar_width - message.size();
		auto offset = bar_width - static_cast<unsigned>(width * fraction);

		os << '\r' << message;
		os.write(full_bar.data() + offset, width);
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " << std::flush;
	}
	void write(double fraction, double nrmse) {
		// clamp fraction to valid range [0,1]
		if (fraction < 0)
			fraction = 0;
		else if (fraction > 1)
			fraction = 1;

		auto width = bar_width - message.size();
		auto offset = bar_width - static_cast<unsigned>(width * fraction);

		os << '\r' << message;
		os.write(full_bar.data() + offset, width);
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " 
		   << " [" << std::setw(3) << std::left << std::setprecision(5) << std::fixed << nrmse * 100.0 << "%] "
		   << std::flush;
	}
};

//
class Node : public GP {
private:
	Node& evalK() {
		K = kernel->K(inputs, inputs, likelihood_variance.value());
		K.array() *= scale.value();
		return *this;
	}
	Node& evalK(const TMatrix& X) {
		TMatrix tmp(inputs.rows(), inputs.cols() + X.cols());
		tmp << inputs, X;
		K.noalias() = kernel->K(tmp, tmp, likelihood_variance.value());
		K.array() *= scale.value();
		return *this;
	}
	TMatrix sample_mvn() {
		TVector mean = TVector::Zero(K.rows());
		Eigen::setNbThreads(1);
		Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(K);
		TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
		std::normal_distribution<> dist;
		return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {return dist(rng); });
	}
	double log_likelihood() {
		chol = K.llt();
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
		double lml = -0.5 * (logdet + quad);
		return lml;
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
	void get_bounds(TVector& lower, TVector& upper) {
		kernel->get_bounds(lower, upper, false);

		if (!(*likelihood_variance.is_fixed)) {
			lower.conservativeResize(lower.rows() + 1);
			upper.conservativeResize(upper.rows() + 1);
			lower.tail(1)(0) = likelihood_variance.get_bounds().first;
			upper.tail(1)(0) = likelihood_variance.get_bounds().second;
		}
	}
	void train() override {
		TVector lower_bound, upper_bound;
		get_bounds(lower_bound, upper_bound);
		TVector theta = get_params();
		if (solver->from_optim) {
			auto objective = [this](const TVector& x, TVector* grad, void* opt_data)
			{return objective_(x, grad, nullptr, opt_data); };
			opt::OptimData optdata;
			solver->solve(theta, objective, optdata);
		}
		else {
			// LBFGSB/ Rprop
			Objective objective(this, static_cast<int>(lower_bound.size()));
			objective.set_bounds(lower_bound, upper_bound);
			solver->solve(theta, objective);
			theta = objective.Xopt;
		}
		set_params(theta);
		if (store_parameters) history.push_back(theta);
	}
	TVector gradients() override {
		auto log_prior_gradient = [=]() {
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
		};
		std::vector<TMatrix> grad_;
		if (alpha.size() == 0) { update_cholesky(); }
		// Kernel Derivatives
		kernel->fod(inputs, grad_);
		// TVector grad(grad_.size());
		TVector grad = TVector::Zero(grad_.size());
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
		// Add likelihood_variance/nugget gradient
		grad -= log_prior_gradient();
		return grad;
	}
	void set_params(const TVector& new_params) override
	{
		kernel->set_params(new_params);
		if (!(*likelihood_variance.is_fixed)) likelihood_variance.transform_value(new_params.tail(1)(0));
		update_cholesky();
	}
	TVector get_params(bool inverse_transform = true) override {
		TVector params = kernel->get_params(inverse_transform);
		if (!(*likelihood_variance.is_fixed)) {
			likelihood_variance.transform_value(inverse_transform);
			params.conservativeResize(params.rows() + 1);
			params.tail(1)(0) = likelihood_variance.value();
		}
		return params;
	}
	MatrixPair predict(const TMatrix& X)
	{
		update_cholesky();
		TMatrix Ks(inputs.rows(), X.rows());
		Ks.noalias() = kernel->K(inputs, X);
		TMatrix mu = Ks.transpose() * alpha;
		TMatrix Kss = kernel->diag(X);
		TMatrix V = chol.solve(Ks);
		TMatrix var = abs((scale.value() * (Kss - (Ks.transpose() * V).diagonal()).array()));
		return std::make_pair(mu, var);
	}
	void linked_predict(const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {

		update_cholesky();
		const Eigen::Index nrows = linked.first.rows();
		kernel->expectations(linked.first, linked.second);
		if (n_thread == 1) {
			for (Eigen::Index i = 0; i < nrows; ++i) {
				TMatrix I = TMatrix::Ones(inputs.rows(), 1);
				TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
				kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
				double trace = (K.llt().solve(J)).trace();
				double Ialpha = (I.cwiseProduct(alpha)).array().sum();
				latent_mu[i] = (Ialpha);
				latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
			}
		}
		else {
			Eigen::initParallel(); // /openmp (MSVC) or -fopenmp (GCC) flag
			thread_pool pool;
			int split = int(nrows / n_thread);
			const int remainder = int(nrows) % n_thread;
			auto task = [=, &latent_mu, &latent_var](int begin, int end)
			{
				for (Eigen::Index i = begin; i < end; ++i) {
					TMatrix I = TMatrix::Ones(inputs.rows(), 1);
					TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
					kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
					double trace = (K.llt().solve(J)).trace();
					double Ialpha = (I.cwiseProduct(alpha)).array().sum();
					latent_mu[i] = (Ialpha);
					latent_var[i] = (abs((((alpha.transpose() * J).cwiseProduct(alpha.transpose()).array().sum() - (pow(Ialpha, 2))) + scale.value() * ((1.0 + likelihood_variance.value()) - trace))));
				}
			};
			for (int s = 0; s < n_thread; ++s) {
				pool.push_task(task, int(s * split), int(s * split) + split);
			}
			pool.wait_for_tasks();
			if (remainder > 0) {
				task(nrows - remainder, nrows);
			}
			pool.reset();
		}

	}

public:
	Node(double likelihood_variance = 1E-9) : GP(likelihood_variance) {}
	void set_kernel(const KernelPtr& rkernel) {
		kernel = std::move(rkernel);
	}
	void set_solver(const SolverPtr& rsolver) {
		solver = std::move(rsolver);
	}
private:
	friend class Layer;
	friend class SIDGP;
	unsigned int n_thread = 1;
	double objective_(const TVector& x, TVector* grad, TVector* hess, void* opt_data) {
		set_params(x);
		if (grad) { (*grad) = gradients() * 1.0; }
		return log_marginal_likelihood();
	}
	TMatrix get_parameter_history() {
		if (history.size() == 0) throw std::runtime_error("No Parameters Saved, set store_parameters = true");
		Eigen::Index param_size = get_params().size();
		TMatrix h(history.size(), param_size);
		for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
			h.row(i) = history[i];
		}
		return h;
	}
	void update_cholesky() {
		//K = kernel->K(inputs, inputs, D);
		//K.diagonal().array() += likelihood_variance.value();
		//chol = K.llt();
		//alpha = chol.solve(outputs);
		//if (!(*scale.is_fixed)) {
		//	scale = (outputs.transpose() * alpha)(0) / outputs.rows();
		//}
		TMatrix noise = TMatrix::Identity(inputs.rows(), outputs.rows());
		K = kernel->K(inputs, inputs, D);
		K += (noise * likelihood_variance.value());
		chol = K.llt();
		alpha = chol.solve(outputs);
		if (!(*scale.is_fixed)) {
			scale = (outputs.transpose() * alpha)(0) / outputs.rows();
		}
	}
public:
	Parameter<double> scale = { "scale", 1.0, "none" };
	bool store_parameters = true;
	std::vector<TVector> history;

protected:
	TVector  alpha;
	TLLT	 chol;
	TMatrix	 K;
	TMatrix	 D;

};

class Layer {
public:
	Layer() = default;
	Layer(const TLayer& layer_state, const unsigned int& n_nodes) : cstate(layer_state), n_nodes(n_nodes) { 
		m_nodes.resize(n_nodes);
		for (unsigned int nn = 0; nn < n_nodes; ++nn) {
			m_nodes[nn] = Node();
		}
	}
	
	void set_input(const TMatrix& input){
		if (cstate == TLayer::TInput) {
			observed_input.noalias() = input;
			set_output(input);
		}
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->inputs = input;
		}
	}
	void set_input(const TMatrix& input, const Eigen::Index& col) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->inputs.col(col) = input;
		}

	}
	void set_output(const TMatrix& output){

		if (!locked) { 
			if (cstate == TLayer::THidden)
			{
				cstate = TLayer::TObserved;
				ostate = TLayer::THidden;
			}
		}

		if (cstate == TLayer::TObserved) observed_output.noalias() = output;
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->outputs = output.col(nn - m_nodes.begin());
		}
	}
	
	TMatrix get_input() {
		if (cstate == TLayer::TInput) return observed_input;
		else return m_nodes[0].inputs;
	}
	TMatrix get_output() {
		if (cstate == TLayer::TObserved) return observed_output;
		else {
			TMatrix output(m_nodes[0].outputs.rows(), static_cast<Eigen::Index>(n_nodes));
			for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
				output.col(nn - m_nodes.begin()) = nn->outputs;
			}
			return output;
		}
	}

	void set_kernels(const TKernel& kernel) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			switch (kernel) {
				case TSquaredExponential:
					nn->set_kernel(KernelPtr(new SquaredExponential));
					continue;
				case TMatern52:
					nn->set_kernel(KernelPtr(new Matern52));
					continue;
				default:
					nn->set_kernel(KernelPtr(new SquaredExponential));
					continue;
			}
		}
	}
	void set_kernels(const TKernel& kernel, const TVector& length_scale) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			switch (kernel) {
			case TSquaredExponential:
				nn->set_kernel(KernelPtr(new SquaredExponential(length_scale, 1.0)));
				continue;
			case TMatern52:
				nn->set_kernel(KernelPtr(new Matern52(length_scale, 1.0)));
				continue;
			default:
				nn->set_kernel(KernelPtr(new SquaredExponential(length_scale, 1.0)));
				continue;
			}
		}
	}
	void set_solvers(const TSolver& solver) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			switch (solver) {
			case TLBFGSB:
				nn->set_solver(SolverPtr(new LBFGSB));
				continue;
			case TPSO:
				nn->set_solver(SolverPtr(new PSO));
				continue;
			default:
				nn->set_solver(SolverPtr(new LBFGSB));
				continue;
			}
		}
	}
	void fix_likelihood_variance() {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->likelihood_variance.fix();
		}
	}
	//
	void add_node(const Node& node) {
		if (locked) throw std::runtime_error("Layer Locked");
		std::vector<Node> intrm(n_nodes + 1);
		intrm.insert(intrm.end(), 
					 std::make_move_iterator(m_nodes.begin() + n_nodes), 
					 std::make_move_iterator(m_nodes.end()));
		intrm.back() = node;
		m_nodes = intrm;
		n_nodes += 1;
	}
	void remove_nodes(const unsigned int& xnodes) {
		if (locked) throw std::runtime_error("Layer Locked");
		if (xnodes > n_nodes) throw std::runtime_error("xnodes > n_nodes");
		m_nodes.erase(m_nodes.begin(), m_nodes.begin() + xnodes);
		n_nodes -= xnodes;
	}
	//


private:
	void train() {
		for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
			if (!node->store_parameters) { node->store_parameters = true; }
			node->train();
		}
	}
	void predict(const TMatrix& X) {
		TMatrix node_mu(X.rows(), static_cast<Eigen::Index>(n_nodes));
		TMatrix node_var(X.rows(), static_cast<Eigen::Index>(n_nodes));
		for (Eigen::Index i = 0; i < n_nodes; ++i)
		{
			MatrixPair pred = m_nodes[i].predict(X);
			node_mu.block(0, i, X.rows(), 1) = pred.first;
			node_var.block(0, i, X.rows(), 1) = pred.second;
		}
		latent_output = std::make_pair(node_mu, node_var);
	}
	void predict(const MatrixPair& linked) {
		latent_output = std::make_pair(
						TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)),
						TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)));
		for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
			Eigen::Index cc = static_cast<Eigen::Index>(node - m_nodes.begin());
			node->n_thread = n_thread;
			node->linked_predict(linked, latent_output.first.col(cc), latent_output.second.col(cc));
		}
	}
	void connect(const TMatrix& Ginput) {
		if (locked) throw std::runtime_error("Layer Locked");
		if (cstate != TLayer::TInputConnected)
		{
			ostate = TLayer::TInputConnected;
			std::swap(cstate, ostate);
		}
		observed_input = Ginput;
	}
	Layer& evalK()  {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			if (cstate == TLayer::TInputConnected) nn->evalK(observed_input);
			else nn->evalK();
		}
		return *this;
	}
	double log_likelihood() {
		double ll = 0.0;
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			ll += nn->log_likelihood();
		}
		return ll;
	}
	double log_likelihood(const TMatrix& input, const Eigen::Index& col) {
		double ll = 0.0;
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->inputs.col(col) = input;
			nn->evalK();
			ll += nn->log_likelihood();
		}
		return ll;
	}
	void estimate_parameters(const Eigen::Index& n_burn) {
		for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
			TMatrix history = node->get_parameter_history();
			TVector theta = (history.bottomRows(history.rows() - n_burn)).colwise().mean();
			node->set_params(theta);
		}
	}
public:
	TLayer cstate; // Current Layer State
	unsigned int n_nodes;
private:
	std::vector<Node> m_nodes;
	TLayer ostate = TLayer::TUnchanged; // Old Layer State;
	bool locked = false;
	unsigned int n_thread = 1;
	TMatrix observed_input;
	TMatrix observed_output;
	MatrixPair latent_output;
	friend struct Graph;
	friend class SIDGP;
};

struct Graph {
	unsigned int n_thread = 1;
	unsigned int n_hidden;
	const int n_layers;

	Graph(const MatrixPair& data, int n_hidden) : 
		n_hidden(n_hidden), n_layers(n_hidden + 2) 
	{
		unsigned int n_nodes = static_cast<unsigned int>(data.first.cols());
		m_layers.resize(n_layers);
		for (unsigned int ll = 0; ll < n_layers; ++ll) {
			if (ll == 0) {
				m_layers[ll] = Layer(TLayer::TInput, n_nodes);
				m_layers[ll].set_input(data.first);
			}
			else if (ll == n_layers - 1) {
				m_layers[ll] = Layer(TLayer::TObserved, static_cast<int>(data.second.cols()));
				m_layers[ll].set_output(data.second);
			}
			else {
				m_layers[ll] = Layer(TLayer::THidden, n_nodes);
			}
		}
	}

	const std::vector<Layer>::iterator layer(const int& ll) {
		if (ll > n_layers) throw std::runtime_error("index > n_layers");
		std::vector<Layer>::iterator lit = m_layers.begin();
		if (ll < 0) {
			return (lit + n_layers) + ll;
		}
		else {
			return lit + ll;
		}

		return lit;
	}
	const std::vector<Node>::iterator operator()(const unsigned int& nn, const unsigned int& ll) {
		if (ll > n_layers) throw std::runtime_error("index > n_layers");
		std::vector<Node>::iterator nit = m_layers[ll].m_nodes.begin() + nn;
		return nit;
	}
	void connect_inputs(const std::size_t& layer_idx) {
		m_layers[layer_idx].connect(m_layers[0].observed_input);
	}

private:
	std::vector<Layer> m_layers;
	
	void lock() {
		for (std::vector<Layer>::iterator ll = m_layers.begin(); ll != m_layers.end(); ++ll) {
			ll->locked = true;
		}
	}
	void propagate(const Task& task) {
		// cp : CurrentLayer(PreviousLayer)
		for (std::vector<Layer>::iterator cp = m_layers.begin() + 1; cp != m_layers.end(); ++cp) {
			switch (task) {
			case (Init):
				if (cp->n_nodes == std::prev(cp)->n_nodes) {
					cp->set_input(std::prev(cp)->get_output());
					cp->set_output(cp->get_input());
				}
				else if (cp->n_nodes < std::prev(cp)->n_nodes) {
					if ((cp - m_layers.begin()) == n_layers - 1) {
						// Output Layer
						cp->set_input(std::prev(cp)->get_output());
					}
					else {
						kernelpca::KernelPCA pca(cp->n_nodes, "sigmoid");
						cp->set_input(pca.transform(std::prev(cp)->get_output()));
						cp->set_output(cp->get_input());
					}
				}
				else {
					/* Dimension Expansion */
				}			
				continue;
			case (Train):
				if (cp == m_layers.begin() + 1) std::prev(cp)->train();
				cp->train();
				continue;
			case(LinkedPredict):
				cp->n_thread = n_thread;
				cp->predict(std::prev(cp)->latent_output);
				continue;
			}
		}
	}	
	friend class SIDGP;


};

class SIDGP  {
private:
	TMatrix update_f(const TMatrix& f, const TMatrix& nu, const double& params) {
		TVector mean = TVector::Zero(f.rows());
		return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	}
	void conditional_mvn(Node& node, TMatrix& mu, TMatrix& var) {
		//TMatrix X1 = operations::mask_matrix(node.inputs, node.missing, false, 0);
		//TMatrix W1 = operations::mask_matrix(node.inputs, node.missing, true, 0);
		//TMatrix W2 = operations::mask_matrix(node.outputs, node.missing, true, 0);
		//TMatrix R = node.kernel->K(W1, W1);
		//TMatrix c = node.kernel->K(X1, X1, node.likelihood_variance.value());
		//TMatrix r = node.kernel->K(W1, X1);
		//TLLT chol = R.llt();
		//TMatrix alpha = chol.solve(r); // Rinv_r = np.linalg.solve(R, r)
		//TMatrix beta = (r.transpose() * alpha); // r_Rinv_r = r.T @ Rinv_r
		//TMatrix tmp(alpha.rows(), alpha.cols());
		//operations::visit_lambda(alpha, [&tmp, &W2](double v, int i, int j) { tmp(i, j) = W2(i) * v; });
		//mu.resize(alpha.rows(), 1);
		//mu = tmp.colwise().sum().transpose();
		//var = (node.kernel->variance.value() * (c - beta)).cwiseAbs();
	}
	void sample(unsigned int n_burn = 1) {
		auto rand_u = [](const double& a, const double& b) {
			std::uniform_real_distribution<> uniform_dist(a, b);
			return uniform_dist(rng);
		};

		double log_y, theta, theta_min, theta_max;
		for (unsigned int nb = 0; nb < n_burn; ++nb) {
			for (std::vector<Layer>::iterator cl = graph.m_layers.begin(); cl != graph.m_layers.end() - 1; ++cl) {
				if (cl->cstate == TLayer::TObserved) continue; // missingness
				auto linked_layer = std::next(cl);
				for (std::vector<Node>::iterator cn = cl->m_nodes.begin(); cn != cl->m_nodes.end(); ++cn) {
					TMatrix nu = cn->evalK().sample_mvn();
					log_y = linked_layer->evalK().log_likelihood() + log(rand_u(0.0, 1.0));
					//
					if (!std::isfinite(log_y)) { throw std::runtime_error("log_y is not finite"); }
					//
					theta = rand_u(0.0, 2.0 * PI);
					theta_min = theta - 2.0 * PI;
					theta_max = theta;
					
					const Eigen::Index col = static_cast<Eigen::Index>((cn - cl->m_nodes.begin()));
					while (true) {
						TMatrix fp = update_f(cn->outputs, nu, theta);
						double log_yp = linked_layer->log_likelihood(fp, col);
						//
						if (!std::isfinite(log_yp)) { throw std::runtime_error("log_yp is not finite"); }
						//
						if (log_yp > log_y) { cn->outputs = fp; break; }
						else {
							if (theta < 0) { theta_min = theta; }
							else { theta_max = theta; }
							theta = rand_u(theta_min, theta_max);
						}
					}
				}
			}
		}
	}
	void initialize_layers() {
		graph.lock();		
		graph.propagate(Task::Init);
	}
public:
	SIDGP(const Graph& graph) : graph(graph) {
		initialize_layers();
		sample(10);
	}

	void train(int n_iter = 50, int ess_burn = 10) {
		train_iter += n_iter;
		auto train_start = std::chrono::system_clock::now();
		std::time_t train_start_t = std::chrono::system_clock::to_time_t(train_start);
		std::cout << "START: " << std::put_time(std::localtime(&train_start_t), "%F %T") << std::endl;
		ProgressBar* train_prog = new ProgressBar(std::clog, 70u, "[TRAIN]");
		for (int i = 0; i < n_iter; ++i) {
			//double progress = double(i) * 100.0 / double(n_iter);
			train_prog->write((double(i) / double(n_iter)));
			// I-step
			sample(ess_burn);
			// M-step
			graph.propagate(Task::Train);
		}
		delete train_prog;
		auto train_end = std::chrono::system_clock::now();
		std::time_t train_end_t = std::chrono::system_clock::to_time_t(train_end);
		std::cout << "END: " << std::put_time(std::localtime(&train_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
	}
	void estimate(Eigen::Index n_burn = 0) {
		if (n_burn == 0) n_burn = std::size_t(0.75 * train_iter);		
		else if (n_burn > train_iter) throw std::runtime_error("n_burn > train_iter");	
		for (std::vector<Layer>::iterator layer = graph.m_layers.begin(); layer != graph.m_layers.end(); ++layer) {
			layer->estimate_parameters(n_burn);
		}
	}
	MatrixPair predict(const TMatrix& X, unsigned int n_impute = 50, unsigned int n_thread = 1) {
		sample(50);
		TMatrix mean = TMatrix::Zero(X.rows(), 1);
		TMatrix variance = TMatrix::Zero(X.rows(), 1);
		std::vector<MatrixPair> predictions;

		auto pred_start = std::chrono::system_clock::now();
		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
		std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
		ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "[PREDICT]");
		graph.n_thread = n_thread;
		for (int i = 0; i < n_impute; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			if (i == 0) {
				mean = output.first;
				variance = square(output.first.array()).matrix() + output.second;
			}
			else {
				mean.noalias() += output.first;
				variance.noalias() += (square(output.first.array()).matrix() + output.second);
			}
			pred_prog->write((double(i) / double(n_impute)));
		}
		delete pred_prog;

		auto pred_end = std::chrono::system_clock::now();
		std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
		std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
		mean.array() /= double(n_impute);
		variance.array() /= double(n_impute);
		variance.array() -= square(mean.array());

		return std::make_pair(mean, variance);
	}
	MatrixPair predict(const TMatrix& X, TMatrix& Yref, unsigned int n_impute = 50, unsigned int n_thread = 1) {
		sample(50);
		TMatrix mean = TMatrix::Zero(X.rows(), 1);
		TMatrix variance = TMatrix::Zero(X.rows(), 1);
		std::vector<MatrixPair> predictions;

		auto pred_start = std::chrono::system_clock::now();
		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
		std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
		ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "[PREDICT]");
		graph.n_thread = n_thread;
		for (int i = 0; i < n_impute; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			if (i == 0) {
				mean = output.first;
				variance = square(output.first.array()).matrix() + output.second;
			}
			else {
				mean.noalias() += output.first;
				variance.noalias() += (square(output.first.array()).matrix() + output.second);
			}
			TVector tmp_mu = mean.array() / double(i);
			double nrmse = metrics::rmse(Yref, tmp_mu) / (Yref.maxCoeff() - Yref.minCoeff());
			pred_prog->write((double(i) / double(n_impute)), nrmse);
		}
		delete pred_prog;

		auto pred_end = std::chrono::system_clock::now();
		std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
		std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
		mean.array() /= double(n_impute);
		variance.array() /= double(n_impute);
		variance.array() -= square(mean.array());

		return std::make_pair(mean, variance);
	}

public:
	Graph graph;
	unsigned int train_iter = 0;
	unsigned int verbosity = 1;
};

void engine() {
	TMatrix X_train = read_data("../datasets/engine/X_train.txt");
	TMatrix Y_train = read_data("../datasets/engine/Y_train.txt");
	TMatrix X_test = read_data("../datasets/engine/X_test.txt");
	TMatrix Y_test = read_data("../datasets/engine/Y_test.txt");
	Graph graph(std::make_pair(X_train, Y_train), 1);
	for (Eigen::Index i = 0; i < X_train.cols(); ++i) {
		graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52);
		graph.layer(static_cast<int>(i))->fix_likelihood_variance();
	}
	SIDGP model(graph);
	model.train(100, 10);
	model.estimate();
	MatrixPair Z = model.predict(X_test, Y_test, 100, 300);
	TMatrix mean = Z.first;
	TMatrix var = Z.second;	
}

void plot(const TMatrix& X_plot, std::string& exp, SIDGP& model) {
    std::cout << "================ PLOT ================" << std::endl;
    MatrixPair Zplot = model.predict(X_plot, 100, 300);
    TMatrix Zpm = Zplot.first;
    TMatrix Zpv = Zplot.second;
    std::string Zpm_path = "../results/analytic2/" + exp + "PM.dat";
    std::string Zpv_path = "../results/analytic2/" + exp + "PV.dat";
    write_data(Zpm_path, Zpm);
    write_data(Zpv_path, Zpv);
}

void analytic2(std::string exp) {
	TMatrix X_train = read_data("../datasets/analytic2/X_train.dat");
	TMatrix Y_train = read_data("../datasets/analytic2/Y_train.dat");
    TMatrix X_test = read_data("../datasets/analytic2/X_test.dat");
    TMatrix Y_test = read_data("../datasets/analytic2/Y_test.dat");    
    TMatrix X_plot = read_data("../datasets/analytic2/X_plot.dat");
	Graph graph(std::make_pair(X_train, Y_train), 1);

	graph.layer(0)->set_kernels(TKernel::TMatern52);
	graph.layer(1)->set_kernels(TKernel::TMatern52);
	graph.layer(2)->set_kernels(TKernel::TMatern52);
	graph.layer(0)->fix_likelihood_variance();
	graph.layer(1)->fix_likelihood_variance();
	graph.layer(2)->fix_likelihood_variance();

	SIDGP model(graph);
	model.train(100, 10);
    model.estimate();
    plot(X_plot, exp, model);
    std::cout << "================= MCS ================" << std::endl;
    MatrixPair Z = model.predict(X_test, Y_test, 75, 300);
    TMatrix Zmcs = Z.first;
    TMatrix Zvcs = Z.second;
    std::string Zmcs_path = "/home/alfaisal/FAIZ/fdml/results/analytic2/" + exp + "MCSM.dat";
    std::string Zvcs_path = "/home/alfaisal/FAIZ/fdml/results/analytic2/" + exp + "MCSV.dat";
    write_data(Zmcs_path, Zmcs);
    write_data(Zvcs_path, Zvcs);

    double nrmse = rmse(Y_test, Zmcs) / (Y_test.maxCoeff() - Y_test.minCoeff());
    std::cout << "NRMSE = " << nrmse << std::endl;    
}

int main() {
    for (unsigned int i = 14; i < 21; ++i){
        std::cout << "================= EXP " << i << " " << "================" << std::endl;
        analytic2(std::to_string(i));
    }
    // engine();
    return 0;
}