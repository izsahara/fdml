// #pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include <fdml/utilities.h>
#include <fdml/kernels.h>
#include <fdml/base_models.h>
#include <fdml/optimizers.h>
#include <chrono>
#include <rapidcsv.h>
#include <pcg/pcg_random.hpp>

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

static void write_to_file(std::string filepath, std::string line)
{
	std::ofstream myfile;
	myfile.open(filepath, std::fstream::app);
	myfile << line << "\n";
	myfile.close();
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
enum TSolver { TLBFGSB, TPSO, TCG, TRprop };

enum State { Input, InputConnected, Hidden, Observed, Unchanged };
enum Task { Init, Train, LinkedPredict };
using KernelPtr = std::shared_ptr<Kernel>;
using SolverPtr = std::shared_ptr<Solver>;

using fdml::optimizers::LBFGSB;
using fdml::optimizers::PSO;
using fdml::optimizers::ConjugateGradient;
using fdml::optimizers::Rprop;

pcg_extras::seed_seq_from<std::random_device> seed_source;
static pcg64 rng(seed_source);
// static std::mt19937_64 rng(std::random_device{}());

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
		os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " << " [" << std::setw(3) << std::left << std::setprecision(5) << std::fixed << nrmse * 100.0 << "%] " << std::flush;
		

	}
	void write(double fraction, double nrmse, double r2) {
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
			<< " [NRMSE = " << std::setw(3) << std::left << std::setprecision(5) << std::fixed << nrmse * 100.0 << "%] "
			<< " [R2 = " << std::setw(3) << std::left << std::setprecision(5) << std::fixed << r2 << "]"			
			<< std::flush;
	}	
};

//
class Node : public GP {
private:
	Node&   evalK(bool with_scale = true) {
		K = kernel->K(inputs, inputs, likelihood_variance.value());
		if (with_scale) K.array() *= scale.value();
		return *this;
	}
	Node&   evalK(const TMatrix& Xpad, bool with_scale = true) {
		TMatrix tmp(inputs.rows(), inputs.cols() + Xpad.cols());
		tmp << inputs, Xpad;
		K.noalias() = kernel->K(tmp, tmp, likelihood_variance.value());
		if (with_scale) K.array() *= scale.value();
		return *this;
	}
	TMatrix sample_mvn() {
		TVector mean = TVector::Zero(K.rows());
		Eigen::setNbThreads(1);
		Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(K);
		// TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
		TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
		std::normal_distribution<> dist;
		return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {return dist(rng); });
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
		kernel->gradients(inputs, grad_);
		TVector grad = TVector::Zero(grad_.size());
		for (int i = 0; i < grad_.size(); ++i) {
			TMatrix KKT = chol.solve(grad_[i]);
			double trace = KKT.trace();
			double YKKT = (outputs.transpose() * KKT * alpha).coeff(0);
			double P1 = -0.5 * trace;
			double P2 = 0.5 * YKKT;
			grad[i] = -P1 - (P2 / scale.value());
		}
		// TODO: Add likelihood_variance/nugget gradient
		grad -= log_prior_gradient();
		return grad;
	}
	TVector get_params() override {
		TVector params = kernel->get_params();
		if (!(*likelihood_variance.is_fixed)) {
			likelihood_variance.transform_value();
			params.conservativeResize(params.rows() + 1);
			params.tail(1)(0) = likelihood_variance.value();
		}
		return params;
	}
	double  log_likelihood() {
		chol = K.llt();
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
		double lml = -0.5 * (logdet + quad);
		return lml;
	}
	double  log_marginal_likelihood() override {
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double YKinvY = (outputs.transpose() * alpha)(0);
		double NLL = 0.0;
		if (*scale.is_fixed) NLL = 0.5 * (logdet + (YKinvY/scale.value()));
		else NLL = 0.5 * (logdet + (inputs.rows() * log(scale.value())));		
		NLL -= log_prior();
		return NLL;
	}
	double  log_prior() {
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
	void    set_params(const TVector& new_params) override
	{
		kernel->set_params(new_params);
		if (!(*likelihood_variance.is_fixed)) likelihood_variance.transform_value(new_params.tail(1)(0));
		update_cholesky();
	}
	void    get_bounds(TVector& lower, TVector& upper) {
		kernel->get_bounds(lower, upper);

		if (!(*likelihood_variance.is_fixed)) {
			lower.conservativeResize(lower.rows() + 1);
			upper.conservativeResize(upper.rows() + 1);
			lower.tail(1)(0) = likelihood_variance.get_bounds().first;
			upper.tail(1)(0) = likelihood_variance.get_bounds().second;
		}
	}
	void    train() override {
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
		if (store_parameters) {
			TVector params(theta.size() + 1);
			params << theta, scale.value();
			history.push_back(params);
		}
	}
	// GP Predict
	void    predict(const TMatrix& X, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {
		update_cholesky();
		TMatrix Ks = kernel->K(inputs, X);
		TMatrix Kss = kernel->diag(X);
		TMatrix V = chol.solve(Ks);
		latent_mu = Ks.transpose() * alpha;
		latent_var = abs((scale.value() * (Kss - (Ks.transpose() * V).diagonal()).array()));
	}
	// Linked Predict (Default)
	void    predict(const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {

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
	// Linked Predict (InputConnected)
	void    predict(const MatrixPair& XX, const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {
		TMatrix X_train = XX.first;
		TMatrix X_test  = XX.second;   
		evalK(X_train, false);
		chol = K.llt();
		alpha = chol.solve(outputs);
		const Eigen::Index nrows = linked.first.rows();
		kernel->expectations(linked.first, linked.second);
		if (n_thread == 1) {
			for (Eigen::Index i = 0; i < nrows; ++i) {
				TMatrix I = TMatrix::Ones(inputs.rows(), 1);
				TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
				kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
				TMatrix Iz = kernel->K(X_train, X_test.row(i));
				TMatrix Jz = Iz * Iz.transpose();
				I.array() *= Iz.array(); J.array() *= Jz.array();
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
			auto task = [=, &X_train, &X_test, &latent_mu, &latent_var](int begin, int end)
			{
				for (Eigen::Index i = begin; i < end; ++i) {
					TMatrix I = TMatrix::Ones(inputs.rows(), 1);
					TMatrix J = TMatrix::Ones(inputs.rows(), inputs.rows());
					kernel->IJ(I, J, linked.first.row(i), linked.second.row(i), inputs, i);
					TMatrix Iz = kernel->K(X_train, X_test.row(i));
					TMatrix Jz = Iz * Iz.transpose();
					I.array() *= Iz.array(); J.array() *= Jz.array();
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
	Node(double likelihood_variance = 1E-8) : GP(likelihood_variance) {}
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
		Eigen::Index param_size = get_params().size() + 1;
		TMatrix h(history.size(), param_size);
		for (std::vector<TVector>::size_type i = 0; i != history.size(); ++i) {
			h.row(i) = history[i];
		}
		return h;
	}
	void    update_cholesky() {
		K = kernel->K(inputs, inputs, likelihood_variance.value());
		chol = K.llt();
		alpha = chol.solve(outputs);
		 //scale is not considered a variable in optimization, it is directly linked to chol
		if (*scale.is_fixed) return;
		else scale = (outputs.transpose() * alpha)(0) / outputs.rows();
	}
public:
	bool store_parameters = true;
	std::vector<TVector> history;
	Parameter<double> scale = { "scale", 1.0, "none" };

private:
	State    cstate;
	TVector  alpha;
	TLLT	 chol;
	TMatrix	 K;
};

class Layer {
public:
	Layer() = default;
	Layer(const State& layer_state, const unsigned int& n_nodes) : cstate(layer_state), n_nodes(n_nodes) {
		m_nodes.resize(n_nodes);
		for (unsigned int nn = 0; nn < n_nodes; ++nn) {
			m_nodes[nn] = Node();
		}
	}

	void set_input(const TMatrix& input) {
		if (cstate == State::Input) {
			observed_input.noalias() = input;
			set_output(input);
		}
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->inputs = input;
			nn->cstate = cstate;
			if (ARD) {
				Eigen::Index ndim = input.cols();
				double val = nn->kernel->length_scale.value()(0);
				TVector new_ls = TVector::Constant(ndim, val);
				nn->kernel->length_scale = new_ls;
			}
		}
	}
	void set_input(const TMatrix& input, const Eigen::Index& col) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->inputs.col(col) = input;
		}

	}
	void set_output(const TMatrix& output) {

		if (!locked) {
			if (cstate == State::Hidden) 
			{
				ostate = State::Observed;
				std::swap(cstate, ostate);
			}
		}

		if (cstate == State::Observed) 
			observed_output.noalias() = output;
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->outputs = output.col(nn - m_nodes.begin());
			nn->cstate = cstate;
		}
	}

	TMatrix get_input() {
		if (cstate == State::Input) return observed_input;
		else return m_nodes[0].inputs;
	}
	TMatrix get_output() {
		if (cstate == State::Observed) return observed_output;
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
	void set_kernels(const TKernel& kernel, TVector& length_scale) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			switch (kernel) {
			case TSquaredExponential:
				nn->set_kernel(KernelPtr(new SquaredExponential(length_scale)));
				continue;
			case TMatern52:
				nn->set_kernel(KernelPtr(new Matern52(length_scale)));
				continue;
			default:
				nn->set_kernel(KernelPtr(new SquaredExponential(length_scale)));
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
				//case TCG:
				//	nn->set_solver(SolverPtr(new ConjugateGradient));
				//	continue;
			case TRprop:
				nn->set_solver(SolverPtr(new Rprop));
				continue;
			}
		}
	}
	void fix_likelihood_variance() {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->likelihood_variance.fix();
		}
	}
	void fix_scale() {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->scale.fix();
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
	void predict(const TMatrix& X, bool store = false) {
		if (store) Xtmp.noalias() = X;
		latent_output = std::make_pair(
			TMatrix::Zero(X.rows(), static_cast<Eigen::Index>(n_nodes)),
			TMatrix::Zero(X.rows(), static_cast<Eigen::Index>(n_nodes)));
		
		for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
			Eigen::Index cc = static_cast<Eigen::Index>(node - m_nodes.begin());
			node->predict(X, latent_output.first.col(cc), latent_output.second.col(cc));
		}
	}
	void predict(const MatrixPair& linked) {
		latent_output = std::make_pair(
			TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)),
			TMatrix::Zero(linked.first.rows(), static_cast<Eigen::Index>(n_nodes)));
		for (std::vector<Node>::iterator node = m_nodes.begin(); node != m_nodes.end(); ++node) {
			Eigen::Index cc = static_cast<Eigen::Index>(node - m_nodes.begin());
			node->n_thread = n_thread;
			if (cstate == State::InputConnected)  
				node->predict(std::make_pair(observed_input, Xtmp), 
							  linked, latent_output.first.col(cc), 
							  latent_output.second.col(cc));
			else 
				node->predict(linked, 
							  latent_output.first.col(cc), 
							  latent_output.second.col(cc));
		}
	}
	void connect(const TMatrix& Ginput) {
		if (locked) throw std::runtime_error("Layer Locked");
		if (cstate != State::InputConnected)
		{
			ostate = State::InputConnected;
			std::swap(cstate, ostate);
		}
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			if (nn->kernel->length_scale.value().size() > 1) throw std::runtime_error("Input Connections Only Available with Non ARD");
			nn->cstate = cstate;
		}
		observed_input = Ginput;
	}
	Layer& evalK(bool with_scale = true) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			if (cstate == State::InputConnected) nn->evalK(observed_input, with_scale);
			else nn->evalK(with_scale);
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
			if (*(node->scale.is_fixed)) node->scale.unfix();
			node->scale = theta.tail(1)(0);
			node->scale.fix();
			TVector tmp = theta.head(theta.size() - 1);
			node->set_params(tmp);
		}
	}
public:
	State cstate; // Current Layer State
	unsigned int n_nodes;
	bool ARD = false;
private:
	std::vector<Node> m_nodes;
	State ostate = State::Unchanged; // Old Layer State;
	bool locked = false;
	unsigned int n_thread = 1;
	TMatrix Xtmp;
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
				m_layers[ll] = Layer(State::Input, n_nodes);
				m_layers[ll].set_input(data.first);
			}
			else if (ll == n_layers - 1) {
				m_layers[ll] = Layer(State::Observed, static_cast<int>(data.second.cols()));
				m_layers[ll].set_output(data.second);
			}
			else {
				m_layers[ll] = Layer(State::Hidden, n_nodes);
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
		if (layer(layer_idx)->cstate == State::Input) throw std::runtime_error("Invalid Connection: InputLayer");
		m_layers[layer_idx].connect(m_layers[0].observed_input);
	}

private:
	std::vector<Layer> m_layers;
	void check_connected(const TMatrix& X) {
		for (std::vector<Layer>::iterator cp = m_layers.begin() + 1; cp != m_layers.end(); ++cp) {
			if (cp->cstate == State::InputConnected) cp->Xtmp = X;
		}
	}

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
						cp->set_input(std::prev(cp)->get_output());
						cp->set_output(pca.transform(std::prev(cp)->get_output()));
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

class SIDGP {
private:
	TMatrix update_f(const TMatrix& f, const TMatrix& nu, const double& params) {
		TVector mean = TVector::Zero(f.rows());
		return ((f - mean).array() * (cos(params))).matrix() + ((nu - mean).array() * (sin(params))).matrix() + mean;
	}
	void sample(unsigned int n_burn = 1) {
		auto rand_u = [](const double& a, const double& b) {
			std::uniform_real_distribution<> uniform_dist(a, b);
			return uniform_dist(rng);
		};

		double log_y, theta, theta_min, theta_max;
		for (unsigned int nb = 0; nb < n_burn; ++nb) {
			for (std::vector<Layer>::iterator cl = graph.m_layers.begin(); cl != graph.m_layers.end() - 1; ++cl) {
				if (cl->cstate == State::Observed) continue; // TODO: missingness
				auto linked_layer = std::next(cl);
				for (std::vector<Node>::iterator cn = cl->m_nodes.begin(); cn != cl->m_nodes.end(); ++cn) {
					TMatrix nu(cn->inputs.rows(), 1);
					if (cl->cstate == State::InputConnected) nu = cn->evalK(cl->observed_input).sample_mvn();
					else nu = cn->evalK().sample_mvn();
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
						linked_layer->set_input(fp, col);
						double log_yp = linked_layer->evalK().log_likelihood();
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

	void train(int n_iter = 50, int ess_burn = 10, Eigen::Index n_burn = 0) {
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
		// Estimate Parameters
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
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
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
	MatrixPair predict(const TMatrix& X, TMatrix& Yref, bool& nanflag, unsigned int n_impute = 50, unsigned int n_thread = 1) {
		sample(50);
		TMatrix mean = TMatrix::Zero(X.rows(), 1);
		TMatrix variance = TMatrix::Zero(X.rows(), 1);
		std::vector<MatrixPair> predictions;

		auto pred_start = std::chrono::system_clock::now();
		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
		std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
		ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
		graph.n_thread = n_thread;
		graph.check_connected(X);
		for (int i = 0; i < n_impute; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
			if ((mean.array().isNaN()).any()) {nanflag = true;  break;}
			TVector tmp_mu = mean.array() / double(i+1);
			double nrmse = metrics::rmse(Yref, tmp_mu, true);
			if (i > 2 && nrmse > 0.03) {nanflag = true;  break;}
			double r2 = metrics::r2_score(Yref, tmp_mu);			
			pred_prog->write((double(i) / double(n_impute)), nrmse, r2);
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
	MatrixPair predict(const TMatrix& X, TMatrix& Yref, metrics::StandardScaler& scaler, unsigned int n_impute = 50, unsigned int n_thread = 1) {
		sample(50);
		TMatrix mean = TMatrix::Zero(X.rows(), 1);
		TMatrix variance = TMatrix::Zero(X.rows(), 1);
		std::vector<MatrixPair> predictions;

		auto pred_start = std::chrono::system_clock::now();
		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
		std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
		ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "");
		graph.n_thread = n_thread;
		graph.check_connected(X);
		for (int i = 0; i < n_impute; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
			TVector tmp_mu = mean.array() / double(i+1);
			// TMatrix reYref = scaler.rescale(Yref);
			// TMatrix reMean = scaler.rescale(tmp_mu);
			double nrmse = metrics::rmse(Yref, tmp_mu, true);
			double r2 = metrics::r2_score(Yref, tmp_mu);
			pred_prog->write((double(i) / double(n_impute)), nrmse, r2);
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
	// TMatrix X_train = read_data("../datasets/engine/X_train.txt");
	// TMatrix Y_train = read_data("../datasets/engine/Y_train.txt");
	// TMatrix X_test = read_data("../datasets/engine/X_test.txt");
	// TMatrix Y_test = read_data("../datasets/engine/Y_test.txt");
	// Graph graph(std::make_pair(X_train, Y_train), 1);
	// for (Eigen::Index i = 0; i < X_train.cols(); ++i) {
	// 	graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52);
	// 	graph.layer(static_cast<int>(i))->fix_likelihood_variance();
	// }
	// // graph.connect_inputs(1);
	// // graph.connect_inputs(2);
	// SIDGP model(graph);
	// model.train(10, 1);
	// MatrixPair Z = model.predict(X_test, Y_test, 100, 5);
	// TMatrix mean = Z.first;
	// TMatrix var = Z.second;	
}
void analytic2(std::string exp) {
	// auto plot = [=](const TMatrix& X_plot, SIDGP& model) {
	// 	std::cout << "================ PLOT ================" << std::endl;
	// 	MatrixPair Zplot = model.predict(X_plot, 100, 300);
	// 	TMatrix Zpm = Zplot.first;
	// 	TMatrix Zpv = Zplot.second;
	// 	std::string Zpm_path = "../results/analytic2/" + exp + "PM.dat";
	// 	std::string Zpv_path = "../results/analytic2/" + exp + "PV.dat";
	// 	write_data(Zpm_path, Zpm);
	// 	write_data(Zpv_path, Zpv);
	// };

	// TMatrix X_train = read_data("../datasets/analytic2/X_train.dat");
	// TMatrix Y_train = read_data("../datasets/analytic2/Y_train.dat");
	// TMatrix X_test = read_data("../datasets/analytic2/X_test.dat");
	// TMatrix Y_test = read_data("../datasets/analytic2/Y_test.dat");
	// TMatrix X_plot = read_data("../datasets/analytic2/X_plot.dat");
	// Graph graph(std::make_pair(X_train, Y_train), 1);

	// graph.layer(0)->set_kernels(TKernel::TMatern52);
	// graph.layer(1)->set_kernels(TKernel::TMatern52);
	// graph.layer(2)->set_kernels(TKernel::TMatern52);
	// graph.layer(0)->fix_likelihood_variance();
	// graph.layer(1)->fix_likelihood_variance();
	// graph.layer(2)->fix_likelihood_variance();

	// SIDGP model(graph);
	// model.train(100, 10);
	// // plot(X_plot, exp, model);
	// std::cout << "================= MCS ================" << std::endl;
	// MatrixPair Z = model.predict(X_test, Y_test, 75, 300);
	// TMatrix Zmcs = Z.first;
	// TMatrix Zvcs = Z.second;
	// std::string Zmcs_path = "/home/alfaisal/FAIZ/fdml/results/analytic2/" + exp + "MCSM.dat";
	// std::string Zvcs_path = "/home/alfaisal/FAIZ/fdml/results/analytic2/" + exp + "MCSV.dat";
	// write_data(Zmcs_path, Zmcs);
	// write_data(Zvcs_path, Zvcs);

	// double nrmse = rmse(Y_test, Zmcs) / (Y_test.maxCoeff() - Y_test.minCoeff());
	// std::cout << "NRMSE = " << nrmse << std::endl;
}

void nrel(std::string output, std::string exp) {
	// metrics::StandardScaler scaler1, scaler2, scaler3, scaler4;
	// TMatrix X_train = read_data("../datasets/nrel/75/X_train.dat");
	// scaler1.scale(X_train);
	// TMatrix X_test = read_data("../datasets/nrel/75/X_test.dat");
	// scaler2.scale(X_test);

	// std::string train_path = "../datasets/nrel/75/" + output + "/TR-" + output + ".dat";
	// std::string test_path = "../datasets/nrel/75/" + output + "/TS-" + output + ".dat";	
	// TMatrix Y_train = read_data(train_path);
	// scaler3.scale(Y_train);
	// TMatrix Y_test = read_data(test_path);
	// scaler4.scale(Y_test);

	// Graph graph(std::make_pair(X_train, Y_train), 1);
	// for (unsigned int i = 0; i < graph.n_layers-1; ++i) {
	// 	graph.layer(static_cast<int>(i))->fix_likelihood_variance();
	// 	TVector ls = TVector::Constant(X_train.cols(), 1.0);
	// 	graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52, ls);
	// }
	// // Last Layer
	// TVector ols = TVector::Constant(X_train.cols(), 1.0);
	// graph.layer(2)->fix_likelihood_variance();
	// graph.layer(2)->set_kernels(TKernel::TMatern52, ols);
	// //
	// SIDGP model(graph);
	// model.train(100, 200);
	// MatrixPair Z = model.predict(X_test, Y_test, scaler4, 100, 190);
	// TMatrix mean = Z.first;
	// TMatrix var = Z.second;
	// std::string m_path = "../results/nrel/75/" + output + "/" + exp + "-M.dat";
    // std::string v_path = "../results/nrel/75/" + output + "/" + exp + "-V.dat";
    // write_data(m_path, mean);
    // write_data(v_path, var);

	// double nrmse = metrics::rmse(Y_test, mean, true);
	// double r2 = metrics::r2_score(Y_test, mean);	
	// std::cout << "(Scaled) NRMSE = " << nrmse << std::endl;
	// std::cout << "(Scaled) R2 = " << r2 << std::endl;

	// TMatrix resc_mean = scaler4.rescale(mean);
	// TMatrix resc_true = scaler4.rescale(Y_test);
	// std::cout << "(Rescaled) NRMSE = " << metrics::rmse(resc_true, resc_mean, true) << std::endl;
	// std::cout << "(Recaled) R2 = " << metrics::r2_score(resc_true, resc_mean) << std::endl;	
	// std::string rem_path = "../results/nrel/75/" + output + "/" + exp + "-MRE.dat";
	// std::string ret_path = "../results/nrel/75/" + output + "/" + exp + "-TRE.dat";
    // write_data(rem_path, resc_mean);
    // write_data(ret_path, resc_true);	
}

void airfoil(std::string exp, bool& restart) {
	TMatrix X_train = read_data("../datasets/airfoil/40/Xsc_train.dat");
	TMatrix Y_train = read_data("../datasets/airfoil/40/Y_train.dat");
	TMatrix X_test = read_data("../datasets/airfoil/40/X_test2.dat");
	TMatrix X_plot = read_data("../datasets/airfoil/40/X_plot.dat");
	TMatrix Y_test = read_data("../datasets/airfoil/40/Y_test.dat");

	Graph graph(std::make_pair(X_train, Y_train), 2);
	for (unsigned int i = 0; i < graph.n_layers; ++i) {
		TVector ls = TVector::Constant(X_train.cols(), 1.0);
		graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52);
		graph.layer(static_cast<int>(i))->fix_likelihood_variance();
	}
	// graph.layer(0)->fix_scale();
	// graph.layer(1)->fix_scale();
	// graph.layer(2)->fix_scale();
	SIDGP model(graph);
	model.train(100, 10);
	bool nanflag = false;
	MatrixPair Z = model.predict(X_test, Y_test, nanflag, 100, 192);
	TMatrix mean = Z.first;
	TMatrix var = Z.second;
	double nrmse = metrics::rmse(Y_test, mean, true);	

	if (nanflag){
		restart = true;
	}
	else {
		std::string e_path = "../results/airfoil/40/NRMSE.dat";		
		std::cout << "NRMSE = " << nrmse << std::endl;
		
		std::string m_path = "../results/airfoil/40/" + exp + "-M.dat";
		std::string v_path = "../results/airfoil/40/" + exp + "-V.dat";
		write_data(m_path, mean);
		write_data(v_path, var);
		if (exp != "1"){
			TVector error_ = read_data(e_path);
			double min = error_.minCoeff();
			if (nrmse < min){
				std::cout << "Plot" << std::endl;
				MatrixPair Zplot = model.predict(X_plot, 100, 192);
				std::string p_path = "../results/airfoil/40/" + exp + "-P.dat";
				TMatrix Zp = Zplot.first;
				write_data(p_path, Zp);
			}
		}
		write_to_file(e_path, std::to_string(nrmse));		

	}
}


int main() {
	bool restart = false;
	unsigned int i = 6;
	unsigned int finish = 41;
	while(true){
		bool restart = false;
		std::cout << "================= " << " EXP " << i << " ================" << std::endl;
		airfoil(std::to_string(i), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else i++;
		if (i == finish) break;
	}
	return 0;
}