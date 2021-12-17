#include <filesystem>
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
enum LLF { Gaussian, Heteroskedastic };
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
struct Likelihood {

	Likelihood() = default;
	Likelihood(const LLF& likelihood) : likelihood(likelihood) {}

	void log_likelihood() {

	}

	TMatrix	 X;
	TMatrix	 Y;
	TVector  alpha;
	TLLT	 chol;
	TMatrix	 K;

	LLF likelihood = LLF::Gaussian;

};

//
class Node : public GP {
private:
	Node&	evalK(bool with_scale = true) {
		K = kernel->K(inputs, inputs, likelihood_variance.value());
		if (with_scale) K.array() *= scale.value();
		return *this;
	}
	Node&	evalK(const TMatrix& Xpad, bool with_scale = true) {
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
		//TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
		TMatrix transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
		std::normal_distribution<> dist;
		return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) {return dist(rng); });
	}
	TMatrix sample_mvn(const TVector& mean, const TMatrix& cov) {
		Eigen::setNbThreads(1);
		Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(cov);
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
		if (!(*likelihood_variance.is_fixed))
			grad_.push_back(likelihood_variance.value() * TMatrix::Identity(inputs.rows(), inputs.rows()));
		for (int i = 0; i < grad.size(); ++i) {
			TMatrix KKT = chol.solve(grad_[i]);
			double trace = KKT.trace();
			double YKKT = (outputs.transpose() * KKT * alpha).coeff(0);
			double P1 = -0.5 * trace;
			double P2 = 0.5 * YKKT;
			grad[i] = -P1 - (P2 / scale.value());
		}
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
		switch(likelihood) {
		case Heteroskedastic:
		{
			if (inputs.cols() != 2) throw std::runtime_error("Heteroskedastic GP requires 2D inputs");
			TMatrix mu = inputs.col(0);
			TMatrix var = exp(inputs.col(1).array());
			double ll = (-0.5 * (log(2 * PI * var.array()) + pow((outputs - mu).array(), 2) / var.array())).sum();
			return ll;
		}
		default:
		{
			chol = K.llt();
			double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
			double quad = (outputs.array() * (chol.solve(outputs)).array()).sum();
			double ll = -0.5 * (logdet + quad);
			return ll;
		}
		}
		
	}
	double  log_marginal_likelihood() override {
		double logdet = 2 * chol.matrixL().toDenseMatrix().diagonal().array().log().sum();
		double YKinvY = (outputs.transpose() * alpha)(0);
		double NLL = 0.0;
		if (*scale.is_fixed) NLL = 0.5 * (logdet + (YKinvY / scale.value()));
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
			std::pair<double, double> llvb = likelihood_variance.get_bounds();
			double lb = llvb.first;
			double ub = llvb.second;
			if (likelihood_variance.get_transform() == "logexp") {
				if (std::isinf(lb)) lb = -23.025850929940457;
				if (std::isinf(ub)) ub = 0.0;
			}
			lower.conservativeResize(lower.rows() + 1);
			upper.conservativeResize(upper.rows() + 1);
			lower.tail(1)(0) = lb;
			upper.tail(1)(0) = ub;
		}
	}
	void    train() override {
		if (likelihood != LLF::Gaussian) return;
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
		switch (likelihood) {
		case Heteroskedastic:
		{
			/*
			* y_mean = m[:,0]
			* y_var = np.exp(m[:,1]+v[:,1]/2)+v[:,0]
			* return y_mean.flatten(),y_var.flatten()
			*/
			latent_mu = linked.first.col(0);
			latent_var = exp((linked.first.col(1) + (linked.second.col(1) / 2)).array()) + linked.second.col(0).array();
		}
		default: 
		{
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
		}


	}
	// Linked Predict (InputConnected)
	void    predict(const MatrixPair& XX, const MatrixPair& linked, Eigen::Ref<TVector> latent_mu, Eigen::Ref<TVector> latent_var) {
		TMatrix X_train = XX.first;
		TMatrix X_test = XX.second;
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
	// Non-Gaussian Likelihoods
	TMatrix	posterior(const TMatrix& K_prev) {
		// Zero Mean posterior
		TMatrix gamma = exp(inputs.col(1).array()).matrix().asDiagonal();
		TVector tmp1 = (gamma + K_prev).fullPivLu().solve(outputs);
		TVector mean = (K_prev.array().colwise() * tmp1.array()).rowwise().sum().matrix();
		TMatrix cov = K_prev * (gamma + K_prev).fullPivLu().solve(gamma);
		return sample_mvn(mean, cov);
	}

public:
	Node(double likelihood_variance = 1E-6) : GP(likelihood_variance) {}
	void set_kernel(const KernelPtr& rkernel) {
		kernel = std::move(rkernel);
	}
	void set_solver(const SolverPtr& rsolver) {
		solver = std::move(rsolver);
	}
	void set_likelihood_variance(const double& lv) {
		if (lv <= 0.0) likelihood_variance.transform_value(lv);
		else likelihood_variance = lv;
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
	LLF  likelihood = LLF::Gaussian;
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
	void set_likelihood(const LLF& likelihood) {
		this->likelihood = likelihood;
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->likelihood = likelihood;
		}
	}
	
	void set_likelihood_variance(const double& value) {
		for (std::vector<Node>::iterator nn = m_nodes.begin(); nn != m_nodes.end(); ++nn) {
			nn->set_likelihood_variance(value);
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
			if (node->likelihood != LLF::Gaussian) continue;
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
	LLF likelihood = LLF::Gaussian;
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
					// idx = np.random.choice(input.shape[1], n_nodes - input.shape[1])
					// col = input[:, idx]
					// output = np.hstack([input, col])
					TMatrix cinputs = std::prev(cp)->get_output();
					TMatrix cols = cinputs.col(0).replicate(1, cp->n_nodes - std::prev(cp)->n_nodes);;
					TMatrix tmp(cinputs.rows(), cp->n_nodes);
					tmp << cinputs, cols;
					cp->set_input(std::prev(cp)->get_output());
					cp->set_output(tmp);
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
					const Eigen::Index col = static_cast<Eigen::Index>((cn - cl->m_nodes.begin()));
					TMatrix nu(cn->inputs.rows(), 1);
					if (cl->cstate == State::InputConnected) nu = cn->evalK(cl->observed_input).sample_mvn();
					else nu = cn->evalK().sample_mvn();

					if (col == 0 && linked_layer->likelihood == LLF::Heteroskedastic) {
						TMatrix ff = linked_layer->m_nodes[0].posterior(cn->K); // cn->K
						linked_layer->set_input(ff, 0);
						cn->outputs = ff;
						continue;
					}
					log_y = linked_layer->evalK().log_likelihood() + log(rand_u(0.0, 1.0));
					//
					if (!std::isfinite(log_y)) { throw std::runtime_error("log_y is not finite"); }
					//
					theta = rand_u(0.0, 2.0 * PI);
					theta_min = theta - 2.0 * PI;
					theta_max = theta;				
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
	MatrixPair predict(const TMatrix& X, unsigned int n_predict = 50, unsigned int n_thread = 1) {
		sample(50);
		TMatrix mean = TMatrix::Zero(X.rows(), 1);
		TMatrix variance = TMatrix::Zero(X.rows(), 1);
		std::vector<MatrixPair> predictions;

		auto pred_start = std::chrono::system_clock::now();
		std::time_t pred_start_t = std::chrono::system_clock::to_time_t(pred_start);
		std::cout << "START: " << std::put_time(std::localtime(&pred_start_t), "%F %T") << std::endl;
		ProgressBar* pred_prog = new ProgressBar(std::clog, 70u, "[PREDICT]");
		graph.n_thread = n_thread;
		for (int i = 0; i < n_predict; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
			pred_prog->write((double(i) / double(n_predict)));
		}
		delete pred_prog;

		auto pred_end = std::chrono::system_clock::now();
		std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
		std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
		mean.array() /= double(n_predict);
		variance.array() /= double(n_predict);
		variance.array() -= square(mean.array());

		return std::make_pair(mean, variance);
	}
	MatrixPair predict(const TMatrix& X, TMatrix& Yref, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
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
		for (int i = 0; i < n_predict; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
			if ((mean.array().isNaN()).any()) { nanflag = true;  break; }
			TVector tmp_mu = mean.array() / double(i + 1);
			double nrmse = metrics::rmse(Yref, tmp_mu, true);
			if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
			double r2 = metrics::r2_score(Yref, tmp_mu);
			pred_prog->write((double(i) / double(n_predict)), nrmse, r2);
		}
		delete pred_prog;

		auto pred_end = std::chrono::system_clock::now();
		std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
		std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
		mean.array() /= double(n_predict);
		variance.array() /= double(n_predict);
		variance.array() -= square(mean.array());

		return std::make_pair(mean, variance);
	}
	MatrixPair predict(const TMatrix& X, TMatrix& Yref, std::string mcs_path, bool& nanflag, unsigned int n_predict = 50, unsigned int n_thread = 1) {
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
		for (int i = 0; i < n_predict; ++i) {
			sample();
			graph.layer(0)->predict(X);
			graph.propagate(Task::LinkedPredict);
			MatrixPair output = graph.layer(-1)->latent_output;
			mean.noalias() += output.first;
			variance.noalias() += (square(output.first.array()).matrix() + output.second);
			if ((mean.array().isNaN()).any()) { nanflag = true;  break; }

			TVector nrmse_mu = mean.array() / double(i + 1);
			double nrmse = metrics::rmse(Yref, nrmse_mu, true);
			// if (i > 2 && nrmse > 0.5) { nanflag = true;  break; }
			double r2 = metrics::r2_score(Yref, nrmse_mu);
			pred_prog->write((double(i) / double(n_predict)), nrmse, r2);

			if (i == 0 || i == 99 || i == 199 || i == 299 || i == 399 || i == 499) {
				std::string mu_path = mcs_path + "-" + std::to_string(i) + "-M-MCS.dat";
				std::string var_path = mcs_path + "-" + std::to_string(i) + "-V-MCS.dat";
				TMatrix tmp_mu = mean.array() / double(i + 1);
				TMatrix tmp_var = variance.array() / double(i + 1);
				tmp_var.array() -= square(tmp_mu.array());
				write_data(mu_path, tmp_mu);
				write_data(var_path, tmp_var);
			}
		}
		delete pred_prog;

		auto pred_end = std::chrono::system_clock::now();
		std::time_t pred_end_t = std::chrono::system_clock::to_time_t(pred_end);
		std::cout << "END: " << std::put_time(std::localtime(&pred_end_t), "%F %T") << std::endl;
		std::cout << std::endl;
		mean.array() /= double(n_predict);
		variance.array() /= double(n_predict);
		variance.array() -= square(mean.array());

		return std::make_pair(mean, variance);
	}


public:
	Graph graph;
	unsigned int train_iter = 0;
	unsigned int verbosity = 1;
};

struct Case {
	Case() = default;
	Case(const std::string& problem) : problem(problem) {}
	Case(const std::string& problem, const std::string& output) : problem(problem), output(output) {}
	std::string problem;
	std::string output = "";
	unsigned int n_train;
	unsigned int experiment;
	unsigned int start;
	unsigned int finish;
	unsigned int train_iter;
	unsigned int train_impute;
	unsigned int pred_iter;
	bool plot = false;
	double likelihood_variance = 1E-10;
};

void case1_training(Case& case_study, int& train_iter, int& train_impute) {
	// Training Phase
	std::string data_path = "../datasets/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/";
	auto run_problem = [=](std::string results_path, std::string exp, bool& restart) {
		TMatrix X_train = read_data(data_path + "Xsc_train.dat");
		TMatrix Y_train = read_data(data_path + "Y_train.dat");
		TMatrix X_test = read_data(data_path + "Xsc_test.dat");
		TMatrix Y_test = read_data(data_path + "Y_test.dat");

		Graph graph(std::make_pair(X_train, Y_train), 1);
		for (unsigned int i = 0; i < graph.n_layers; ++i) {
			TVector ls = TVector::Constant(X_train.cols(), 1.0);
			graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52, ls);
			graph.layer(static_cast<int>(i))->set_likelihood_variance(case_study.likelihood_variance);
			graph.layer(static_cast<int>(i))->fix_likelihood_variance();
		}
		SIDGP model(graph);
		model.train(train_iter, train_impute);
		bool nanflag = false;
		MatrixPair Z = model.predict(X_test, Y_test, nanflag, case_study.pred_iter, 96);
		TMatrix mean = Z.first;
		TMatrix var = Z.second;
		double nrmse = metrics::rmse(Y_test, mean, true);

		if (nanflag) {
			restart = true;
		}
		else {
			std::string e_path = results_path + "NRMSE.dat";
			std::cout << "NRMSE = " << nrmse << std::endl;

			std::string m_path = results_path + exp + "-M.dat";
			std::string v_path = results_path + exp + "-V.dat";
			write_data(m_path, mean);
			write_data(v_path, var);
			write_to_file(e_path, std::to_string(nrmse));
		}
	};


	if (!std::filesystem::exists("../results/case_1/"))
		std::filesystem::create_directory("../results/case_1/");
	// ../results/case_1/analytic2
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem);
	// ../results/case_1/analytic2/25
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train));
	// ../results/case_1/analytic2/25/100
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter));

	// ../results/case_1/analytic2/25/100/100 || TRAIN_ITER/TRAIN_IMPUTE
	std::string main_results_path = "../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter) + "/" + std::to_string(train_impute);
	if (!std::filesystem::exists(main_results_path)) std::filesystem::create_directory(main_results_path);
	unsigned int ii = case_study.start;
	while (true) {
		std::cout << "================= " << "Running " << case_study.problem << "-" << case_study.n_train << " :"
			<< train_iter << "-" << train_impute << "================= " << std::endl;
		bool restart = false;
		std::cout << "================= " << "" << " REP " << ii << " ================" << std::endl;
		// ../results/case_1/analytic2/25/100/100/1 .... 25 || TRAIN_ITER/TRAIN_IMPUTE/REP
		std::string results_path = main_results_path + "/" + std::to_string(ii) + "/";
		if (!std::filesystem::exists(results_path)) std::filesystem::create_directory(results_path);
		run_problem(results_path, std::to_string(case_study.experiment), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else ii++;
		if (ii == case_study.finish) break;

	}
}
void case1_prediction(Case& case_study, int& train_iter, int& train_impute) {
	std::string data_path = "../datasets/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/";
	auto run_problem = [=](std::string results_path, std::string exp, bool& restart) {
		TMatrix X_train = read_data(data_path + "Xsc_train.dat");
		TMatrix Y_train = read_data(data_path + "Y_train.dat");

		//TMatrix X_test = read_data(data_path + "Xsc_test.dat");
		//TMatrix Y_test = read_data(data_path + "Y_test.dat");

		TMatrix X_plot = read_data(data_path + "X_plot.dat");
		TMatrix X_test = read_data(data_path + "X_mcs.dat");
		TMatrix Y_test = read_data(data_path + "Y_mcs.dat");

		Graph graph(std::make_pair(X_train, Y_train), 1);
		for (unsigned int i = 0; i < graph.n_layers; ++i) {
			TVector ls = TVector::Constant(X_train.cols(), 1.0);
			graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52, ls);
			graph.layer(static_cast<int>(i))->set_likelihood_variance(case_study.likelihood_variance);
			graph.layer(static_cast<int>(i))->fix_likelihood_variance();
		}
		SIDGP model(graph);
		model.train(train_iter, train_impute);
		bool nanflag = false;

		std::string m_path = results_path + exp + "-M.dat";
		std::string v_path = results_path + exp + "-V.dat";
		std::string mcs_path = results_path + exp;
		MatrixPair Z = model.predict(X_test, Y_test, mcs_path, nanflag, case_study.pred_iter, 96);
		TMatrix mean = Z.first;
		TMatrix var = Z.second;
		double nrmse = metrics::rmse(Y_test, mean, true);

		if (nanflag) {
			restart = true;
		}
		else {
			std::string e_path = results_path + "NRMSE.dat";
			std::cout << "NRMSE = " << nrmse << std::endl;
			write_data(m_path, mean);
			write_data(v_path, var);
			write_to_file(e_path, std::to_string(nrmse));

			// Plot
			MatrixPair Zplot = model.predict(X_plot, case_study.pred_iter, 96);
			TMatrix mplt = Zplot.first;
			TMatrix vplt = Zplot.second;
			std::string mplt_path = results_path + exp + "-M-PLT.dat";
			std::string vplt_path = results_path + exp + "-V-PLT.dat";
			write_data(mplt_path, mplt);
			write_data(vplt_path, vplt);


		}
	};


	if (!std::filesystem::exists("../results/case_1/"))
		std::filesystem::create_directory("../results/case_1/");
	// ../results/case_1/analytic2
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem);
	// ../results/case_1/analytic2/25
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train));
	// ../results/case_1/analytic2/25/100
	if (!std::filesystem::exists("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter)))
		std::filesystem::create_directory("../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter));

	// ../results/case_1/analytic2/25/100/100 || TRAIN_ITER/TRAIN_IMPUTE
	std::string main_results_path = "../results/case_1/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(train_iter) + "/" + std::to_string(train_impute);
	if (!std::filesystem::exists(main_results_path)) std::filesystem::create_directory(main_results_path);
	unsigned int ii = case_study.start;
	while (true) {
		std::cout << "================= " << "Running " << case_study.problem << "-" << case_study.n_train << " :"
			<< train_iter << "-" << train_impute << "================= " << std::endl;
		bool restart = false;
		std::cout << "================= " << "" << " REP " << ii << " ================" << std::endl;
		// ../results/case_1/analytic2/25/100/100/1 .... 25 || TRAIN_ITER/TRAIN_IMPUTE/REP
		std::string results_path = main_results_path + "/" + std::to_string(ii) + "/";
		if (!std::filesystem::exists(results_path)) std::filesystem::create_directory(results_path);
		run_problem(results_path, std::to_string(case_study.experiment), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else ii++;
		if (ii == case_study.finish) break;

	}
}
void case2(Case& case_study) {
	std::cout << "Running " << case_study.problem << " : " << case_study.n_train << std::endl;

	auto run_problem = [&case_study](std::string sp_path, std::string results_path, std::string exp, bool& restart) {
		TMatrix X_train = read_data(sp_path + "Xsc_train.dat");
		TMatrix Y_train = read_data(sp_path + "Y_train.dat");

		TMatrix X_test = read_data(sp_path + "Xsc_test.dat");
		TMatrix Y_test = read_data(sp_path + "Y_test.dat");

		Graph graph(std::make_pair(X_train, Y_train), 1);
		for (unsigned int i = 0; i < graph.n_layers; ++i) {
			TVector ls = TVector::Constant(X_train.cols(), 1.0);
			graph.layer(static_cast<int>(i))->set_kernels(TKernel::TMatern52, ls);
			graph.layer(static_cast<int>(i))->set_likelihood_variance(case_study.likelihood_variance);
			graph.layer(static_cast<int>(i))->fix_likelihood_variance();
		}
		SIDGP model(graph);
		model.train(case_study.train_iter, case_study.train_impute);
		bool nanflag = false;
		MatrixPair Z = model.predict(X_test, Y_test, nanflag, case_study.pred_iter, 48);
		TMatrix mean = Z.first;
		TMatrix var = Z.second;
		double nrmse = metrics::rmse(Y_test, mean, true);

		if (nanflag) {
			restart = true;
		}
		else {
			std::string e_path = results_path + "NRMSE.dat";
			std::cout << "NRMSE = " << nrmse << std::endl;

			std::string m_path = results_path + exp + "-M.dat";
			std::string v_path = results_path + exp + "-V.dat";
			write_data(m_path, mean);
			write_data(v_path, var);
			write_to_file(e_path, std::to_string(nrmse));
		}
	};

	if (!std::filesystem::exists("../results/case_2/"))
		std::filesystem::create_directory("../results/case_2/");
	// ../results/case_2/airfoil
	if (!std::filesystem::exists("../results/case_2/" + case_study.problem))
		std::filesystem::create_directory("../results/case_2/" + case_study.problem);
	// ../results/case_2/airfoil/40
	if (!std::filesystem::exists("../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train)))
		std::filesystem::create_directory("../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train));


	unsigned int ii = case_study.start;
	while (true) {
		bool restart = false;
		std::cout << "================= " << "" << " SAMP PLAN " << ii << " ================" << std::endl;
		std::string results_path;
		std::string data_path;
		if (!case_study.output.empty()) {
			if (!std::filesystem::exists("../datasets/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii)))
				std::filesystem::create_directory("../datasets/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii));

			if (!std::filesystem::exists("../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii)))
				std::filesystem::create_directory("../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii));

			data_path = "../datasets/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/" + case_study.output + "/";
			results_path = "../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/" + case_study.output + "/";
		}
		else {
			data_path = "../datasets/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/";
			results_path = "../results/case_2/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/";
		}
		if (!std::filesystem::exists(results_path)) std::filesystem::create_directory(results_path);
		run_problem(data_path, results_path, std::to_string(case_study.experiment), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else ii++;
		if (ii == case_study.finish) break;
	}
	std::cout << "End " << case_study.problem << " : " << case_study.n_train << std::endl;
}
void case3(Case& case_study) {
	// Case 2 + Heteroskedastic
	std::cout << "Running " << case_study.problem << " : " << case_study.n_train << std::endl;

	auto run_problem = [&case_study](std::string sp_path, std::string results_path, std::string exp, bool& restart) {
		TMatrix X_train = read_data(sp_path + "Xsc_train.dat");
		TMatrix Y_train = read_data(sp_path + "Y_train.dat");

		TMatrix X_test = read_data(sp_path + "Xsc_test.dat");
		TMatrix Y_test = read_data(sp_path + "Y_test.dat");

		Graph graph(std::make_pair(X_train, Y_train), 1);
		if (X_train.cols() > 2) graph.layer(-2)->remove_nodes(X_train.cols() - 2);
		else graph.layer(-2)->add_node(1);		
		for (unsigned int i = 0; i < graph.n_layers; ++i) {
			TVector ls = TVector::Constant(X_train.cols(), 1.0);
			graph.layer(static_cast<int>(i))->set_kernels(TKernel::TSquaredExponential, ls);
			graph.layer(static_cast<int>(i))->set_likelihood_variance(case_study.likelihood_variance);
			graph.layer(static_cast<int>(i))->fix_likelihood_variance();
		}
		graph.layer(0)->fix_scale();
		graph.layer(2)->set_likelihood(LLF::Heteroskedastic);
		SIDGP model(graph);
		model.train(case_study.train_iter, case_study.train_impute);
		bool nanflag = false;
		MatrixPair Z = model.predict(X_test, Y_test, nanflag, case_study.pred_iter, 96);
		TMatrix mean = Z.first;
		TMatrix var = Z.second;
		double nrmse = metrics::rmse(Y_test, mean, true);
		if (nanflag) {
			restart = true;
		}
		else {
			std::string e_path = results_path + "NRMSE.dat";
			std::cout << "NRMSE = " << nrmse << std::endl;

			std::string m_path = results_path + exp + "-M.dat";
			std::string v_path = results_path + exp + "-V.dat";
			write_data(m_path, mean);
			write_data(v_path, var);
			write_to_file(e_path, std::to_string(nrmse));

			if (case_study.plot) {
				TMatrix X_plot = read_data(sp_path + "X_plot.dat");
				MatrixPair Zplot = model.predict(X_plot, case_study.pred_iter, 96);
				TMatrix mplot = Zplot.first;
				TMatrix vplot = Zplot.second;
				std::string mplt_path = results_path + exp + "-PLT-M.dat";
				std::string vplt_path = results_path + exp + "-PLT-V.dat";
				write_data(mplt_path, mplot);
				write_data(vplt_path, vplot);
			}

		}
	};

	if (!std::filesystem::exists("../results/case_3/"))
		std::filesystem::create_directory("../results/case_3/");
	// ../results/case_3/airfoil
	if (!std::filesystem::exists("../results/case_3/" + case_study.problem))
		std::filesystem::create_directory("../results/case_3/" + case_study.problem);
	// ../results/case_3/airfoil/40
	if (!std::filesystem::exists("../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train)))
		std::filesystem::create_directory("../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train));


	unsigned int ii = case_study.start;
	while (true) {
		bool restart = false;
		std::cout << "================= " << "" << " SAMP PLAN " << ii << " ================" << std::endl;
		std::string results_path;
		std::string data_path;
		if (!case_study.output.empty()) {
			if (!std::filesystem::exists("../datasets/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii)))
				std::filesystem::create_directory("../datasets/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii));

			if (!std::filesystem::exists("../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii)))
				std::filesystem::create_directory("../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii));

			data_path = "../datasets/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/" + case_study.output + "/";
			results_path = "../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/" + case_study.output + "/";
		}
		else {
			data_path = "../datasets/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/";
			results_path = "../results/case_3/" + case_study.problem + "/" + std::to_string(case_study.n_train) + "/" + std::to_string(ii) + "/";
		}
		if (!std::filesystem::exists(results_path)) std::filesystem::create_directory(results_path);
		run_problem(data_path, results_path, std::to_string(case_study.experiment), restart);
		if (restart) {
			std::system("clear");
			continue;
		}
		else ii++;
		if (ii == case_study.finish) break;
	}
	std::cout << "End " << case_study.problem << " : " << case_study.n_train << std::endl;
}

void case1() {

	std::vector<int> train_iter = { 500 };
	// std::vector<int> train_impute = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
	std::vector<int> train_impute = { 900 };

	// Experiment 1: 1E-6
	// Experiment 2: 1E-3
	// Experiment 3: PREDICT
	{
		Case AN_C1_1("analytic2");
		AN_C1_1.n_train = 25;
		AN_C1_1.experiment = 3;
		AN_C1_1.start = 1;
		AN_C1_1.finish = 2;
		AN_C1_1.pred_iter = 500;
		AN_C1_1.likelihood_variance = 1E-3;

		for (int ii : train_iter) {
			for (int jj : train_impute) {
				case1_prediction(AN_C1_1, ii, jj);
			}
		}
	}
	{
		// Case AN_C1_2("analytic2");
		// AN_C1_2.n_train = 50;
		// AN_C1_2.experiment = 1;
		// AN_C1_2.start = 1;
		// AN_C1_2.finish = 21;
		// AN_C1_2.pred_iter = 100;
		// AN_C1_2.likelihood_variance = 1E-3;

		// for (int ii : train_iter) {
		// 	for (int jj : train_impute) {
		// 		case1(AN_C1_2, ii, jj);
		// 	}
		// }
	}
}

void airfoil_case2() {
	{
		Case AN_C2_1("airfoil");
		AN_C2_1.n_train = 20;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}
	{
		Case AN_C2_1("airfoil");
		AN_C2_1.n_train = 40;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}
	{
		Case AN_C2_1("airfoil");
		AN_C2_1.n_train = 60;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}

}

void engine_case2() {
	{
		Case AN_C2_1("engine");
		AN_C2_1.n_train = 20;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}
	{
		Case AN_C2_1("engine");
		AN_C2_1.n_train = 40;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}
	{
		Case AN_C2_1("engine");
		AN_C2_1.n_train = 60;
		AN_C2_1.experiment = 1;
		AN_C2_1.start = 1;
		AN_C2_1.finish = 26;
		AN_C2_1.train_iter = 500;
		AN_C2_1.train_impute = 900;
		AN_C2_1.pred_iter = 200;
		AN_C2_1.likelihood_variance = 1E-3;
		case2(AN_C2_1);
	}

}

void nrel_case3() {
	{
		Case study("nrel", "RootMyc1");
		study.n_train = 40;
		study.experiment = 1;
		study.start = 1;
		study.finish = 3;
		study.train_iter = 500;
		study.train_impute = 900;
		study.pred_iter = 200;
		study.likelihood_variance = 1E-3;
		case3(study);
	}
}

void motorcycle_case3() {
	{
		Case study("motorcycle");
		study.n_train = 27;
		study.experiment = 1;
		study.start = 1;
		study.finish = 2;
		study.train_iter = 500;
		study.train_impute = 10;
		study.pred_iter = 200;
		study.likelihood_variance = 1E-8;
		study.plot = true;
		case3(study);
		//debug_case3(study);
	}
}

int main() {
	//case1();
	//airfoil_case2();
	//engine_case2();
	motorcycle_case3();
	return 0;
}