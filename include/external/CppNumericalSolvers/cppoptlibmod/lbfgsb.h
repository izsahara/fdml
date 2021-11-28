// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
#ifndef INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
#define INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>
#include <Eigen/LU>
#include "solver.h"
#include "more_thuente.h"

namespace cppoptlib::solver::LBFGSB {

    class LBFGSB : public Solver {
        
    private:
        int m = 5;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        LBFGSB() : Solver() {}
        LBFGSB(const SolverState& stopping_state) : Solver(stopping_state) {}
        
        void initialize_solver(const FunctionState &initial_state) override {
            dim_ = initial_state.x.rows();
            theta_ = 1.0;
            W_ = TMatrix::Zero(dim_, 0);
            M_ = TMatrix::Zero(0, 0);
            y_history_ = TMatrix::Zero(dim_, 0);
            s_history_ = TMatrix::Zero(dim_, 0);
        }

        void SetLowerBound(const TVector& lb) { lower_bound_ = lb; }
        void SetUpperBound(const TVector& ub) { upper_bound_ = ub; }

        FunctionState step(const Function &function, const FunctionState &current, const SolverState & /*state*/) override 
        {
            // STEP 2: compute the cauchy point
            TVector cauchy_point = TVector::Zero(dim_);
            TVector c = TVector::Zero(W_.cols());
            GetGeneralizedCauchyPoint(current, &cauchy_point, &c);

            // STEP 3: compute a search direction d_k by the primal method for the
            // sub-problem
            const TVector subspace_min = SubspaceMinimization(current, cauchy_point, c);

            // STEP 4: perform linesearch and STEP 5: compute gradient
            double alpha_init = 1.0;
            const double rate = linesearch::MoreThuente::search(current.x, subspace_min - current.x, function, alpha_init);

            // update current guess and function information
            const TVector x_next = current.x - rate * (current.x - subspace_min);
            // if current solution is out of bound, we clip it
            const TVector clipped_x_next = x_next.cwiseMin(upper_bound_).cwiseMax(lower_bound_);

            const FunctionState next = function.evaluate(clipped_x_next, 1);

            // prepare for next iteration
            const TVector new_y = next.gradient - current.gradient;
            const TVector new_s = next.x - current.x;

            // STEP 6:
            const double test = fabs(new_s.dot(new_y));
            if (test > 1e-7 * new_y.squaredNorm()) 
            {
                if (y_history_.cols() < m) 
                {
                    y_history_.conservativeResize(dim_, y_history_.cols() + 1);
                    s_history_.conservativeResize(dim_, s_history_.cols() + 1);
                } 
                else {
                    y_history_.leftCols(m - 1) = y_history_.rightCols(m - 1).eval();
                    s_history_.leftCols(m - 1) = s_history_.rightCols(m - 1).eval();
                }
                y_history_.rightCols(1) = new_y;
                s_history_.rightCols(1) = new_s;
                // STEP 7:
                theta_ = (double)(new_y.transpose() * new_y) / (new_y.transpose() * new_s);
                W_ = TMatrix::Zero(y_history_.rows(),
                                    y_history_.cols() + s_history_.cols());
                W_ << y_history_, (theta_ * s_history_);
                TMatrix A = s_history_.transpose() * y_history_;
                TMatrix L = A.template triangularView<Eigen::StrictlyLower>();
                TMatrix MM(A.rows() + L.rows(), A.rows() + L.cols());
                TMatrix D = -1 * A.diagonal().asDiagonal();
                MM << D, L.transpose(), L,
                    ((s_history_.transpose() * s_history_) * theta_);
                M_ = MM.inverse();
            }

            return next;
        }        

    private:
        std::vector<int> SortIndexes(const std::vector<std::pair<int, double>> &v) {
            std::vector<int> idx(v.size());
            for (size_t i = 0; i != idx.size(); ++i) idx[i] = v[i].first;
            sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1].second < v[i2].second; });
            return idx;
        }        
    
        void GetGeneralizedCauchyPoint(const FunctionState &current, TVector *x_cauchy, TVector *c) 
        {
            constexpr double max_value = std::numeric_limits<double>::max();
            constexpr double epsilon = std::numeric_limits<double>::epsilon();

            // Given x,l,u,g, and B = \theta_ I-WMW
            // {all t_i} = { (idx,value), ... }
            // TODO(patwie): use "std::set" ?
            std::vector<std::pair<int, double>> set_of_t;
            // The feasible set is implicitly given by "set_of_t - {t_i==0}".
            TVector d = -current.gradient;
            // n operations
            for (int j = 0; j < dim_; j++) {
                if (current.gradient(j) == 0) {
                set_of_t.push_back(std::make_pair(j, max_value));
                } else {
                double tmp = 0;
                if (current.gradient(j) < 0) {
                    tmp = (current.x(j) - upper_bound_(j)) / current.gradient(j);
                } else {
                    tmp = (current.x(j) - lower_bound_(j)) / current.gradient(j);
                }
                set_of_t.push_back(std::make_pair(j, tmp));
                if (tmp == 0) d(j) = 0;
                }
            }
            // sortedindices [1,0,2] means the minimal element is on the 1-st entry
            std::vector<int> sorted_indices = SortIndexes(set_of_t);
            *x_cauchy = current.x;
            // Initialize
            // p :=     W^double*p
            TVector p = (W_.transpose() * d);  // (2mn operations)
            // c :=     0
            *c = TVector::Zero(W_.cols());
            // f' :=    g^double*d = -d^Td
            double f_prime = -d.dot(d);  // (n operations)
            // f'' :=   \theta_*d^double*d-d^double*W*M*W^double*d = -\theta_*f'
            // -
            // p^double*M*p
            double f_doubleprime = (double)(-1.0 * theta_) * f_prime -
                                        p.dot(M_ * p);  // (O(m^2) operations)
            f_doubleprime = std::max<double>(epsilon, f_doubleprime);
            double f_dp_orig = f_doubleprime;
            // \delta t_min :=  -f'/f''
            double dt_min = -f_prime / f_doubleprime;
            // t_old :=     0
            double t_old = 0;
            // b :=     argmin {t_i , t_i >0}
            int i = 0;
            for (int j = 0; j < dim_; j++) {
                i = j;
                if (set_of_t[sorted_indices[j]].second > 0) break;
            }
            int b = sorted_indices[i];
            // see below
            // t                    :=  min{t_i : i in F}
            double t = set_of_t[b].second;
            // \delta double             :=  t - 0
            double dt = t;
            // examination of subsequent segments
            while ((dt_min >= dt) && (i < dim_)) {
                if (d(b) > 0)
                (*x_cauchy)(b) = upper_bound_(b);
                else if (d(b) < 0)
                (*x_cauchy)(b) = lower_bound_(b);
                // z_b = x_p^{cp} - x_b
                const double zb = (*x_cauchy)(b)-current.x(b);
                // c   :=  c +\delta t*p
                *c += dt * p;
                // cache
                TVector wbt = W_.row(b);
                f_prime += dt * f_doubleprime +
                            current.gradient(b) * current.gradient(b) +
                            theta_ * current.gradient(b) * zb -
                            current.gradient(b) * wbt.transpose() * (M_ * *c);
                f_doubleprime +=
                    double(-1.0) * theta_ * current.gradient(b) * current.gradient(b) -
                    double(2.0) * (current.gradient(b) * (wbt.dot(M_ * p))) -
                    current.gradient(b) * current.gradient(b) * wbt.transpose() *
                        (M_ * wbt);
                f_doubleprime = std::max<double>(epsilon * f_dp_orig, f_doubleprime);
                p += current.gradient(b) * wbt.transpose();
                d(b) = 0;
                dt_min = -f_prime / f_doubleprime;
                t_old = t;
                ++i;
                if (i < dim_) {
                b = sorted_indices[i];
                t = set_of_t[b].second;
                dt = t - t_old;
                }
            }
            dt_min = std::max<double>(dt_min, double{0});
            t_old += dt_min;

            (*x_cauchy)(sorted_indices) =
                current.x(sorted_indices) + t_old * d(sorted_indices);

            *c += dt_min * p;
        }

        double FindAlpha(const TVector &x_cp, const TVector &du, const std::vector<int> &free_variables) 
        {
            double alphastar = 1;
            //const unsigned int n = free_variables.size(); [CHANGED]
            std::size_t n = free_variables.size();
            assert(du.rows() == n);

            for (unsigned int i = 0; i < n; i++) {
                if (du(i) > 0) {
                alphastar = std::min<double>(
                    alphastar,
                    (upper_bound_(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                        du(i));
                } else {
                alphastar = std::min<double>(
                    alphastar,
                    (lower_bound_(free_variables.at(i)) - x_cp(free_variables.at(i))) /
                        du(i));
                }
            }
            return alphastar;
        }

        TVector SubspaceMinimization(const FunctionState &current, const TVector &x_cauchy, const TVector &c) {
            const double theta_inverse = 1 / theta_;

            std::vector<int> free_variables_index;
            for (int i = 0; i < x_cauchy.rows(); i++) {
                if ((x_cauchy(i) != upper_bound_(i)) && (x_cauchy(i) != lower_bound_(i))) {
                free_variables_index.push_back(i);
                }
            }
            //const int free_var_count = free_variables_index.size(); [CHANGED]
            std::size_t free_var_count = free_variables_index.size();
            const TMatrix WZ =
                W_(free_variables_index, Eigen::indexing::all).transpose();

            const TVector rr =
                (current.gradient + theta_ * (x_cauchy - current.x) - W_ * (M_ * c));
            // r=r(free_variables);
            const TMatrix r = rr(free_variables_index);

            // STEP 2: "v = w^T*Z*r" and STEP 3: "v = M*v"
            TVector v = M_ * (WZ * r);
            // STEP 4: N = 1/theta*W^T*Z*(W^T*Z)^T
            TMatrix N = theta_inverse * WZ * WZ.transpose();
            // N = I - MN
            N = TMatrix::Identity(N.rows(), N.rows()) - M_ * N;
            // STEP: 5
            // v = N^{-1}*v
            if (v.size() > 0) {
                v = N.lu().solve(v);
            }
            // STEP: 6
            // HERE IS A MISTAKE IN THE ORIGINAL PAPER!
            const TVector du =
                -theta_inverse * r - theta_inverse * theta_inverse * WZ.transpose() * v;
            // STEP: 7
            const double alpha_star =
                FindAlpha(x_cauchy, du, free_variables_index);
            // STEP: 8
            TVector dStar = alpha_star * du;
            TVector subspace_min = x_cauchy.eval();
            for (int i = 0; i < free_var_count; i++) {
                subspace_min(free_variables_index[i]) =
                    subspace_min(free_variables_index[i]) + dStar(i);
            }
            return subspace_min;
        }

    private:
        //int dim_; [CHANGED]
        Eigen::Index dim_;
        double theta_;
        TVector lower_bound_ = TVector::Zero(1);
        TVector upper_bound_ = TVector::Zero(1);
        TMatrix M_;
        TMatrix W_;
        TMatrix y_history_;
        TMatrix s_history_;

    };



}

#endif  // INCLUDE_CPPOPTLIB_SOLVER_LBFGSB_H_
