// Copyright https://github.com/PatWie/CppNumericalSolvers, MIT license
#ifndef INCLUDE_CPPOPTLIB_FUNCTION_H_
#define INCLUDE_CPPOPTLIB_FUNCTION_H_

#include "base_opt.h"

namespace cppoptlib::function {

    struct FunctionState {
        int dim;
        int order;

        double value = 0;               // The objective value.
        TVector x;                       // The current input value in x.
        TVector gradient;                // The gradient in x.
        std::optional<TMatrix> hessian;  // The hessian in x;

        // TODO(patwie): There is probably a better way.
        FunctionState() : dim(-1), order(-1) {}

        FunctionState(const int dim, const int order) : dim(dim), order(order), x(TVector::Zero(dim)), gradient(TVector::Zero(dim)) 
        { if (order > 1) { hessian = std::optional<TMatrix>(TMatrix::Zero(dim, dim)); } }

        FunctionState(const FunctionState &rhs) { CopyState(rhs); }
        FunctionState operator=(const FunctionState &rhs) { CopyState(rhs); return *this; }

        void CopyState(const FunctionState &rhs) {
            assert(rhs.order > -1);
            dim = rhs.dim;
            order = rhs.order;
            value = rhs.value;
            x = rhs.x.eval();
            if (order >= 1) { gradient = rhs.gradient.eval(); }
            if ((order >= 2) && rhs.hessian) { 
                hessian = std::optional<TMatrix>(rhs.hessian->eval());
            }
        }
    };

    class Function {

    public:
        Function() = default;
        virtual ~Function() = default;

        // Computes the value of a function.
        virtual double operator()(const TVector &x) const = 0;
        
        // Computes the gradient of a function.
        virtual void gradient(const TVector &x, TVector *grad) const { ComputeFiniteGradient(*this, x, grad); }
        // Computes the hessian of a function.
        virtual void hessian(const TVector &x, THessian *hessian) const { ComputeFiniteHessian(*this, x, hessian); }

        virtual int order() const { return 1; }
        virtual FunctionState evaluate(const TVector &x, const int order = 2)  const {
            //FunctionState state(x.rows() , order); [CHANGED]
            FunctionState state(static_cast<const int>(x.rows()) , order);
            state.value = this->operator()(x);
            state.x = x;
            if (order >= 1) {
                this->gradient(x, &state.gradient);
            }
            if ((order >= 2) && (state.hessian)) {
                this->hessian(x, &*(state.hessian));
            }
            return state;
        }
    
    private:

        static void ComputeFiniteGradient(const Function& function, const TVector& x0, TVector* grad, const int accuracy = 0)
        {
            // The 'accuracy' can be 0, 1, 2, 3.
            constexpr double eps = 2.2204e-6;
            static const std::array<std::vector<double>, 4> coeff = {
                {{1, -1},
                {1, -8, 8, -1},
                {-1, 9, -45, 45, -9, 1},
                {3, -32, 168, -672, 672, -168, 32, -3}} };
            static const std::array<std::vector<double>, 4> coeff2 = {
                {{1, -1},
                {-2, -1, 1, 2},
                {-3, -2, -1, 1, 2, 3},
                {-4, -3, -2, -1, 1, 2, 3, 4}} };
            static const std::array<double, 4> dd = { 2, 12, 60, 840 };
            grad->resize(x0.rows());
            TVector& x = const_cast<TVector&>(x0);

            const int innerSteps = 2 * (accuracy + 1);
            const double ddVal = dd[accuracy] * eps;

            for (TIndex d = 0; d < x0.rows(); d++) {
                (*grad)[d] = 0;
                for (int s = 0; s < innerSteps; ++s)
                {
                    double tmp = x[d];
                    x[d] += coeff2[accuracy][s] * eps;
                    (*grad)[d] += coeff[accuracy][s] * function(x);
                    x[d] = tmp;
                }
                (*grad)[d] /= ddVal;
            }
        }
        static void ComputeFiniteHessian(const Function& function, const TVector& x0, THessian* hessian, int accuracy = 0) {

            constexpr double eps = std::numeric_limits<double>::epsilon() * 10e7;

            hessian->resize(x0.rows(), x0.rows());
            TVector& x = const_cast<TVector&>(x0);

            if (accuracy == 0) {
                for (TIndex i = 0; i < x0.rows(); i++) {
                    for (TIndex j = 0; j < x0.rows(); j++) {
                        double tmpi = x[i];
                        double tmpj = x[j];

                        double f4 = function(x);
                        x[i] += eps;
                        x[j] += eps;
                        double f1 = function(x);
                        x[j] -= eps;
                        double f2 = function(x);
                        x[j] += eps;
                        x[i] -= eps;
                        double f3 = function(x);
                        (*hessian)(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);

                        x[i] = tmpi;
                        x[j] = tmpj;
                    }
                }
            }
            else {
                /*
                    \displaystyle{{\frac{\partial^2{f}}{\partial{x0}\partial{y}}}\approx
                    \frac{1}{600\,h^2} \left[\begin{matrix}
                    -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
                        63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
                        44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
                        74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
                    \end{matrix}\right] }
                */
                for (TIndex i = 0; i < x0.rows(); i++) {
                    for (TIndex j = 0; j < x0.rows(); j++) {
                        double tmpi = x[i];
                        double tmpj = x[j];

                        double term_1 = 0;
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 1 * eps;
                        x[j] += -2 * eps;
                        term_1 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 2 * eps;
                        x[j] += -1 * eps;
                        term_1 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -2 * eps;
                        x[j] += 1 * eps;
                        term_1 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -1 * eps;
                        x[j] += 2 * eps;
                        term_1 += function(x);

                        double term_2 = 0;
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -1 * eps;
                        x[j] += -2 * eps;
                        term_2 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -2 * eps;
                        x[j] += -1 * eps;
                        term_2 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 1 * eps;
                        x[j] += 2 * eps;
                        term_2 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 2 * eps;
                        x[j] += 1 * eps;
                        term_2 += function(x);

                        double term_3 = 0;
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 2 * eps;
                        x[j] += -2 * eps;
                        term_3 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -2 * eps;
                        x[j] += 2 * eps;
                        term_3 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -2 * eps;
                        x[j] += -2 * eps;
                        term_3 -= function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 2 * eps;
                        x[j] += 2 * eps;
                        term_3 -= function(x);

                        double term_4 = 0;
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -1 * eps;
                        x[j] += -1 * eps;
                        term_4 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 1 * eps;
                        x[j] += 1 * eps;
                        term_4 += function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += 1 * eps;
                        x[j] += -1 * eps;
                        term_4 -= function(x);
                        x[i] = tmpi;
                        x[j] = tmpj;
                        x[i] += -1 * eps;
                        x[j] += 1 * eps;
                        term_4 -= function(x);

                        x[i] = tmpi;
                        x[j] = tmpj;

                        (*hessian)(i, j) =
                            (-63 * term_1 + 63 * term_2 + 44 * term_3 + 74 * term_4) /
                            (600.0 * eps * eps);
                    }
                }
            }
        }
    };


}
#endif  // INCLUDE_CPPOPTLIB_FUNCTION_H_
