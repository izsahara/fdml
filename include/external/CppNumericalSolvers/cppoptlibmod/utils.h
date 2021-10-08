#ifndef INCLUDE_CPPOPTLIB_UTILS_H
#define INCLUDE_CPPOPTLIB_UTILS_H

#include "base_opt.h"

//namespace cppoptlib::utils {
//
//    class Function;
//    /* utils/derivatives.h */
//    // Approximates the gradient of the given function in x0.  
//    void ComputeFiniteGradient(const Function& function, const typename TVector& x0, typename TVector* grad, const int accuracy = 0)
//    {
//        // The 'accuracy' can be 0, 1, 2, 3.
//        constexpr double eps = 2.2204e-6;
//        static const std::array<std::vector<double>, 4> coeff = {
//            {{1, -1},
//            {1, -8, 8, -1},
//            {-1, 9, -45, 45, -9, 1},
//            {3, -32, 168, -672, 672, -168, 32, -3}} };
//        static const std::array<std::vector<double>, 4> coeff2 = {
//            {{1, -1},
//            {-2, -1, 1, 2},
//            {-3, -2, -1, 1, 2, 3},
//            {-4, -3, -2, -1, 1, 2, 3, 4}} };
//        static const std::array<double, 4> dd = { 2, 12, 60, 840 };
//        grad->resize(x0.rows());
//        TVector& x = const_cast<TVector&>(x0);
//
//        const int innerSteps = 2 * (accuracy + 1);
//        const double ddVal = dd[accuracy] * eps;
//
//        for (TIndex d = 0; d < x0.rows(); d++) {
//            (*grad)[d] = 0;
//            for (int s = 0; s < innerSteps; ++s)
//            {
//                double tmp = x[d];
//                x[d] += coeff2[accuracy][s] * eps;
//                (*grad)[d] += coeff[accuracy][s] * function(x);
//                x[d] = tmp;
//            }
//            (*grad)[d] /= ddVal;
//        }
//    }
//
//    // Approximates the hessian_t of the given function in x0.
//    void ComputeFiniteHessian(const Function& function, const typename TVector& x0, typename THessian* hessian, int accuracy = 0) {
//
//        constexpr double eps = std::numeric_limits<double>::epsilon() * 10e7;
//
//        hessian->resize(x0.rows(), x0.rows());
//        TVector& x = const_cast<TVector&>(x0);
//
//        if (accuracy == 0) {
//            for (TIndex i = 0; i < x0.rows(); i++) {
//                for (TIndex j = 0; j < x0.rows(); j++) {
//                    double tmpi = x[i];
//                    double tmpj = x[j];
//
//                    double f4 = function(x);
//                    x[i] += eps;
//                    x[j] += eps;
//                    double f1 = function(x);
//                    x[j] -= eps;
//                    double f2 = function(x);
//                    x[j] += eps;
//                    x[i] -= eps;
//                    double f3 = function(x);
//                    (*hessian)(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);
//
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                }
//            }
//        }
//        else {
//            /*
//                \displaystyle{{\frac{\partial^2{f}}{\partial{x0}\partial{y}}}\approx
//                \frac{1}{600\,h^2} \left[\begin{matrix}
//                -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
//                    63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
//                    44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
//                    74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
//                \end{matrix}\right] }
//            */
//            for (TIndex i = 0; i < x0.rows(); i++) {
//                for (TIndex j = 0; j < x0.rows(); j++) {
//                    double tmpi = x[i];
//                    double tmpj = x[j];
//
//                    double term_1 = 0;
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 1 * eps;
//                    x[j] += -2 * eps;
//                    term_1 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 2 * eps;
//                    x[j] += -1 * eps;
//                    term_1 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -2 * eps;
//                    x[j] += 1 * eps;
//                    term_1 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -1 * eps;
//                    x[j] += 2 * eps;
//                    term_1 += function(x);
//
//                    double term_2 = 0;
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -1 * eps;
//                    x[j] += -2 * eps;
//                    term_2 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -2 * eps;
//                    x[j] += -1 * eps;
//                    term_2 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 1 * eps;
//                    x[j] += 2 * eps;
//                    term_2 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 2 * eps;
//                    x[j] += 1 * eps;
//                    term_2 += function(x);
//
//                    double term_3 = 0;
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 2 * eps;
//                    x[j] += -2 * eps;
//                    term_3 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -2 * eps;
//                    x[j] += 2 * eps;
//                    term_3 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -2 * eps;
//                    x[j] += -2 * eps;
//                    term_3 -= function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 2 * eps;
//                    x[j] += 2 * eps;
//                    term_3 -= function(x);
//
//                    double term_4 = 0;
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -1 * eps;
//                    x[j] += -1 * eps;
//                    term_4 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 1 * eps;
//                    x[j] += 1 * eps;
//                    term_4 += function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += 1 * eps;
//                    x[j] += -1 * eps;
//                    term_4 -= function(x);
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//                    x[i] += -1 * eps;
//                    x[j] += 1 * eps;
//                    term_4 -= function(x);
//
//                    x[i] = tmpi;
//                    x[j] = tmpj;
//
//                    (*hessian)(i, j) =
//                        (-63 * term_1 + 63 * term_2 + 44 * term_3 + 74 * term_4) /
//                        (600.0 * eps * eps);
//                }
//            }
//        }
//    }
//
//    bool IsGradientCorrect(const Function& function, const typename TVector& x0, int accuracy = 3)
//    {
//        constexpr float tolerance = 1e-2;
//        const TIndex D = x0.rows();
//        TVector actual_gradient(D);
//        TVector expected_gradient(D);
//
//        function.gradient(x0, &actual_gradient);
//        ComputeFiniteGradient(function, x0, &expected_gradient, accuracy);
//
//        for (TIndex d = 0; d < D; ++d) {
//            double scale = std::max(static_cast<double>(std::max(fabs(actual_gradient[d]), fabs(expected_gradient[d]))), double(1.));
//            if (fabs(actual_gradient[d] - expected_gradient[d]) > tolerance * scale)
//                return false;
//        }
//        return true;
//    }
//
//    bool IsHessianCorrect(const Function& function, const typename TVector& x0, int accuracy = 3) {
//        constexpr float tolerance = 1e-1;
//        const TIndex D = x0.rows();
//        THessian actual_hessian = THessian::Zero(D, D);
//        THessian expected_hessian = THessian::Zero(D, D);
//        function.hessian(x0, &actual_hessian);
//        ComputeFiniteHessian(function, x0, &expected_hessian, accuracy);
//        for (TIndex d = 0; d < D; ++d) {
//            for (TIndex e = 0; e < D; ++e) {
//                double scale = std::max(static_cast<double>(std::max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e)))), double(1.));
//                if (fabs(actual_hessian(d, e) - expected_hessian(d, e)) >
//                    tolerance * scale)
//                    return false;
//            }
//        }
//        return true;
//    }
//}


#endif // !INCLUDE_CPPOPTLIB_UTILS_H
