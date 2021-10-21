#ifndef UTILITIES_H
#define UTILITIES_H

#include <fstream>
#include <filesystem>
#include <numeric>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "base.h"

namespace fdml::utilities {

    void minmax(TMatrix& X) {
        TMatrix num = X.rowwise() - X.colwise().minCoeff();
        TVector den = X.colwise().maxCoeff() - X.colwise().minCoeff();
        X = num.array().rowwise() / den.array().transpose();
    }

    void scale_to_range(TMatrix& X, const TVector& lower, TVector& upper) {
        minmax(X);
        upper.array() -= lower.array();
        for (Eigen::Index c = 0; c < X.cols(); ++c) {
            X.col(c).array() *= upper(c);
            X.col(c).array() += lower(c);
        }
    }

    void standardize(TMatrix& X) {
        TMatrix Xmean = X.rowwise() - X.colwise().mean();
        X = (Xmean).array().rowwise() / (((Xmean).array().square().colwise().sum()) / ((Xmean).rows())).sqrt();
    }
    double rmse(TMatrix& Ypred, const TMatrix& Ytrue) {
        return sqrt((Ypred - Ytrue).array().square().sum() / Ytrue.rows());
    }
    TMatrix pnorm(const TMatrix& X) {
        // 0.5*(1+erf(x/sqrt(2)))
        TMatrix P = Eigen::erf(X.array() / sqrt(2));
        return 0.5 * (1.0 + P.array());
    }

    // Scaler
    struct Scaler {
        Scaler() = default;

        TMatrix scale(const TMatrix& X) {
            min = X.colwise().minCoeff();
            max = X.colwise().maxCoeff();
            TRVector range = max - min;
            TMatrix Xscaled = (X.array().rowwise() - min.array()).rowwise() / range.array();
            return Xscaled;
        }

        TRVector min, max;
        TRVector mean, stdev;

    };

    // Distance Utilities
    void euclidean_distance(const TMatrix& X1, const TMatrix& X2, TMatrix& D, bool squared = false) {
        // Compute squared euclidean distance |X1(i) - X2(j)^T|^2 = |X1(i)|^2 + |X2(j)|^2 - (2 * X1(i)^T X2(j)) by default
        D = ((-2.0 * (X1 * X2.transpose())).colwise()
            + X1.rowwise().squaredNorm()).rowwise()
            + X2.rowwise().squaredNorm().transpose();
        //if ((D.array() < 0.0).any()) { D = abs(D.array()); }
        if (!squared) { D = (D.array().sqrt()).matrix(); }
    }
    
    void euclidean_distance(const TMatrix& X1, const TMatrix& X2, TMatrix& D, const double& jitter, bool squared = false) {
        // Compute squared euclidean distance |X1(i) - X2(j)^T|^2 = |X1(i)|^2 + |X2(j)|^2 - (2 * X1(i)^T X2(j)) by default
        D = ((-2.0 * (X1 * X2.transpose())).colwise()
            + X1.rowwise().squaredNorm()).rowwise()
            + X2.rowwise().squaredNorm().transpose();
        
        // Add Jitter, Numerical Issues involved with Matern Kernels
        D.array() += jitter;

        if (!squared) {
            D = (D.array().sqrt()).matrix();
        }
    }    
    
    TMatrix distance_metric(const TMatrix& X1, const TMatrix& X2, std::string metric) {
        TMatrix D;
        if (metric == "sqeuclidean") {
            euclidean_distance(X1, X2, D, true);            
        }
        else if (metric == "euclidean") {
            euclidean_distance(X1, X2, D, false);
        }
        else { throw std::runtime_error("Unrecognized Metric"); }
        return D;
    }
    void pdist(const TVector& X1, const TVector& X2, TMatrix& D) {
        // Pairwise distance between two Vectors
        D.resize(X1.rows(), X2.rows());
        for (int i = 0; i < X1.rows(); i++) {
            for (int j = 0; j < X2.rows(); j++) {
                D(i, j) = X1(i) - X2(j);
            }
        }
    }
    void pdist(const TMatrix& X1, const TMatrix& X2, std::vector<TMatrix>& D, bool squared = true) {
        // Pairwise distance between two each column/dimension of 2 Matrices
        for (int i = 0; i < X1.cols(); i++) {
            TMatrix tmp(X1.rows(), X2.rows());
            for (int j = 0; j < X1.rows(); j++) {
                for (int k = 0; k < X2.rows(); k++) {
                    tmp(j, k) = X1.col(i)(j) - X2.col(i)(k);
                }
            }
            if (squared) { D.push_back(pow(tmp.array(), 2)); }
            else { D.push_back(tmp); }

        }
    }

    // Print Utilities
    enum class Justified { LEFT, CENTRE, RIGHT };
    void position_print(Justified pos, std::string s, int linelength)
    {
        int spaces = 0;
        switch (pos)
        {
            case Justified::CENTRE: spaces = (linelength - static_cast<int>(s.size())) / 2; break;
            case Justified::RIGHT: spaces = linelength - static_cast<int>(s.size()); break;
        }
        if (spaces > 0) { std::cout << std::string(spaces, ' '); }
        std::cout << s << '\n';
    }
    void position_print(Justified pos, std::string s, int linelength, int index)
    {
        int spaces = 0;
        switch (pos)
        {
            case Justified::CENTRE: spaces = (linelength - static_cast<int>(s.size())) / 2; break;
            case Justified::RIGHT: spaces = linelength - static_cast<int>(s.size()); break;
        }
        if (spaces > 0) { std::cout << std::string(spaces, ' '); }
        std::cout << s << " " << index << '\n';
    }
    void position_print(std::vector<Justified> pos, std::vector<std::string> s, int linelength)
    {
        int spaces = 0;
        for (std::vector<Justified>::size_type i = 0; i != pos.size(); ++i) {
            switch (pos[i]) {
                case Justified::CENTRE: spaces = (linelength - static_cast<int>(s.size())) / 2; break;
                case Justified::RIGHT: spaces = linelength - static_cast<int>(s.size()); break;
            }
            if (spaces > 0) { std::cout << std::string(spaces, ' '); }
            if (i == pos.size() - 1) { std::cout << s[i] << std::endl; }
            else { std::cout << s[i]; }            
        }
    }

    // Eigen Indexing
    std::vector<int> arange(int a, int b) {
        if (b < a) { throw std::runtime_error("b < a"); }
        std::vector<int> range_;
        for (int i = a; i < b; ++i) {
            range_.push_back(i);
        }
        return range_;
    }
    template<typename Func>
    struct lambda_as_visitor_wrapper : Func {
        lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
        template<typename S, typename I>
        void init(const S& v, I i, I j) { return Func::operator()(v, i, j); }
    };
    template<typename Mat, typename Func>
    void visit_lambda(const Mat& m, const Func& f)
    {
        lambda_as_visitor_wrapper<Func> visitor(f);
        m.visit(visitor);
    }    

    template<typename TBool>
    TBool get_missing_index(const TMatrix& X1);
    template<>
    BoolVector get_missing_index(const TMatrix& X1) {
        BoolVector missing(X1.rows());
        visit_lambda(X1, [&missing, &X1](double v, int i, int j) {
            if (j == 0) {
                if (X1.row(i).array().isNaN().any()) { missing(i) = true; return; }
                else { missing(i) = false; }
            }
            else { return; }
            }
        );
        return missing;
    }
    template<>
    BoolMatrix get_missing_index(const TMatrix& X1) {
        BoolMatrix missing(X1.rows(), X1.cols());
        visit_lambda(X1, [&missing](double v, int i, int j)
            {
                if (std::isnan(v)) { missing(i, j) = true; }
                else { missing(i, j) = false; }
            });
        return missing;
    }
      
    void mask_matrix_in_place(TMatrix& X2, const TMatrix& X1, const BoolVector& mask, const bool& inverse, const int& axis) {
        /*
        * Masks a matrix X1 to get X2:
        *   -> X2 = X1[mask, :] / if (inverse) -> X2 = X1[~mask, :] (axis = 0 Row Wise)
        *   -> X2 = X1[:, mask] / if (inverse) -> X2 = X1[:, ~mask] (axis = 1 Col Wise)
        */
        int i, j;
        if (axis == 0) {
            // Row Wise
            if (inverse) {
                X2.resize(X1.rows() - mask.count(), X1.cols());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { continue; }
                    else { X2.row(j) = X1.row(i); ++j; }
                }
            }
            else {
                X2.resize(mask.count(), X1.cols());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { X2.row(j) = X1.row(i); ++j; }
                    else { continue; }
                }
            }
        }
        else if (axis == 1) {
            // Column Wise
            if (inverse) {
                X2.resize(X1.rows(), X1.cols() - mask.count());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { continue; }
                    else { X2.col(j) = X1.col(i); ++j; }
                }
            }
            else {
                X2.resize(X1.rows(), mask.count());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { X2.col(j) = X1.col(i); ++j; }
                    else { continue; }
                }
            }
        }
        else { throw std::runtime_error("Axis Out of Bounds"); }
    }
    TMatrix mask_matrix(const TMatrix& X, const BoolVector& mask, const bool& inverse, const int& axis) {
        /*
        * Masks a matrix X to get X2:
        *   -> X2 = X[mask, :] / if (inverse) -> X2 = X[~mask, :] (axis = 0 Row Wise)
        *   -> X2 = X[:, mask] / if (inverse) -> X2 = X[:, ~mask] (axis = 1 Col Wise)
        */
        int i, j;
        TMatrix XM;
        if (axis == 0) {
            // Row Wise
            if (inverse) {
                XM.resize(X.rows() - mask.count(), X.cols());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { continue; }
                    else { XM.row(j) = X.row(i); ++j; }
                }
            }
            else {
                XM.resize(mask.count(), X.cols());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { XM.row(j) = X.row(i); ++j; }
                    else { continue; }
                }
            }
        }
        else if (axis == 1) {
            // Column Wise
            if (inverse) {
                XM.resize(X.rows(), X.cols() - mask.count());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { continue; }
                    else { XM.col(j) = X.col(i); ++j; }
                }
            }
            else {
                XM.resize(X.rows(), mask.count());
                for (i = 0, j = 0; i < mask.size(), j >= 0; ++i) {
                    if (mask(i)) { XM.col(j) = X.col(i); ++j; }
                    else { continue; }
                }
            }
        }
        else { throw std::runtime_error("Axis Out of Bounds"); }
        return XM;
    }

    // Random Utilities
    double random_uniform(double a, double b) {
        auto seed = std::random_device{}();
        std::mt19937 gen_primitive(seed);
        std::uniform_real_distribution<> uniform_dist(a, b);
        return uniform_dist(gen_primitive);
    }
    double random_uniform() {
        auto seed = std::random_device{}();
        std::mt19937 gen_primitive(seed);
        std::uniform_real_distribution<> uniform_dist;
        return uniform_dist(gen_primitive);
    }
    TMatrix gen_normal_matrix(Eigen::Index n_rows, Eigen::Index n_samples) {
        std::normal_distribution<double> normal_sampler{ 0,1 };
        TMatrix norm_matrix = TMatrix::Zero(n_rows, n_samples);
        auto seed = std::random_device{}();
        std::mt19937 gen_primitive(seed);
        visit_lambda(norm_matrix, [&norm_matrix, &normal_sampler, &gen_primitive](double v, int i, int j) {norm_matrix(i, j) = normal_sampler(gen_primitive); });
        return norm_matrix;
    }    

    // Taken from: https://stackoverflow.com/a/40245513
    struct MVN
    {
        MVN(TMatrix const& covar): MVN(TVector::Zero(covar.rows()), covar) {}
        MVN(TVector const& mean, TMatrix const& covar): mean(mean)
        {
            Eigen::SelfAdjointEigenSolver<TMatrix> eigenSolver(covar);
            transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
        }
        TVector operator()() const
        {
            auto seed = std::random_device{}();
            std::mt19937 gen_primitive(seed);
            std::normal_distribution<> dist;
            return mean + transform * TVector{ mean.size() }.unaryExpr([&](auto x) { return dist(gen_primitive); });
        }
        TVector mean;
        TMatrix transform;
    };

}

namespace fdml::utilities::kernel_pca {

    using std::cout;
    using std::endl;
    // Follows sklearn
    class KernelPCA {
    private:
        TMatrix compute_kernel(const TMatrix X0) {
            TMatrix _K(X0.rows(), X0.rows());
            gamma = 1.0 / X0.cols();
            // Sigmoid: k(x, y) = \tanh( \gamma x^\top y + c_0)
            //if (kernel == "sigmoid") {
            _K = X0 * X0.transpose();
            _K.array() *= gamma;
            _K.array() += constant;
            return tanh(_K.array());
            //}
            // Add more kernels
        }
        void center_kernel(TMatrix& K) {
            //  https://github.com/scikit-learn/scikit-learn/blob/e7fb5b8c8dd2cd4d7ccd6f9f9ad6c1d206c43a33/sklearn/preprocessing/_data.py#L2132
            TRVector K_fit_rows = (K.colwise().sum()).array() / K.rows();
            double K_fit_all = K_fit_rows.sum() / K.rows();
            TVector K_pred_cols = (K.rowwise().sum()).array() / K_fit_rows.size();
            K.array().rowwise() -= K_fit_rows.array();
            K.array().colwise() -= K_pred_cols.array();
            K.array() += K_fit_all;
        }

    public:
        KernelPCA(Eigen::Index n_components, std::string kernel) : n_components(n_components), kernel(kernel) {}

        TMatrix transform(const TMatrix& X) {
            gamma = 1.0 / X.cols();
            TMatrix K = compute_kernel(X);
            center_kernel(K);
            // SelfAdjointEigenSolver outputs in ascending order
            Eigen::SelfAdjointEigenSolver<TMatrix> eigsolve(K);
            TVector eigenvalues = eigsolve.eigenvalues();
            TMatrix eigenvectors = eigsolve.eigenvectors();
            // Select eigenvalues/vectors and sort in descending order
            std::vector<Eigen::Index> indices(n_components);
            std::iota(indices.begin(), indices.end(), K.rows() - n_components);
            lambda = eigenvalues(indices).reverse();
            alpha = eigenvectors(Eigen::all, indices).rowwise().reverse();
            // Remove zero eigenvalues
            if (lambda.size() > 1) {
                BoolVector non_zero_index = lambda.array() > 0.0;
                lambda = lambda(non_zero_index);
                alpha = alpha(Eigen::all, non_zero_index);
            }
            // Multiply alpha by -1?
            return alpha * sqrt(lambda.array()).matrix();
        }


    public:
        Eigen::Index n_components;
        std::string kernel;
        TVector lambda;
        TMatrix alpha;
        double gamma = 0.001;
        double constant = 1.0;

    };


}

namespace fdml::utilities::tensor {
    // Taken from : https://stackoverflow.com/questions/48795789/eigen-unsupported-tensor-to-eigen-matrix
    template<typename T>
    using  MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template<typename T>
    using  VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    /* Convert Eigen::Tensor --> Eigen::Matrix */ 

     // Evaluates tensor expressions if needed
    template<typename T, typename Device = Eigen::DefaultDevice>
    auto asEval(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, // An Eigen::TensorBase object (Tensor, TensorMap, TensorExpr... )
        const Device& device = Device()                            // Override to evaluate on another device, e.g. thread pool or gpu.
    ) {
        using Evaluator = Eigen::TensorEvaluator<const Eigen::TensorForcedEvalOp<const T>, Device>;
        Eigen::TensorForcedEvalOp<const T> eval = expr.eval();
        Evaluator                          tensor(eval, device);
        tensor.evalSubExprsIfNeeded(nullptr);
        return tensor;
    }

    // Converts any Eigen::Tensor (or expression) to an Eigen::Matrix with shape rows/cols
    template<typename T, typename sizeType, typename Device = Eigen::DefaultDevice>
    auto MatrixCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, const sizeType rows, const sizeType cols, const Device& device = Device()) {
        auto tensor = asEval(expr, device);
        using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
        return static_cast<MatrixType<Scalar>>(Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols));
    }

    // Converts any Eigen::Tensor (or expression) to an Eigen::Vector with the same size
    template<typename T, typename Device = Eigen::DefaultDevice>
    auto VectorCast(const Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>& expr, const Device& device = Device()) {
        auto tensor = asEval(expr, device);
        auto size = Eigen::internal::array_prod(tensor.dimensions());
        using Scalar = typename Eigen::internal::remove_const<typename decltype(tensor)::Scalar>::type;
        return static_cast<VectorType<Scalar>>(Eigen::Map<const VectorType<Scalar>>(tensor.data(), size));
    }

    // View an existing Eigen::Tensor as an Eigen::Map<Eigen::Matrix>
    template<typename Scalar, auto rank, typename sizeType>
    auto MatrixMap(const Eigen::Tensor<Scalar, rank>& tensor, const sizeType rows, const sizeType cols) {
        return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), rows, cols);
    }

    // View an existing Eigen::Tensor of rank 2 as an Eigen::Map<Eigen::Matrix>
    // Rows/Cols are determined from the matrix
    template<typename Scalar>
    auto MatrixMap(const Eigen::Tensor<Scalar, 2>& tensor) {
        return Eigen::Map<const MatrixType<Scalar>>(tensor.data(), tensor.dimension(0), tensor.dimension(1));
    }

    // View an existing Eigen::Tensor of rank 1 as an Eigen::Map<Eigen::Vector>
    // Rows is the same as the size of the tensor. 
    template<typename Scalar, auto rank>
    auto VectorMap(const Eigen::Tensor<Scalar, rank>& tensor) {
        return Eigen::Map<const VectorType<Scalar>>(tensor.data(), tensor.size());
    }

    /* Convert Eigen::Matrix--> Eigen::Tensor */ 

     // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
     // with dimensions specified in std::array
    template<typename Derived, typename T, auto rank>
    Eigen::Tensor<typename Derived::Scalar, rank>
        TensorCast(const Eigen::EigenBase<Derived>& matrix, const std::array<T, rank>& dims) {
        return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
            (matrix.derived().eval().data(), dims);
    }

    // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
    // with dimensions specified in Eigen::DSizes
    template<typename Derived, typename T, auto rank>
    Eigen::Tensor<typename Derived::Scalar, rank>
        TensorCast(const Eigen::EigenBase<Derived>& matrix, const Eigen::DSizes<T, rank>& dims) {
        return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>
            (matrix.derived().eval().data(), dims);
    }

    // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
    // with dimensions as variadic arguments
    template<typename Derived, typename... Dims>
    auto TensorCast(const Eigen::EigenBase<Derived>& matrix, const Dims... dims) {
        static_assert(sizeof...(Dims) > 0, "TensorCast: sizeof... (Dims) must be larger than 0");
        return TensorCast(matrix, std::array<Eigen::Index, sizeof...(Dims)>{dims...});
    }

    // Converts an Eigen::Matrix (or expression) to Eigen::Tensor
    // with dimensions directly as arguments in a variadic template
    template<typename Derived>
    auto TensorCast(const Eigen::EigenBase<Derived>& matrix) {
        if constexpr (Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
            return TensorCast(matrix, matrix.size());
        }
        else {
            return TensorCast(matrix, matrix.rows(), matrix.cols());
        }
    }

    // View an existing Eigen::Matrix as Eigen::TensorMap
    // with dimensions specified in std::array
    template<typename Derived, auto rank>
    auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix, const std::array<long, rank>& dims) {
        return Eigen::TensorMap<const Eigen::Tensor<const typename Derived::Scalar, rank>>(matrix.derived().data(), dims);
    }

    // View an existing Eigen::Matrix as Eigen::TensorMap
    // with dimensions as variadic arguments
    template<typename Derived, typename... Dims>
    auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix, const Dims... dims) {
        return TensorMap(matrix, std::array<long, static_cast<int>(sizeof...(Dims))>{dims...});
    }

    // View an existing Eigen::Matrix as Eigen::TensorMap
    // with dimensions determined automatically from the given matrix
    template<typename Derived>
    auto TensorMap(const Eigen::PlainObjectBase<Derived>& matrix) {
        if constexpr (Derived::ColsAtCompileTime == 1 or Derived::RowsAtCompileTime == 1) {
            return TensorMap(matrix, matrix.size());
        }
        else {
            return TensorMap(matrix, matrix.rows(), matrix.cols());
        }
    }
}

namespace fdml::utilities::sobol {

    // Frances Y. Kuo
    //
    // Email: <f.kuo@unsw.edu.au>
    // School of Mathematics and Statistics
    // University of New South Wales
    // Sydney NSW 2052, Australia
    // 
    // Last updated: 21 October 2008
    //
    //   You may incorporate this source code into your own program 
    //   provided that you
    //   1) acknowledge the copyright owner in your program and publication
    //   2) notify the copyright owner by email
    //   3) offer feedback regarding your experience with different direction numbers
    //
    //
    // -----------------------------------------------------------------------------
    // Licence pertaining to sobol.cc and the accompanying sets of direction numbers
    // -----------------------------------------------------------------------------
    // Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
    // All rights reserved.
    // 
    // Redistribution and use in source and binary forms, with or without
    // modification, are permitted provided that the following conditions are met:
    // 
    //     * Redistributions of source code must retain the above copyright
    //       notice, this list of conditions and the following disclaimer.
    // 
    //     * Redistributions in binary form must reproduce the above copyright
    //       notice, this list of conditions and the following disclaimer in the
    //       documentation and/or other materials provided with the distribution.
    // 
    //     * Neither the names of the copyright holders nor the names of the
    //       University of New South Wales and the University of Waikato
    //       and its contributors may be used to endorse or promote products derived
    //       from this software without specific prior written permission.
    // 
    // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
    // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    // WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    // DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    // (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    // LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    // ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    // SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    // -----------------------------------------------------------------------------

    using namespace std;

    // ----- SOBOL POINTS GENERATOR BASED ON GRAYCODE ORDER -----------------
    // INPUT: 
    //   N         number of points  (cannot be greater than 2^32)
    //   D         dimension  (make sure that the data file contains enough data!!)      
    //   dir_file  the input file containing direction numbers
    //
    // OUTPUT:
    //   A 2-dimensional array POINTS, where
    //     
    //     POINTS[i][j] = the jth component of the ith point,
    //   
    //   with i indexed from 0 to N-1 and j indexed from 0 to D-1
    //
    // ----------------------------------------------------------------------

    bool replace(std::string& str, const std::string& from, const std::string& to) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos)
            return false;
        str.replace(start_pos, from.length(), to);
        return true;
    }

    double** sobol_points(unsigned N, unsigned D, const std::string filename)
    {
        std::ifstream infile(filename);
        if (!infile) {
            cout << "Input file containing direction numbers cannot be found!\n";
            exit(1);
        }
        char buffer[1000];
        infile.getline(buffer, 1000, '\n');

        // L = max number of bits needed 
        unsigned L = (unsigned)ceil(log((double)N) / log(2.0));

        // C[i] = index from the right of the first zero bit of i
        unsigned* C = new unsigned[N];
        C[0] = 1;
        for (unsigned i = 1; i <= N - 1; i++) {
            C[i] = 1;
            unsigned value = i;
            while (value & 1) {
                value >>= 1;
                C[i]++;
            }
        }

        // POINTS[i][j] = the jth component of the ith point
        //                with i indexed from 0 to N-1 and j indexed from 0 to D-1
        double** POINTS = new double* [N];
        for (unsigned i = 0; i <= N - 1; i++) POINTS[i] = new double[D];
        for (unsigned j = 0; j <= D - 1; j++) POINTS[0][j] = 0;

        // ----- Compute the first dimension -----

        // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
        unsigned* V = new unsigned[L + 1];
        for (unsigned i = 1; i <= L; i++) V[i] = 1 << (32 - i); // all m's = 1

        // Evalulate X[0] to X[N-1], scaled by pow(2,32)
        unsigned* X = new unsigned[N];
        X[0] = 0;
        for (unsigned i = 1; i <= N - 1; i++) {
            X[i] = X[i - 1] ^ V[C[i - 1]];
            POINTS[i][0] = (double)X[i] / pow(2.0, 32); // *** the actual points
            //        ^ 0 for first dimension
        }

        // Clean up
        delete[] V;
        delete[] X;


        // ----- Compute the remaining dimensions -----
        for (unsigned j = 1; j <= D - 1; j++) {

            // Read in parameters from file 
            unsigned d, s;
            unsigned a;
            infile >> d >> s >> a;
            unsigned* m = new unsigned[s + 1];
            for (unsigned i = 1; i <= s; i++) infile >> m[i];

            // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
            unsigned* V = new unsigned[L + 1];
            if (L <= s) {
                for (unsigned i = 1; i <= L; i++) V[i] = m[i] << (32 - i);
            }
            else {
                for (unsigned i = 1; i <= s; i++) V[i] = m[i] << (32 - i);
                for (unsigned i = s + 1; i <= L; i++) {
                    V[i] = V[i - s] ^ (V[i - s] >> s);
                    for (unsigned k = 1; k <= s - 1; k++)
                        V[i] ^= (((a >> (s - 1 - k)) & 1) * V[i - k]);
                }
            }

            // Evalulate X[0] to X[N-1], scaled by pow(2,32)
            unsigned* X = new unsigned[N];
            X[0] = 0;
            for (unsigned i = 1; i <= N - 1; i++) {
                X[i] = X[i - 1] ^ V[C[i - 1]];
                POINTS[i][j] = (double)X[i] / pow(2.0, 32); // *** the actual points
                //        ^ j for dimension (j+1)
            }

            // Clean up
            delete[] m;
            delete[] V;
            delete[] X;
        }
        delete[] C;

        return POINTS;
    }

    TMatrix generate_sobol(unsigned N, unsigned D) {
        std::string sobol_dir(__FILE__);
        replace(sobol_dir, "utilities.h", "sobol_dir.dat");
        //std::cout << sobol_dir << std::endl;
        double** sobol = sobol_points(N, D, sobol_dir);
        TMatrix points = TMatrix::Zero(N, D);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
                points(i, j) = sobol[i][j];
            }
            delete sobol[i];
        }
        return points;
    }



}


#endif