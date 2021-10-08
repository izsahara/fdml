#ifndef INCLUDE_BASE_OPT
#define INCLUDE_BASE_OPT
#include <cmath>
#include <optional>
#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include <utility>
#include <Eigen/Core>

static inline const int Dim = Eigen::Dynamic;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> TVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> TMatrix;
typedef Eigen::Matrix<double, Dim, Dim> THessian;
typedef TVector::Index TIndex;



#endif