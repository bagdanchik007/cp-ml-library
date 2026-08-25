#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

struct FRegressionResult {
    std::vector<double> scores;
    std::vector<double> p_values;
};

FRegressionResult f_regression(
    const Matrix& X,
    const Matrix& y
);

} // namespace ml