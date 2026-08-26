#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <vector>

namespace ml {

struct Chi2Result {
    std::vector<double> scores;
    std::vector<double> p_values;
};

Chi2Result chi2(
    const Matrix& X,
    const Matrix& y
);

} // namespace ml
