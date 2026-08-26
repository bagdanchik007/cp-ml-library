#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <vector>

namespace ml {

struct MutualInfoResult {
    std::vector<double> scores;
};

MutualInfoResult mutual_info(
    const Matrix& X,
    const Matrix& y
);

} // namespace ml
