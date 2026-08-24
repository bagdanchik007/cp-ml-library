#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <utility>

namespace ml {

struct TrainTestSplit {
    Matrix x_train;
    Matrix x_test;
};

TrainTestSplit train_test_split(
    const Matrix& data,
    double test_size = 0.2
);

} // namespace ml