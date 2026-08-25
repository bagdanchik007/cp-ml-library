#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

struct TrainTestSplit {
    Matrix x_train;
    Matrix x_test;
};

inline TrainTestSplit train_test_split(
    const Matrix& data,
    double test_size = 0.2
) {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "train_test_split: input data must not be empty"
        );
    }

    if (test_size <= 0.0 || test_size >= 1.0) {
        throw std::invalid_argument(
            "train_test_split: test_size must be between 0 and 1"
        );
    }

    const size_t test_rows =
        static_cast<size_t>(
            std::ceil(
                static_cast<double>(data.rows) * test_size
            )
        );

    const size_t train_rows = data.rows - test_rows;

    if (train_rows == 0 || test_rows == 0) {
        throw std::invalid_argument(
            "train_test_split: split produced an empty dataset"
        );
    }

    Matrix x_train(train_rows, data.cols);
    Matrix x_test(test_rows, data.cols);

    for (size_t row = 0; row < train_rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            x_train(row, column) = data(row, column);
        }
    }

    for (size_t row = 0; row < test_rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            x_test(row, column) =
                data(train_rows + row, column);
        }
    }

    return {x_train, x_test};
}

} // namespace ml
