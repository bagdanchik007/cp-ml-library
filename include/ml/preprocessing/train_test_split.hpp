#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {

struct TrainTestSplit {
    Matrix x_train;
    Matrix x_test;
};

inline TrainTestSplit train_test_split(
    const Matrix& data,
    double test_size = 0.2,
    bool shuffle = true,
    unsigned int seed = 42
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

    const size_t train_rows =
        data.rows - test_rows;

    if (train_rows == 0 || test_rows == 0) {
        throw std::invalid_argument(
            "train_test_split: split produced an empty dataset"
        );
    }

    std::vector<size_t> indices(data.rows);

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

    if (shuffle) {
        std::mt19937 generator(seed);

        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );
    }

    Matrix x_train(train_rows, data.cols);
    Matrix x_test(test_rows, data.cols);

    for (size_t row = 0; row < train_rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            x_train(row, column) =
                data(indices[row], column);
        }
    }

    for (size_t row = 0; row < test_rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            x_test(row, column) =
                data(indices[train_rows + row], column);
        }
    }

    return {x_train, x_test};
}

} // namespace ml