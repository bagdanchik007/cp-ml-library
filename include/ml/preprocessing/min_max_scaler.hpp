#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

class MinMaxScaler {
public:
    MinMaxScaler() = default;

    void fit(const Matrix& data);

    Matrix transform(const Matrix& data) const;

    Matrix fit_transform(const Matrix& data);

    Matrix inverse_transform(const Matrix& data) const;

private:
    Matrix min_;
    Matrix max_;
    bool fitted_ = false;
};

inline void MinMaxScaler::fit(const Matrix& data) {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "MinMaxScaler::fit: input data must not be empty"
        );
    }

    min_ = Matrix(1, data.cols);
    max_ = Matrix(1, data.cols);

    for (size_t column = 0; column < data.cols; ++column) {
        double minimum = data(0, column);
        double maximum = data(0, column);

        for (size_t row = 1; row < data.rows; ++row) {
            minimum = std::min(
                minimum,
                data(row, column)
            );

            maximum = std::max(
                maximum,
                data(row, column)
            );
        }

        min_(0, column) = minimum;
        max_(0, column) = maximum;
    }

    fitted_ = true;
}

inline Matrix MinMaxScaler::transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "MinMaxScaler::transform: scaler has not been fitted"
        );
    }

    if (data.cols != min_.cols) {
        throw std::invalid_argument(
            "MinMaxScaler::transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            const double range =
                max_(0, column) - min_(0, column);

            result(row, column) =
                std::abs(range) > 1e-12
                    ? (data(row, column) - min_(0, column)) / range
                    : 0.0;
        }
    }

    return result;
}

inline Matrix MinMaxScaler::fit_transform(
    const Matrix& data
) {
    fit(data);

    return transform(data);
}

inline Matrix MinMaxScaler::inverse_transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "MinMaxScaler::inverse_transform: scaler has not been fitted"
        );
    }

    if (data.cols != min_.cols) {
        throw std::invalid_argument(
            "MinMaxScaler::inverse_transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            const double range =
                max_(0, column) - min_(0, column);

            result(row, column) =
                data(row, column) * range +
                min_(0, column);
        }
    }

    return result;
}

} // namespace ml