#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

class StandardScaler {
public:
    StandardScaler() = default;

    void fit(const Matrix& data);

    Matrix transform(const Matrix& data) const;

    Matrix fit_transform(const Matrix& data);

    Matrix inverse_transform(const Matrix& data) const;

private:
    Matrix mean_;
    Matrix scale_;
    bool fitted_ = false;
};

inline void StandardScaler::fit(const Matrix& data) {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "StandardScaler::fit: input data must not be empty"
        );
    }

    mean_ = Matrix(1, data.cols);
    scale_ = Matrix(1, data.cols);

    for (size_t column = 0; column < data.cols; ++column) {
        double mean = 0.0;

        for (size_t row = 0; row < data.rows; ++row) {
            mean += data(row, column);
        }

        mean /= static_cast<double>(data.rows);
        mean_(0, column) = mean;

        double variance = 0.0;

        for (size_t row = 0; row < data.rows; ++row) {
            const double diff = data(row, column) - mean;
            variance += diff * diff;
        }

        variance /= static_cast<double>(data.rows);

        const double standard_deviation = std::sqrt(variance);

        scale_(0, column) =
            standard_deviation > 1e-12
                ? standard_deviation
                : 1.0;
    }

    fitted_ = true;
}

inline Matrix StandardScaler::transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "StandardScaler::transform: scaler has not been fitted"
        );
    }

    if (data.cols != mean_.cols) {
        throw std::invalid_argument(
            "StandardScaler::transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            result(row, column) =
                (data(row, column) - mean_(0, column)) /
                scale_(0, column);
        }
    }

    return result;
}

inline Matrix StandardScaler::fit_transform(
    const Matrix& data
) {
    fit(data);
    return transform(data);
}

inline Matrix StandardScaler::inverse_transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "StandardScaler::inverse_transform: scaler has not been fitted"
        );
    }

    if (data.cols != mean_.cols) {
        throw std::invalid_argument(
            "StandardScaler::inverse_transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            result(row, column) =
                data(row, column) * scale_(0, column) +
                mean_(0, column);
        }
    }

    return result;
}

} // namespace ml