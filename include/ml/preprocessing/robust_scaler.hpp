#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ml {

class RobustScaler {
public:
    RobustScaler() = default;

    void fit(
        const Matrix& data
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data
    );

    Matrix inverse_transform(
        const Matrix& data
    ) const;

private:
    Matrix median_;
    Matrix scale_;
    bool fitted_ = false;
};

inline void RobustScaler::fit(
    const Matrix& data
) {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "RobustScaler::fit: input data must not be empty"
        );
    }

    median_ = Matrix(1, data.cols);
    scale_ = Matrix(1, data.cols);

    for (size_t column = 0; column < data.cols; ++column) {
        std::vector<double> values;

        values.reserve(data.rows);

        for (size_t row = 0; row < data.rows; ++row) {
            values.push_back(data(row, column));
        }

        std::sort(
            values.begin(),
            values.end()
        );

        const size_t middle =
            values.size() / 2;

        const double median =
            values.size() % 2 == 0
                ? (values[middle - 1] + values[middle]) / 2.0
                : values[middle];

        const size_t q1_index =
            values.size() / 4;

        const size_t q3_index =
            (values.size() * 3) / 4;

        const double q1 =
            values[q1_index];

        const double q3 =
            values[q3_index];

        const double iqr =
            q3 - q1;

        median_(0, column) = median;

        scale_(0, column) =
            iqr > 1e-12
                ? iqr
                : 1.0;
    }

    fitted_ = true;
}

inline Matrix RobustScaler::transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "RobustScaler::transform: scaler has not been fitted"
        );
    }

    if (data.cols != median_.cols) {
        throw std::invalid_argument(
            "RobustScaler::transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            result(row, column) =
                (data(row, column) - median_(0, column)) /
                scale_(0, column);
        }
    }

    return result;
}

inline Matrix RobustScaler::fit_transform(
    const Matrix& data
) {
    fit(data);

    return transform(data);
}

inline Matrix RobustScaler::inverse_transform(
    const Matrix& data
) const {
    if (!fitted_) {
        throw std::logic_error(
            "RobustScaler::inverse_transform: scaler has not been fitted"
        );
    }

    if (data.cols != median_.cols) {
        throw std::invalid_argument(
            "RobustScaler::inverse_transform: feature count mismatch"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        for (size_t column = 0; column < data.cols; ++column) {
            result(row, column) =
                data(row, column) * scale_(0, column) +
                median_(0, column);
        }
    }

    return result;
}

} // namespace ml