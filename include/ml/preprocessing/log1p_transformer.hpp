#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

/**
 * @brief Applies log(1 + x) element-wise, useful for compressing
 * right-skewed, non-negative features such as counts or prices.
 *
 * transform() throws if any value would make (1 + x) non-positive,
 * since the natural logarithm is undefined there.
 */
class Log1pTransformer {
public:
    Matrix transform(const Matrix& data) const {
        Matrix result(data.rows, data.cols);
        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                const double value = data(row, column);
                if (1.0 + value <= 0.0) {
                    throw std::invalid_argument("Log1pTransformer::transform: value out of domain (1 + x must be positive)");
                }
                result(row, column) = std::log1p(value);
            }
        }
        return result;
    }

    Matrix inverse_transform(const Matrix& data) const {
        Matrix result(data.rows, data.cols);
        for (size_t row = 0; row < data.rows; ++row) {
            for (size_t column = 0; column < data.cols; ++column) {
                result(row, column) = std::expm1(data(row, column));
            }
        }
        return result;
    }
};

} // namespace ml
