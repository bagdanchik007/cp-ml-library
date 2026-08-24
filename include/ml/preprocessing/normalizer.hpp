#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

class Normalizer {
public:
    Normalizer() = default;

    Matrix transform(
        const Matrix& data
    ) const;
};

inline Matrix Normalizer::transform(
    const Matrix& data
) const {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "Normalizer::transform: input data must not be empty"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        double squared_norm = 0.0;

        for (size_t column = 0; column < data.cols; ++column) {
            const double value = data(row, column);

            squared_norm += value * value;
        }

        const double row_norm =
            std::sqrt(squared_norm);

        if (row_norm <= 1e-12) {
            continue;
        }

        for (size_t column = 0; column < data.cols; ++column) {
            result(row, column) =
                data(row, column) / row_norm;
        }
    }

    return result;
}

} // namespace ml