#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace ml {

enum class Norm {
    L1,
    L2,
    Max
};

class Normalizer {
public:
    Normalizer() = default;

    Matrix transform(
        const Matrix& data,
        Norm norm = Norm::L2
    ) const;
};

inline Matrix Normalizer::transform(
    const Matrix& data,
    Norm norm
) const {
    if (data.rows == 0 || data.cols == 0) {
        throw std::invalid_argument(
            "Normalizer::transform: input data must not be empty"
        );
    }

    Matrix result(data.rows, data.cols);

    for (size_t row = 0; row < data.rows; ++row) {
        double row_norm = 0.0;

        switch (norm) {
        case Norm::L1:
            for (size_t column = 0; column < data.cols; ++column) {
                row_norm += std::abs(data(row, column));
            }
            break;

        case Norm::L2:
            for (size_t column = 0; column < data.cols; ++column) {
                const double value = data(row, column);
                row_norm += value * value;
            }

            row_norm = std::sqrt(row_norm);
            break;

        case Norm::Max:
            for (size_t column = 0; column < data.cols; ++column) {
                row_norm = std::max(
                    row_norm,
                    std::abs(data(row, column))
                );
            }
            break;
        }

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