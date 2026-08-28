#pragma once

#include "../matrix.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

// ============================================================
// Matrix transformations
// ============================================================

inline Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

inline void Matrix::reshape(
    size_t new_rows,
    size_t new_cols
) {
    if (new_rows * new_cols != rows * cols) {
        throw std::invalid_argument(
            "Matrix::reshape: new dimensions must contain the same number of elements"
        );
    }

    rows = new_rows;
    cols = new_cols;
}

inline std::vector<double> Matrix::flatten() const {
    return data;
}

} // namespace ml
