#pragma once

#include "../matrix.hpp"

#include <stdexcept>

namespace ml {

// ============================================================
// Submatrix extraction
// ============================================================

inline Matrix Matrix::submatrix(
    size_t row_begin,
    size_t row_end,
    size_t column_begin,
    size_t column_end
) const {
    if (row_begin > row_end || column_begin > column_end) {
        throw std::invalid_argument(
            "Matrix submatrix: invalid range"
        );
    }

    if (row_end > rows || column_end > cols) {
        throw std::out_of_range(
            "Matrix submatrix: range out of bounds"
        );
    }

    Matrix result(
        row_end - row_begin,
        column_end - column_begin
    );

    for (size_t i = row_begin; i < row_end; ++i) {
        for (size_t j = column_begin; j < column_end; ++j) {
            result(
                i - row_begin,
                j - column_begin
            ) = (*this)(i, j);
        }
    }

    return result;
}

} // namespace ml