#pragma once

#include <cstddef>
#include <stdexcept>

namespace ml::detail {

inline void validate_same_shape(
    std::size_t lhs_rows,
    std::size_t lhs_cols,
    std::size_t rhs_rows,
    std::size_t rhs_cols
) {
    if (lhs_rows != rhs_rows || lhs_cols != rhs_cols) {
        throw std::invalid_argument(
            "Matrix dimensions must match"
        );
    }
}

inline void validate_multiplication_dimensions(
    std::size_t lhs_cols,
    std::size_t rhs_rows
) {
    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication"
        );
    }
}

inline void validate_square_matrix(
    std::size_t rows,
    std::size_t cols
) {
    if (rows != cols) {
        throw std::invalid_argument(
            "Matrix must be square"
        );
    }
}

inline void validate_index(
    std::size_t row,
    std::size_t col,
    std::size_t rows,
    std::size_t cols
) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range(
            "Matrix index out of range"
        );
    }
}

inline void validate_row_index(
    std::size_t row,
    std::size_t rows
) {
    if (row >= rows) {
        throw std::out_of_range(
            "Matrix row index out of range"
        );
    }
}

inline void validate_column_index(
    std::size_t col,
    std::size_t cols
) {
    if (col >= cols) {
        throw std::out_of_range(
            "Matrix column index out of range"
        );
    }
}

} // namespace ml::detail