#pragma once

#include "../matrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ml {

// ============================================================
// Element access
// ============================================================

inline double& Matrix::operator()(
    size_t i,
    size_t j
) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range(
            "Matrix index out of bounds"
        );
    }

    return data[i * cols + j];
}

inline const double& Matrix::operator()(
    size_t i,
    size_t j
) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range(
            "Matrix index out of bounds"
        );
    }

    return data[i * cols + j];
}

// ============================================================
// Row operations
// ============================================================

inline std::vector<double> Matrix::row(
    size_t i
) const {
    if (i >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    return std::vector<double>(
        data.begin() + i * cols,
        data.begin() + (i + 1) * cols
    );
}

inline void Matrix::set_row(
    size_t i,
    const std::vector<double>& values
) {
    if (i >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    if (values.size() != cols) {
        throw std::invalid_argument(
            "Row size mismatch"
        );
    }

    std::copy(
        values.begin(),
        values.end(),
        data.begin() + i * cols
    );
}

// ============================================================
// Row swapping
// ============================================================

inline void Matrix::swap_rows(
    size_t first,
    size_t second
) {
    if (first >= rows || second >= rows) {
        throw std::out_of_range(
            "Matrix row index out of bounds"
        );
    }

    if (first == second) {
        return;
    }

    for (size_t j = 0; j < cols; ++j) {
        std::swap(
            (*this)(first, j),
            (*this)(second, j)
        );
    }
}

// ============================================================
// Column operations
// ============================================================

inline std::vector<double> Matrix::column(
    size_t j
) const {
    if (j >= cols) {
        throw std::out_of_range(
            "Matrix column index out of bounds"
        );
    }

    std::vector<double> result;
    result.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        result.push_back((*this)(i, j));
    }

    return result;
}

inline void Matrix::set_column(
    size_t j,
    const std::vector<double>& values
) {
    if (j >= cols) {
        throw std::out_of_range(
            "Matrix column index out of bounds"
        );
    }

    if (values.size() != rows) {
        throw std::invalid_argument(
            "Column size mismatch"
        );
    }

    for (size_t i = 0; i < rows; ++i) {
        (*this)(i, j) = values[i];
    }
}

// ============================================================
// Column swapping
// ============================================================

inline void Matrix::swap_columns(
    size_t first,
    size_t second
) {
    if (first >= cols || second >= cols) {
        throw std::out_of_range(
            "Matrix column index out of bounds"
        );
    }

    if (first == second) {
        return;
    }

    for (size_t i = 0; i < rows; ++i) {
        std::swap(
            (*this)(i, first),
            (*this)(i, second)
        );
    }
}

} // namespace ml
