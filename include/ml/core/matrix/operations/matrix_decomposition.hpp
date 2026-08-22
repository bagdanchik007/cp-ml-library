#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ml {

// ============================================================
// LU decomposition
// ============================================================

inline std::pair<Matrix, Matrix> Matrix::lu_decomposition() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "LU decomposition requires a square matrix"
        );
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "LU decomposition of an empty matrix is undefined"
        );
    }

    Matrix lower = Matrix::identity(rows);
    Matrix upper(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        // Compute upper triangular matrix.
        for (size_t k = i; k < cols; ++k) {
            double sum = 0.0;

            for (size_t j = 0; j < i; ++j) {
                sum += lower(i, j) * upper(j, k);
            }

            upper(i, k) = (*this)(i, k) - sum;
        }

        if (std::abs(upper(i, i)) < 1e-12) {
            throw std::invalid_argument(
                "Matrix cannot be decomposed without pivoting"
            );
        }

        // Compute lower triangular matrix.
        for (size_t k = i + 1; k < rows; ++k) {
            double sum = 0.0;

            for (size_t j = 0; j < i; ++j) {
                sum += lower(k, j) * upper(j, i);
            }

            lower(k, i) =
                ((*this)(k, i) - sum) / upper(i, i);
        }
    }

    return {lower, upper};
}

} // namespace ml