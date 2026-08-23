#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

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
    Matrix upper(rows, cols, 0.0);

    for (size_t i = 0; i < rows; ++i) {

        // Compute the upper triangular matrix.
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

        // Compute the lower triangular matrix.
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

// ============================================================
// Linear system solver
// ============================================================

inline std::vector<double> Matrix::solve(
    const std::vector<double>& b
) const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Solving a linear system requires a square matrix"
        );
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "Cannot solve a linear system with an empty matrix"
        );
    }

    if (b.size() != rows) {
        throw std::invalid_argument(
            "Right-hand side vector size must match matrix rows"
        );
    }

    const auto [lower, upper] = lu_decomposition();

    std::vector<double> y(rows, 0.0);
    std::vector<double> x(rows, 0.0);

    // ========================================================
    // Forward substitution: Ly = b
    // ========================================================

    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;

        for (size_t j = 0; j < i; ++j) {
            sum += lower(i, j) * y[j];
        }

        if (std::abs(lower(i, i)) < 1e-12) {
            throw std::invalid_argument(
                "Matrix decomposition produced a singular lower matrix"
            );
        }

        y[i] =
            (b[i] - sum) / lower(i, i);
    }

    // ========================================================
    // Backward substitution: Ux = y
    // ========================================================

    for (size_t i = rows; i-- > 0;) {
        double sum = 0.0;

        for (size_t j = i + 1; j < cols; ++j) {
            sum += upper(i, j) * x[j];
        }

        if (std::abs(upper(i, i)) < 1e-12) {
            throw std::invalid_argument(
                "Matrix is singular and cannot be solved"
            );
        }

        x[i] =
            (y[i] - sum) / upper(i, i);
    }

    return x;
}

} // namespace ml