#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ml {

// ============================================================
// LU decomposition with partial pivoting
// ============================================================

inline LUDecomposition Matrix::lu_decomposition() const {
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
    Matrix upper = *this;
    Matrix permutation = Matrix::identity(rows);

    constexpr double epsilon = 1e-12;

    for (size_t i = 0; i < rows; ++i) {

        // ========================================================
        // Find pivot
        // ========================================================

        size_t pivot_row = i;
        double pivot_value = std::abs(upper(i, i));

        for (size_t k = i + 1; k < rows; ++k) {
            const double value =
                std::abs(upper(k, i));

            if (value > pivot_value) {
                pivot_value = value;
                pivot_row = k;
            }
        }

        if (pivot_value < epsilon) {
            throw std::invalid_argument(
                "Matrix is singular and cannot be decomposed"
            );
        }

        // ========================================================
        // Swap rows if necessary
        // ========================================================

        if (pivot_row != i) {
            upper.swap_rows(i, pivot_row);
            permutation.swap_rows(i, pivot_row);

            // Previously calculated values in L must also
            // be swapped.
            for (size_t j = 0; j < i; ++j) {
                std::swap(
                    lower(i, j),
                    lower(pivot_row, j)
                );
            }
        }

        // ========================================================
        // Compute lower matrix
        // ========================================================

        for (size_t k = i + 1; k < rows; ++k) {
            lower(k, i) =
                upper(k, i) / upper(i, i);

            for (size_t j = i; j < cols; ++j) {
                upper(k, j) -=
                    lower(k, i) * upper(i, j);
            }

            upper(k, i) = 0.0;
        }
    }

    return {
        lower,
        upper,
        permutation
    };
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

    const LUDecomposition decomposition =
        lu_decomposition();

    const Matrix& lower =
        decomposition.lower;

    const Matrix& upper =
        decomposition.upper;

    const Matrix& permutation =
        decomposition.permutation;

    // ========================================================
    // Apply permutation: Pb
    // ========================================================

    std::vector<double> permuted_b(rows, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            permuted_b[i] +=
                permutation(i, j) * b[j];
        }
    }

    // ========================================================
    // Forward substitution: Ly = Pb
    // ========================================================

    std::vector<double> y(rows, 0.0);

    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;

        for (size_t j = 0; j < i; ++j) {
            sum +=
                lower(i, j) * y[j];
        }

        y[i] =
            permuted_b[i] - sum;
    }

    // ========================================================
    // Backward substitution: Ux = y
    // ========================================================

    std::vector<double> x(rows, 0.0);

    constexpr double epsilon = 1e-12;

    for (size_t i = rows; i-- > 0;) {
        double sum = 0.0;

        for (size_t j = i + 1; j < cols; ++j) {
            sum +=
                upper(i, j) * x[j];
        }

        if (std::abs(upper(i, i)) < epsilon) {
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