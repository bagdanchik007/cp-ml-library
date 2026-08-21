#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

// ============================================================
// Determinant
// ============================================================

inline double Matrix::determinant() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Determinant requires a square matrix"
        );
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "Determinant of an empty matrix is undefined"
        );
    }

    if (rows == 1) {
        return (*this)(0, 0);
    }

    if (rows == 2) {
        return
            (*this)(0, 0) * (*this)(1, 1) -
            (*this)(0, 1) * (*this)(1, 0);
    }

    double result = 0.0;

    for (size_t column = 0; column < cols; ++column) {
        Matrix minor(rows - 1, cols - 1);

        for (size_t i = 1; i < rows; ++i) {
            size_t minor_column = 0;

            for (size_t j = 0; j < cols; ++j) {
                if (j == column) {
                    continue;
                }

                minor(i - 1, minor_column) =
                    (*this)(i, j);

                ++minor_column;
            }
        }

        const double sign =
            (column % 2 == 0) ? 1.0 : -1.0;

        result +=
            sign *
            (*this)(0, column) *
            minor.determinant();
    }

    return result;
}

// ============================================================
// Trace
// ============================================================

inline double Matrix::trace() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Trace requires a square matrix"
        );
    }

    double result = 0.0;

    for (size_t i = 0; i < rows; ++i) {
        result += (*this)(i, i);
    }

    return result;
}

// ============================================================
// Matrix inverse
// ============================================================

inline Matrix Matrix::inverse() const {
    if (rows != cols) {
        throw std::invalid_argument(
            "Inverse requires a square matrix"
        );
    }

    if (rows == 0) {
        throw std::invalid_argument(
            "Inverse of an empty matrix is undefined"
        );
    }

    Matrix augmented(rows, cols * 2);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            augmented(i, j) = (*this)(i, j);

            augmented(i, j + cols) =
                (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < rows; ++i) {
        size_t pivot_row = i;

        for (size_t row = i + 1; row < rows; ++row) {
            if (
                std::abs(augmented(row, i)) >
                std::abs(augmented(pivot_row, i))
            ) {
                pivot_row = row;
            }
        }

        if (
            std::abs(augmented(pivot_row, i)) <
            1e-12
        ) {
            throw std::invalid_argument(
                "Matrix is singular and cannot be inverted"
            );
        }

        if (pivot_row != i) {
            for (size_t j = 0; j < augmented.cols; ++j) {
                std::swap(
                    augmented(i, j),
                    augmented(pivot_row, j)
                );
            }
        }

        const double pivot = augmented(i, i);

        for (size_t j = 0; j < augmented.cols; ++j) {
            augmented(i, j) /= pivot;
        }

        for (size_t row = 0; row < rows; ++row) {
            if (row == i) {
                continue;
            }

            const double factor = augmented(row, i);

            for (size_t j = 0; j < augmented.cols; ++j) {
                augmented(row, j) -=
                    factor * augmented(i, j);
            }
        }
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) =
                augmented(i, j + cols);
        }
    }

    return result;
}

} // namespace ml
