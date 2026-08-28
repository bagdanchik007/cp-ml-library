#pragma once

#include "../matrix.hpp"

#include <stdexcept>
#include <vector>

namespace ml {

// ============================================================
// Matrix-vector multiplication
// ============================================================

inline std::vector<double> Matrix::operator*(
    const std::vector<double>& v
) const {
    if (cols != v.size()) {
        throw std::invalid_argument(
            "Matrix-vector dimension mismatch"
        );
    }

    std::vector<double> result(
        rows,
        0.0
    );

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] +=
                (*this)(i, j) * v[j];
        }
    }

    return result;
}

// ============================================================
// Matrix addition
// ============================================================

inline Matrix Matrix::operator+(
    const Matrix& other
) const {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for addition"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] + other.data[i];
    }

    return result;
}

// ============================================================
// Matrix subtraction
// ============================================================

inline Matrix Matrix::operator-(
    const Matrix& other
) const {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for subtraction"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] - other.data[i];
    }

    return result;
}

// ============================================================
// In-place addition
// ============================================================

inline Matrix& Matrix::operator+=(
    const Matrix& other
) {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for addition"
        );
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }

    return *this;
}

// ============================================================
// In-place subtraction
// ============================================================

inline Matrix& Matrix::operator-=(
    const Matrix& other
) {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for subtraction"
        );
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }

    return *this;
}

// ============================================================
// Matrix multiplication
// ============================================================

inline Matrix Matrix::operator*(
    const Matrix& other
) const {
    if (cols != other.rows) {
        throw std::invalid_argument(
            "Matrix dimensions incompatible for multiplication"
        );
    }

    Matrix result(
        rows,
        other.cols,
        0.0
    );

    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < cols; ++k) {
            const double value =
                (*this)(i, k);

            for (size_t j = 0; j < other.cols; ++j) {
                result(i, j) +=
                    value * other(k, j);
            }
        }
    }

    return result;
}

// ============================================================
// Hadamard product (element-wise multiplication)
// ============================================================

inline Matrix Matrix::hadamard(
    const Matrix& other
) const {
    if (
        rows != other.rows ||
        cols != other.cols
    ) {
        throw std::invalid_argument(
            "Matrix dimensions must match for Hadamard product"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] * other.data[i];
    }

    return result;
}

} // namespace ml
