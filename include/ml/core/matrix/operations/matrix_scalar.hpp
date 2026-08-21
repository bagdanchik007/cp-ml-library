#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

// ============================================================
// Scalar multiplication
// ============================================================

inline Matrix Matrix::operator*(
    double scalar
) const {
    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] * scalar;
    }

    return result;
}

// ============================================================
// Scalar division
// ============================================================

inline Matrix Matrix::operator/(
    double scalar
) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument(
            "Division by zero"
        );
    }

    Matrix result(rows, cols);

    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] =
            data[i] / scalar;
    }

    return result;
}

// ============================================================
// In-place scalar multiplication
// ============================================================

inline Matrix& Matrix::operator*=(
    double scalar
) {
    for (auto& value : data) {
        value *= scalar;
    }

    return *this;
}

// ============================================================
// In-place scalar division
// ============================================================

inline Matrix& Matrix::operator/=(
    double scalar
) {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument(
            "Division by zero"
        );
    }

    for (auto& value : data) {
        value /= scalar;
    }

    return *this;
}

} // namespace ml
