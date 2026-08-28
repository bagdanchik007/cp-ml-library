#pragma once

#include "../matrix.hpp"

#include <random>

namespace ml {

// ============================================================
// Factory functions
// ============================================================

inline Matrix Matrix::zeros(
    size_t r,
    size_t c
) {
    return Matrix(r, c, 0.0);
}

inline Matrix Matrix::ones(
    size_t r,
    size_t c
) {
    return Matrix(r, c, 1.0);
}

inline Matrix Matrix::identity(
    size_t size
) {
    Matrix result(
        size,
        size,
        0.0
    );

    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }

    return result;
}

// ============================================================
// Random matrix
// ============================================================

inline Matrix Matrix::random(
    size_t r,
    size_t c,
    double low,
    double high,
    unsigned seed
) {
    Matrix result(r, c);

    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> dist(
        low,
        high
    );

    for (auto& value : result.data) {
        value = dist(gen);
    }

    return result;
}

} // namespace ml
