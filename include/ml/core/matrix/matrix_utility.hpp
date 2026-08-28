#pragma once

/**
 * @file matrix_utility.hpp
 * @brief Free utility functions related to Matrix.
 */

#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ml {

/**
 * @brief Euclidean (L2) distance between two vectors.
 */
inline double euclidean_distance(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have the same dimension"
        );
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

} // namespace ml
