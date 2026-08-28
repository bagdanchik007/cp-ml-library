#pragma once

#include "../matrix.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ml {

// ============================================================
// Statistical operations
// ============================================================

inline double Matrix::sum() const {
    return std::accumulate(
        data.begin(),
        data.end(),
        0.0
    );
}

inline double Matrix::mean() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Mean of an empty matrix is undefined"
        );
    }

    return sum() / static_cast<double>(data.size());
}

inline double Matrix::min() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Minimum of an empty matrix is undefined"
        );
    }

    return *std::min_element(
        data.begin(),
        data.end()
    );
}

inline double Matrix::max() const {
    if (data.empty()) {
        throw std::invalid_argument(
            "Maximum of an empty matrix is undefined"
        );
    }

    return *std::max_element(
        data.begin(),
        data.end()
    );
}

// ============================================================
// Matrix norms
// ============================================================

inline double Matrix::squared_norm() const {
    double result = 0.0;

    for (const double value : data) {
        result += value * value;
    }

    return result;
}

inline double Matrix::norm() const {
    return std::sqrt(squared_norm());
}

} // namespace ml
