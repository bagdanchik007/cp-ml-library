#pragma once

#include "../matrix.hpp"

#include <cmath>
#include <iostream>

namespace ml {

inline bool Matrix::operator==(const Matrix& other) const {
    constexpr double epsilon = 1e-9;

    if (rows != other.rows || cols != other.cols) {
        return false;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        const double difference = std::abs(data[i] - other.data[i]);

        std::cerr
            << "comparison: "
            << data[i]
            << " vs "
            << other.data[i]
            << " diff="
            << difference
            << " epsilon="
            << epsilon
            << '\n';

        if (difference > epsilon) {
            return false;
        }
    }

    return true;
}

inline bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

} // namespace ml