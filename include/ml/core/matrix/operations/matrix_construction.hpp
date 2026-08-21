#pragma once

#include "../matrix.hpp"

#include <stdexcept>

namespace ml {

// ============================================================
// Constructors
// ============================================================

inline Matrix::Matrix(
    size_t r,
    size_t c,
    double fill
)
    : rows(r),
      cols(c),
      data(r * c, fill) {
}

inline Matrix::Matrix(
    std::initializer_list<std::initializer_list<double>> init
) {
    rows = init.size();

    if (rows == 0) {
        cols = 0;
        return;
    }

    cols = init.begin()->size();

    data.reserve(rows * cols);

    for (const auto& row : init) {
        if (row.size() != cols) {
            throw std::invalid_argument(
                "All rows must have the same length"
            );
        }

        data.insert(
            data.end(),
            row.begin(),
            row.end()
        );
    }
}

} // namespace ml
