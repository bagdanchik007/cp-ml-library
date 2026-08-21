#pragma once

#include "../matrix.hpp"

#include <iomanip>
#include <ostream>

namespace ml {

// ============================================================
// Output
// ============================================================

inline void Matrix::print(
    std::ostream& os,
    int precision
) const {
    os << std::fixed
       << std::setprecision(precision);

    for (size_t i = 0; i < rows; ++i) {
        os << "[ ";

        for (size_t j = 0; j < cols; ++j) {
            os << std::setw(8)
               << (*this)(i, j)
               << " ";
        }

        os << "]\n";
    }
}

} // namespace ml