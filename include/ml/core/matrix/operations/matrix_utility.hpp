#pragma once

#include "../matrix.hpp"

namespace ml {

inline bool Matrix::empty() const {
    return data.empty();
}

inline size_t Matrix::size() const {
    return data.size();
}

inline std::pair<size_t, size_t> Matrix::shape() const {
    return {rows, cols};
}

inline void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

inline void Matrix::clear() {
    rows = 0;
    cols = 0;
    data.clear();
}

inline void Matrix::resize(size_t r, size_t c, double fill) {
    std::vector<double> new_data(r * c, fill);

    const size_t copy_rows = std::min(rows, r);
    const size_t copy_cols = std::min(cols, c);

    for (size_t i = 0; i < copy_rows; ++i) {
        for (size_t j = 0; j < copy_cols; ++j) {
            new_data[i * c + j] = data[i * cols + j];
        }
    }

    rows = r;
    cols = c;
    data = std::move(new_data);
}

} // namespace ml
