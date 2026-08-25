#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

class SelectFpr {
public:
    explicit SelectFpr(double alpha);

    void fit(
        const Matrix& X,
        const Matrix& y
    );

    Matrix transform(
        const Matrix& X
    ) const;

    Matrix fit_transform(
        const Matrix& X,
        const Matrix& y
    );

    const std::vector<std::size_t>&
    selected_features() const noexcept;

private:
    double alpha_;
    std::vector<std::size_t> selected_features_;
};

} // namespace ml
