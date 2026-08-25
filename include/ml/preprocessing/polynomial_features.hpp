#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>

namespace ml {

class PolynomialFeatures {
public:
    explicit PolynomialFeatures(
        size_t degree = 2
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data
    ) const;

private:
    size_t degree_;
};

} // namespace ml