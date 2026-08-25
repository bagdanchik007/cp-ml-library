#pragma once

#include "ml/core/matrix/matrix.hpp"

namespace ml {

class Binarizer {
public:
    explicit Binarizer(
        double threshold = 0.0
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data
    ) const;

private:
    double threshold_;
};

} // namespace ml