#pragma once

#include "ml/core/matrix/matrix.hpp"

namespace ml {

class MaxAbsScaler {
public:
    MaxAbsScaler() = default;

    void fit(
        const Matrix& data
    );

    Matrix transform(
        const Matrix& data
    ) const;

    Matrix fit_transform(
        const Matrix& data
    );

private:
    Matrix max_abs_values_;
    bool fitted_ = false;
};

} // namespace ml