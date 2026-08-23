#pragma once

#include "ml/core/matrix/matrix.hpp"

namespace ml {

class StandardScaler {
public:
    StandardScaler() = default;

    void fit(const Matrix& data);

    Matrix transform(const Matrix& data) const;

    Matrix fit_transform(const Matrix& data);

    Matrix inverse_transform(const Matrix& data) const;

private:
    Matrix mean_;
    Matrix scale_;
    bool fitted_ = false;
};

} // namespace ml
