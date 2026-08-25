#pragma once

#include "ml/core/matrix/matrix.hpp"

namespace ml {

enum class ImputationStrategy {
    Mean,
    Median,
    Constant
};

class SimpleImputer {
public:
    explicit SimpleImputer(
        ImputationStrategy strategy =
            ImputationStrategy::Mean,
        double fill_value = 0.0
    );

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
    ImputationStrategy strategy_;
    double fill_value_;
    bool fitted_ = false;
};

} // namespace ml
