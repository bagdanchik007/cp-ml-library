#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

class VarianceThreshold {
public:
    explicit VarianceThreshold(
        double threshold = 0.0
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

    const std::vector<double>& variances() const;

private:
    double threshold_;
    std::vector<double> variances_;
    std::vector<size_t> selected_columns_;
    bool fitted_ = false;
};

} // namespace ml