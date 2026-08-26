#pragma once

#include "ml/core/matrix/matrix.hpp"

#include <cstddef>
#include <vector>

namespace ml {

class SelectPercentile {
public:
    explicit SelectPercentile(double percentile);

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

    const std::vector<double>& scores() const;
    const std::vector<std::size_t>& selected_features() const;

private:
    double percentile_;

    std::vector<double> scores_;
    std::vector<std::size_t> selected_features_;

    bool fitted_ = false;
};

} // namespace ml